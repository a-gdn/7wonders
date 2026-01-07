"""
AlphaZero Training Script for 7 Wonders

This script implements a self-play reinforcement learning loop based on AlphaZero:
1. Self-Play: Agents play against copies of themselves using MCTS to generate data.
2. Training: Network is trained to minimize error between MCTS policy/value and network predictions.
3. Arena: Periodically evaluates the new model against the best previous model.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, optimizers
import random
from typing import List, Dict, Tuple
import copy
import subprocess
import platform
import math

from seven_wonders.environment import GameEnv
from seven_wonders.constants import CardColor

# --- Hyperparameters ---
ITERATIONS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS_PER_ITERATION = 5
SELF_PLAY_GAMES = 15    # Fewer games because MCTS is slower
ARENA_GAMES = 40        # Evaluation games
EVAL_INTERVAL = 2       # Evaluate frequently
NUM_PLAYERS = 4

# MCTS Hyperparameters
MCTS_SIMS = 300          # Simulations per move (Increase for stronger play, decrease for speed)
C_PUCT = 1.5            # Exploration constant
DIRICHLET_ALPHA = 0.3   # Noise for root exploration
DIRICHLET_EPSILON = 0.25
BUFFER_SIZE = 10000     # Replay buffer size

class DataProcessor:
    """
    Handles conversion between Environment Observations/Actions and Neural Network Tensors.
    """
    def __init__(self, env: GameEnv):
        self.card_to_id = {}
        self.id_to_card = {}
        self.card_colors = {}
        self.wonder_to_id = {}
        self._build_vocab(env)
        
        self.num_cards = len(self.card_to_id)
        self.num_wonders = len(self.wonder_to_id)
        self.action_space_size = self.num_cards * 3
        
        # Initialize Logit Bias Vector for Guided Exploration
        self.logit_bias = np.zeros(self.action_space_size, dtype=np.float32)
        self._init_logit_bias()
        
        # Input Dimension Calculation (Raw Data Philosophy)
        # Global: Age(1) + Turn(1) = 2
        # Per Player (Self + Opponents):
        #   Wonder Name (One-hot): num_wonders
        #   Wonder Side (One-hot): 2 (Day/Night)
        #   Wonder Stage: 1
        #   Coins: 1
        #   Production: 7
        #   Science: 3
        #   Shields: 1
        #   Military Score: 1
        #   Built Cards (Multi-hot): num_cards
        self.per_player_dim = self.num_wonders + 2 + 1 + 1 + 7 + 3 + 1 + 1 + self.num_cards
        
        # Self Only (Private): Hand(num_cards) + Memory(num_cards)
        # + Action Mask (action_space_size) to explicitly tell the network what is legal
        self.input_dim = 2 + (env.num_players * self.per_player_dim) + (2 * self.num_cards) + self.action_space_size
        
    def _build_vocab(self, env: GameEnv):
        """Build vocabulary from all possible cards in the game database."""
        # Extract all unique card names from the loaded data
        unique_names = set()
        for card_data in env.cards_data:
            unique_names.add(card_data["name"])
            self.card_colors[card_data["name"]] = card_data["color"]
        
        # Create mappings
        sorted_names = sorted(list(unique_names))
        for idx, name in enumerate(sorted_names):
            self.card_to_id[name] = idx
            self.id_to_card[idx] = name
            
        # Build Wonder vocabulary
        unique_wonders = set()
        for w in env.wonder_data:
            unique_wonders.add(w["name"])
        self.wonder_to_id = {name: i for i, name in enumerate(sorted(list(unique_wonders)))}
            
        print(f"DataProcessor: Vocab size {len(self.card_to_id)} cards.")

    def _init_logit_bias(self):
        """Create a static bias vector to guide exploration."""
        N = self.num_cards
        
        # 1. Strong Penalty for Discards (Indices 2N to 3N)
        # We want the agent to build whenever possible.
        # Masking handles legality, so this just discourages voluntary discarding.
        self.logit_bias[2*N : 3*N] = -5.0
        
        # 2. Encourage Strategic Cards (Green/Purple) and Wonders
        for name, idx in self.card_to_id.items():
            color = self.card_colors.get(name, "")
            
            # Build Structure (0..N-1)
            if color == "green":
                self.logit_bias[idx] += 1.0  # Encourage Science (High Risk/High Reward)
            elif color == "purple":
                self.logit_bias[idx] += 1.0  # Encourage Guilds (End game scoring)
            
            # Build Wonder (N..2N-1) - Encouraging wonder stages
            self.logit_bias[N + idx] += 0.5

    def encode_observation(self, obs: Dict, player_id: int) -> np.ndarray:
        """
        Convert a player's observation dictionary into a flat float vector.
        """
        p_obs = obs["players"][player_id]
        
        # Feature list
        features = []
        
        # 1. Global State
        features.append(obs["current_age"] / 3.0)
        features.append(obs["current_turn"] / 6.0)
        
        num_players = obs["num_players"]
        
        # 2. All Players (Self then Opponents in relative order)
        for i in range(num_players):
            target_pid = (player_id + i) % num_players
            target_obs = obs["players"][target_pid]
            
            # Wonder Name (One-hot)
            w_vec = np.zeros(self.num_wonders)
            if target_obs["wonder_name"] in self.wonder_to_id:
                w_vec[self.wonder_to_id[target_obs["wonder_name"]]] = 1.0
            features.extend(w_vec)
            
            # Wonder Side (One-hot)
            features.append(1.0 if target_obs["wonder_side"] == "day" else 0.0)
            features.append(1.0 if target_obs["wonder_side"] == "night" else 0.0)
            
            # Wonder Stage
            features.append(target_obs["wonder_stage_progress"] / 4.0)
            
            # Coins
            features.append(target_obs["coins"] / 20.0)
            
            # Production
            for r in ["wood", "stone", "ore", "clay", "glass", "papyrus", "textile"]:
                features.append(target_obs["production"].get(r, 0) / 5.0)
                
            # Science
            for s in ["compass", "gear", "tablet"]:
                features.append(target_obs["science"].get(s, 0) / 5.0)
                
            # Shields
            features.append(target_obs["shields"] / 10.0)
            
            # Military Score
            features.append(target_obs["military_tokens_score"] / 10.0)
            
            # Built Cards (Multi-hot) - Raw Data
            built_vec = np.zeros(self.num_cards)
            for card_name in target_obs["built_card_names"]:
                if card_name in self.card_to_id:
                    built_vec[self.card_to_id[card_name]] = 1.0
            features.extend(built_vec)

        # 3. Self Only (Private Information)
        self_obs = obs["players"][player_id]
        
        # Hand (Multi-hot)
        hand_vec = np.zeros(self.num_cards)
        for card_name in self_obs["current_hand"]:
            if card_name in self.card_to_id:
                hand_vec[self.card_to_id[card_name]] = 1.0
        features.extend(hand_vec)
        
        # Memory of Circulation (Multi-hot)
        mem_vec = np.zeros(self.num_cards)
        for card_name in self_obs.get("memory_known_cards", []):
            if card_name in self.card_to_id:
                mem_vec[self.card_to_id[card_name]] = 1.0
        features.extend(mem_vec)
        
        vec = np.array(features, dtype=np.float32)
            
        return vec

    def get_action_mask(self, env: GameEnv, player_id: int) -> np.ndarray:
        """
        Create a boolean mask of valid actions for the current state.
        1 = Valid, 0 = Invalid.
        """
        mask = np.zeros(self.action_space_size, dtype=np.float32)
        legal_actions = env.get_legal_actions(player_id)
        
        for action_str in legal_actions:
            idx = self.action_to_index(action_str)
            if idx is not None and idx < self.action_space_size:
                mask[idx] = 1.0
                
        return mask

    def action_to_index(self, action_str: str) -> int:
        """Map action string to integer index."""
        # Action types:
        # 0..N-1: Build Structure (Card ID)
        # N..2N-1: Build Wonder (Card ID)
        # 2N..3N-1: Discard (Card ID)
        
        N = self.num_cards
        
        if action_str.startswith("wonder_stage_"):
            card_name = action_str.replace("wonder_stage_", "")
            if card_name in self.card_to_id:
                return N + self.card_to_id[card_name]
                
        elif action_str.startswith("discard_"):
            card_name = action_str.replace("discard_", "")
            if card_name in self.card_to_id:
                return 2 * N + self.card_to_id[card_name]
        
        elif action_str == "discard":
            # Generic discard not mapped to specific card index here
            # Handled by fallback in env, but for RL we want specific discard actions
            return None
            
        else:
            # Build structure
            if action_str in self.card_to_id:
                return self.card_to_id[action_str]
                
        return None

    def index_to_action(self, idx: int) -> str:
        """Map integer index back to action string."""
        N = self.num_cards
        
        if idx < N:
            return self.id_to_card.get(idx, "discard")
        elif idx < 2 * N:
            card_name = self.id_to_card.get(idx - N)
            return f"wonder_stage_{card_name}" if card_name else "discard"
        elif idx < 3 * N:
            card_name = self.id_to_card.get(idx - 2 * N)
            return f"discard_{card_name}" if card_name else "discard"
            
        return "discard"


def create_actor_critic_model(input_dim: int, action_space_size: int):
    """Creates the AlphaZero Dual-Head Neural Network."""
    inputs = layers.Input(shape=(input_dim,))
    
    # Shared Backbone
    x = layers.Dense(1024, activation="relu")(inputs)
    x = layers.Dropout(0.1)(x)  # Prevent overfitting to specific game states
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    
    # Actor Head (Policy)
    # Outputs logits for all possible actions
    actor = layers.Dense(action_space_size, activation=None, name="actor_logits")(x)
    
    # Critic Head (Value)
    # Outputs estimated win probability/score (-1 to 1 or raw score)
    critic = layers.Dense(1, activation=None, name="critic_value")(x)
    
    model = keras.Model(inputs=inputs, outputs=[actor, critic])
    return model


class MCTSNode:
    """Node in the MCTS tree."""
    def __init__(self, prior: float):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}  # Action Index -> MCTSNode
        self.state = None   # (env_snapshot, pending_actions_dict, next_player_id)

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """
    Determinized Monte Carlo Tree Search.
    Handles simultaneous moves by serializing them in the tree.
    """
    def __init__(self, model, processor, num_sims=MCTS_SIMS, c_puct=C_PUCT):
        self.model = model
        self.processor = processor
        self.num_sims = num_sims
        self.c_puct = c_puct

    def search(self, env: GameEnv, root_player_id: int, temperature=1.0):
        """
        Run MCTS from the current state.
        Returns: policy_vector (probs over actions)
        """
        # 1. Determinize: Create a perfect information world by shuffling hidden hands
        det_env = self._determinize_env(env, root_player_id)
        
        # Root of the tree starts with the determinized environment
        # State tuple: (env, pending_actions, current_player_to_act)
        root = MCTSNode(prior=0.0)
        root.state = (det_env, {}, root_player_id) # Start serialization with the requesting player
        
        # Add Dirichlet noise to root for exploration
        self._expand(root)
        self._add_dirichlet_noise(root)

        for _ in range(self.num_sims):
            node = root
            search_path = [node]
            
            # SELECT
            while node.children:
                action_idx, node = self._select_child(node)
                search_path.append(node)
            
            # EXPAND & EVALUATE
            # Get the state for this leaf node
            leaf_env, pending, pid = node.state
            
            # Check if game over
            if leaf_env.is_done():
                # Calculate final score/reward
                # We need the reward for the player who just moved (parent of this node)
                # But simpler: evaluate value for the 'pid' whose turn it would be
                # Value is usually in range [0, 1] or [-1, 1]
                from seven_wonders.scoring import calculate_scores
                scores = calculate_scores(leaf_env)
                # Normalize score roughly 0-1 (max ~80)
                value = min(scores[pid], 80.0) / 80.0
            else:
                # Expand the node using the Neural Network
                value = self._expand(node)
            
            # BACKPROPAGATE
            self._backpropagate(search_path, value, root_player_id)

        # Calculate return policy based on visit counts
        counts = np.zeros(self.processor.action_space_size, dtype=np.float32)
        for action_idx, child in root.children.items():
            counts[action_idx] = child.visit_count
            
        if temperature == 0:
            best_action = np.argmax(counts)
            probs = np.zeros_like(counts)
            probs[best_action] = 1.0
        else:
            # Apply temperature
            counts = counts ** (1.0 / temperature)
            probs = counts / np.sum(counts)
            
        return probs

    def _determinize_env(self, env: GameEnv, observer_pid: int) -> GameEnv:
        """
        Clones the environment and shuffles cards in opponents' hands
        to create a concrete state consistent with the observer's view.
        """
        sim_env = copy.deepcopy(env)
        
        # Gather all cards from opponents
        hidden_cards = []
        hand_sizes = {}
        
        for pid in range(sim_env.num_players):
            if pid != observer_pid:
                hand = sim_env.players[pid].current_hand
                hidden_cards.extend(hand)
                hand_sizes[pid] = len(hand)
                sim_env.players[pid].current_hand = [] # Clear hand
        
        # Shuffle the unknown cards
        random.shuffle(hidden_cards)
        
        # Deal back
        current_idx = 0
        for pid in range(sim_env.num_players):
            if pid != observer_pid:
                size = hand_sizes[pid]
                new_hand = hidden_cards[current_idx : current_idx + size]
                sim_env.players[pid].current_hand = new_hand
                current_idx += size
                
        return sim_env

    def _select_child(self, node):
        """Select child using PUCT formula."""
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action_idx, child in node.children.items():
            # PUCT score
            u = self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            score = child.value() + u
            
            if score > best_score:
                best_score = score
                best_action = action_idx
                best_child = child
                
        return best_action, best_child

    def _expand(self, node):
        """
        Expand a leaf node.
        1. Get NN prediction for the current player (p, v).
        2. Create children for valid actions.
        3. Return value v.
        """
        env, pending, pid = node.state
        
        # Prepare input for NN
        # We need observation from the perspective of 'pid' in the determinized env
        # Note: We must construct the observation dict manually or use env internals
        # Since we have a deepcopy, we can just peek at the player object
        # But DataProcessor expects the full obs dict structure.
        # We can reconstruct a minimal obs dict for the processor.
        
        # Hack: We can't easily call env.step() to get obs without advancing state.
        # But we can manually build the obs dict from env state.
        # For simplicity/speed in this prototype, we will rely on the fact that
        # DataProcessor.encode_observation mostly reads from env.players[pid] attributes.
        # We'll construct a synthetic obs dict.
        
        obs_dict = {
            "current_age": env.current_age,
            "current_turn": env.current_turn,
            "num_players": env.num_players,
            "players": []
        }
        
        for i in range(env.num_players):
            p = env.players[i]
            p_data = {
                "wonder_name": p.wonder_name,
                "wonder_side": p.wonder_side,
                "wonder_stage_progress": p.current_wonder_stage,
                "coins": p.coins,
                "production": p.production,
                "science": p.science,
                "shields": sum(p.military_tokens), # Approx
                "military_tokens_score": sum(p.military_tokens),
                "built_card_names": [c.name for c in p.built_cards],
                "current_hand": [c.name for c in p.current_hand],
                "memory_known_cards": [c.name if not isinstance(c, str) else c for c in p.memory_known_cards]
            }
            obs_dict["players"].append(p_data)
            
        # Encode
        state_vec = self.processor.encode_observation(obs_dict, pid)
        mask = self.processor.get_action_mask(env, pid)
        full_state = np.concatenate([state_vec, mask])
        
        # Predict
        obs_tensor = tf.convert_to_tensor([full_state], dtype=tf.float32)
        logits, value = self.model(obs_tensor)
        value = value[0, 0].numpy()
        
        # Apply Mask & Bias to Logits to get Priors
        logits = logits[0].numpy()
        
        # Apply Mask (-inf)
        logits = logits + (1.0 - mask) * -1e9
        
        # Apply Guided Exploration Bias (from DataProcessor)
        logits = logits + self.processor.logit_bias
        
        # Softmax to get probabilities
        # Stable softmax
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        
        # Create children
        valid_indices = np.where(mask > 0)[0]
        for idx in valid_indices:
            child = MCTSNode(prior=probs[idx])
            
            # Determine child state (Transition Logic)
            new_pending = pending.copy()
            new_pending[pid] = self.processor.index_to_action(idx)
            
            if len(new_pending) == env.num_players:
                # All players have moved -> Step Environment
                next_env = copy.deepcopy(env)
                next_env.step(new_pending)
                child.state = (next_env, {}, 0) # Reset to Player 0
            else:
                # Partial turn -> Next player
                next_pid = (pid + 1) % env.num_players
                child.state = (copy.deepcopy(env), new_pending, next_pid)
                
            node.children[idx] = child
            
        return value

    def _backpropagate(self, search_path, value, root_player_id):
        """
        Backpropagate value up the tree.
        Note: Value is always positive (Score/WinRate).
        In 7 Wonders, everyone maximizes their own score.
        The 'value' computed at the leaf is for the player whose turn it was.
        But wait, the NN predicts value for the *input* player.
        
        Simplification for 7 Wonders:
        We assume the NN predicts the final normalized score for the player 'pid'.
        We want to maximize this.
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value

    def _add_dirichlet_noise(self, node):
        """Add noise to root priors to encourage exploration."""
        actions = list(node.children.keys())
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(actions))
        
        for i, action_idx in enumerate(actions):
            child = node.children[action_idx]
            child.prior = (1 - DIRICHLET_EPSILON) * child.prior + DIRICHLET_EPSILON * noise[i]


class AlphaZeroAgent:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
        self.mcts = MCTS(model, processor)
        
    def train_step(self, states, target_pis, target_vs):
        """
        Perform one AlphaZero update step.
        Loss = (z - v)^2 - pi^T * log(p) + reg
        """
        states = tf.cast(states, dtype=tf.float32)
        target_pis = tf.cast(target_pis, dtype=tf.float32)
        target_vs = tf.cast(target_vs, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Forward pass
            logits, values = self.model(states)
            values = tf.squeeze(values)
            
            # 1. Policy Loss (Cross Entropy)
            # We need to mask the logits before softmax to match MCTS valid moves
            N_actions = self.processor.action_space_size
            masks = states[:, -N_actions:]
            inf_mask = (1.0 - masks) * -1e9
            
            # Apply bias to help convergence
            bias_tensor = tf.convert_to_tensor([self.processor.logit_bias], dtype=tf.float32)
            
            masked_logits = logits + inf_mask + bias_tensor
            
            # Cross Entropy: -sum(target * log(pred))
            policy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=target_pis, logits=masked_logits)
            )
            
            # 2. Value Loss (MSE)
            value_loss = tf.reduce_mean(tf.square(target_vs - values))
            
            # Total Loss
            total_loss = value_loss + policy_loss
            
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return total_loss, policy_loss, value_loss


def run_self_play_episode(env: GameEnv, agent: AlphaZeroAgent, processor: DataProcessor, iteration: int, game_idx: int):
    """
    Run a single game of self-play using MCTS.
    Returns training examples: (state, mcts_policy, final_value)
    """
    obs = env.reset()
    done = False
    
    # Storage: list of (state, policy, player_id)
    episode_data = []
    action_stats = {}
    
    while not done:
        print(f"\rIteration {iteration}/{ITERATIONS}, Game {game_idx + 1}/{SELF_PLAY_GAMES + 1}, Age {env.current_age + 1}/3, Turn {env.current_turn + 1}/6", end="", flush=True)
        actions = {}
        
        # 1. Run MCTS for each player to select actions
        for pid in range(env.num_players):
            # Run MCTS
            # Note: In self-play, we want exploration, so we use temperature=1.0
            # In later moves (e.g. Age 3), we can reduce temp to 0.
            mcts_probs = agent.mcts.search(env, pid, temperature=1.0)
            
            # Select action from MCTS distribution
            action_idx = np.random.choice(len(mcts_probs), p=mcts_probs)
            
            # Store data for training
            # We need the state vector that the network sees
            state_vec = processor.encode_observation(obs[pid], pid)
            mask = processor.get_action_mask(env, pid)
            full_state = np.concatenate([state_vec, mask])
            
            episode_data.append((full_state, mcts_probs, pid))
            
            # Map to string for env
            actions[pid] = processor.index_to_action(action_idx)
            
            # Track stats
            act_str = actions[pid]
            stat_key = "unknown"
            if act_str.startswith("wonder_stage"): stat_key = "Wonder"
            elif act_str.startswith("discard"): stat_key = "Discard"
            else:
                # It's a build, get color
                stat_key = f"Build {processor.card_colors.get(act_str, 'Unknown').title()}"
            action_stats[stat_key] = action_stats.get(stat_key, 0) + 1
            
        # 2. Step environment
        next_obs, rewards, done, _ = env.step(actions)
        
        obs = next_obs

    # 3. Game Over - Assign Values
    from seven_wonders.scoring import calculate_scores
    final_scores = calculate_scores(env)
    
    # Calculate normalized values for each player
    # We use a mix of Rank and Raw Score to encourage winning AND high scores
    player_values = {}
    
    # Rank calculation
    ranking = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    ranks = {pid: i for i, (pid, score) in enumerate(ranking)}
    
    for pid in range(env.num_players):
        score = final_scores[pid]
        rank = ranks[pid]
        
        # Rank Reward: 1.0 (1st) to -1.0 (Last)
        if env.num_players > 1:
            rank_reward = 1.0 - (2.0 * rank / (env.num_players - 1))
        else:
            rank_reward = 1.0
            
        # Mix Rank (Winning) with Normalized Score (Performance)
        # This helps the value head distinguish between "bad loss" and "good loss"
        # Max score approx 80 for normalization
        score_reward = min(score, 80) / 80.0
        
        # Shift to 20% Rank / 80% Score to heavily encourage high-scoring strategies (Science)
        player_values[pid] = (0.2 * rank_reward) + (0.8 * score_reward)

    # 4. Build Training Examples
    training_examples = []
    for state, pi, pid in episode_data:
        v = player_values[pid]
        training_examples.append((state, pi, v))
        
    return training_examples, action_stats, list(final_scores.values())


def arena_battle(env: GameEnv, candidate_agent: AlphaZeroAgent, best_agent: AlphaZeroAgent, processor: DataProcessor, num_games: int):
    """
    Pit Candidate Agent vs Best Agent (Using raw network policy, no MCTS for speed).
    Candidate plays as Player 0, Best plays as others.
    """
    candidate_wins = 0
    candidate_total_score = 0
    best_agent_total_score = 0
    
    for _ in range(num_games):
        obs = env.reset()
        done = False
        
        while not done:
            actions = {}
            for pid in range(env.num_players):
                state_vec = processor.encode_observation(obs[pid], pid)
                mask = processor.get_action_mask(env, pid)
                full_state = np.concatenate([state_vec, mask])
                
                # Use raw network output (greedy)
                obs_tensor = tf.convert_to_tensor([full_state], dtype=tf.float32)
                if pid == 0:
                    logits, _ = candidate_agent.model(obs_tensor)
                else:
                    logits, _ = best_agent.model(obs_tensor)
                
                logits = logits[0].numpy()
                
                # Apply mask
                inf_mask = (1.0 - mask) * -1e9
                logits += inf_mask
                
                # Apply bias
                logits += processor.logit_bias
                
                # Greedy selection
                idx = np.argmax(logits)
                
                # Note: We skip MCTS in Arena for speed, testing the "intuition"
                
                actions[pid] = processor.index_to_action(idx)
            
            obs, _, done, _ = env.step(actions)
            
        # Check winner
        from seven_wonders.scoring import get_winner, calculate_scores
        scores = calculate_scores(env)
        winner_id = get_winner(env)
        
        candidate_total_score += scores[0]
        # Average score of opponents (who are all the best_agent)
        opponents_score_sum = sum(scores[i] for i in range(1, env.num_players))
        best_agent_total_score += opponents_score_sum / (env.num_players - 1)
        
        if winner_id == 0:
            candidate_wins += 1
            
    return candidate_wins / num_games, candidate_total_score / num_games, best_agent_total_score / num_games


def main():
    # Prevent sleep on macOS
    if platform.system() == "Darwin":
        try:
            subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])
            print("☕️ Caffeinate active: System sleep prevented.")
        except FileNotFoundError:
            pass

    # Initialize Environment and Processor
    env = GameEnv(num_players=NUM_PLAYERS)
    processor = DataProcessor(env)
    
    latest_model_path = f"latest_model_{NUM_PLAYERS}p.keras"
    best_model_path = f"best_model_{NUM_PLAYERS}p.keras"

    # Initialize Models
    # Always create new model structure, load weights if exist
    model = create_actor_critic_model(processor.input_dim, processor.action_space_size)
    if os.path.exists(latest_model_path):
        print(f"Loading weights from {latest_model_path}...")
        model.load_weights(latest_model_path)

    agent = AlphaZeroAgent(model, processor)
    
    # Best model
    if os.path.exists(best_model_path):
        print(f"Loading existing best model ({best_model_path})...")
        best_model = keras.models.load_model(best_model_path)
    else:
        best_model = keras.models.clone_model(model)
        best_model.set_weights(model.get_weights())

    best_agent = AlphaZeroAgent(best_model, processor)
    
    # Replay Buffer
    replay_buffer = []
    
    print("Starting Training Loop...")
    
    try:
        for iteration in range(1, ITERATIONS + 1):
            print(f"\n=== Iteration {iteration} (AlphaZero MCTS) ===")
            
            # 1. Self-Play Data Collection
            new_examples = []
            total_action_stats = {}
            total_scores = []
            
            for g in range(SELF_PLAY_GAMES):
                examples, stats, scores = run_self_play_episode(env, agent, processor, iteration, g + 1)
                new_examples.extend(examples)
                total_scores.extend(scores)
                
                # Aggregate stats
                for k, v in stats.items():
                    total_action_stats[k] = total_action_stats.get(k, 0) + v
            print()
                    
            # Add to replay buffer
            replay_buffer.extend(new_examples)
            if len(replay_buffer) > BUFFER_SIZE:
                replay_buffer = replay_buffer[-BUFFER_SIZE:]
            
            # 2. Training
            print(f"Training on {len(replay_buffer)} examples...")
            
            # Prepare batch data
            states = np.array([x[0] for x in replay_buffer])
            pis = np.array([x[1] for x in replay_buffer])
            vs = np.array([x[2] for x in replay_buffer])
            
            dataset = tf.data.Dataset.from_tensor_slices((
                states, pis, vs
            )).shuffle(len(states)).batch(BATCH_SIZE)
            
            total_loss_sum = 0
            policy_loss_sum = 0
            value_loss_sum = 0
            num_steps = 0

            for _ in range(EPOCHS_PER_ITERATION):
                for batch in dataset:
                    t_loss, p_loss, v_loss = agent.train_step(*batch)
                    total_loss_sum += t_loss
                    policy_loss_sum += p_loss
                    value_loss_sum += v_loss
                    num_steps += 1

            avg_total = total_loss_sum / num_steps if num_steps > 0 else 0
            avg_policy = policy_loss_sum / num_steps if num_steps > 0 else 0
            avg_value = value_loss_sum / num_steps if num_steps > 0 else 0
            
            avg_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
            print(f"Loss: {avg_total:.4f} (P: {avg_policy:.4f}, V: {avg_value:.4f}) | Avg Score: {avg_score:.1f}")
            
            # Print Action Distribution
            print("Action Distribution:")
            total_acts = sum(total_action_stats.values())
            for k, v in sorted(total_action_stats.items(), key=lambda x: x[1], reverse=True):
                if v/total_acts > 0.01: # Only show > 1%
                    print(f"  {k}: {v/total_acts:.1%}")

            # 3. Arena Evaluation
            if iteration % EVAL_INTERVAL == 0:
                print("Evaluating against best model...")
                win_rate, cand_score, best_score = arena_battle(env, agent, best_agent, processor, ARENA_GAMES)
                print(f"Candidate Win Rate: {win_rate:.2%} (Avg Score: {cand_score:.1f} vs {best_score:.1f})")
                
                # Dynamic margin: 5% for 2p (AlphaZero standard), scaled down for more players
                update_margin = 0.1 / NUM_PLAYERS
                if win_rate >= (1 / NUM_PLAYERS) + update_margin:
                    print("🚀 New Best Model Found! Saving...")
                    best_model.set_weights(model.get_weights())
                    model.save(best_model_path)
                else:
                    print("Candidate failed to beat best model.")
                    
            # Save latest model (overwriting previous)
            model.save(latest_model_path)

            # Save checkpoint
            # if iteration % 10 == 0:
            #     model.save(f"seven_wonders_iter_{iteration}.keras")

    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current model state...")
        model.save(latest_model_path)
        print("Model saved. Exiting.")

if __name__ == "__main__":
    main()