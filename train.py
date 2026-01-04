"""
AlphaZero-like Training Script for 7 Wonders (using PPO)

This script implements a self-play reinforcement learning loop:
1. Self-Play: Agents play against copies of themselves to generate data.
2. Training: PPO (Proximal Policy Optimization) updates the policy and value networks.
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

from seven_wonders.environment import GameEnv
from seven_wonders.constants import CardColor

# --- Hyperparameters ---
LEARNING_RATE = 0.0003
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
BATCH_SIZE = 64
EPOCHS_PER_ITERATION = 4
SELF_PLAY_GAMES = 100  # Games per iteration
ARENA_GAMES = 300      # Games for evaluation
EVAL_INTERVAL = 5     # Iterations between evaluations
NUM_PLAYERS = 4       # Number of players

class DataProcessor:
    """
    Handles conversion between Environment Observations/Actions and Neural Network Tensors.
    """
    def __init__(self, env: GameEnv):
        self.card_to_id = {}
        self.id_to_card = {}
        self._build_vocab(env)
        
        self.num_cards = len(self.card_to_id)
        self.action_space_size = self.num_cards * 3
        # 2 (Global) + 14 (Player) + 3 * num_cards (Hand, Built, Memory)
        self.input_dim = 16 + 3 * self.num_cards
        
    def _build_vocab(self, env: GameEnv):
        """Build vocabulary from all possible cards in the game database."""
        # Extract all unique card names from the loaded data
        unique_names = set()
        for card_data in env.cards_data:
            unique_names.add(card_data["name"])
        
        # Create mappings
        sorted_names = sorted(list(unique_names))
        for idx, name in enumerate(sorted_names):
            self.card_to_id[name] = idx
            self.id_to_card[idx] = name
            
        print(f"DataProcessor: Vocab size {len(self.card_to_id)} cards.")

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
        
        # 2. Player State (Self)
        features.append(p_obs["coins"] / 20.0)
        features.append(p_obs["shields"] / 10.0)
        features.append(p_obs["wonder_stage_progress"] / 4.0)
        features.append(1.0 if p_obs["wonder_side"] == "day" else 0.0)
        
        # Resources (Production) - simplified vector
        resources = ["wood", "stone", "ore", "clay", "glass", "papyrus", "textile"]
        for r in resources:
            features.append(p_obs["production"].get(r, 0) / 5.0)
            
        # Science
        science = ["compass", "gear", "tablet"]
        for s in science:
            features.append(p_obs["science"].get(s, 0) / 5.0)
            
        # 3. Hand (Multi-hot encoding)
        hand_vec = np.zeros(self.num_cards)
        for card_name in p_obs["current_hand"]:
            if card_name in self.card_to_id:
                hand_vec[self.card_to_id[card_name]] = 1.0
        features.extend(hand_vec)
        
        # 4. Built Cards (Multi-hot encoding)
        built_vec = np.zeros(self.num_cards)
        for card_name in p_obs["built_card_names"]:
            if card_name in self.card_to_id:
                built_vec[self.card_to_id[card_name]] = 1.0
        features.extend(built_vec)
        
        # 5. Memory of Circulation (Multi-hot encoding)
        mem_vec = np.zeros(self.num_cards)
        for card_name in p_obs.get("memory_known_cards", []):
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
    """Creates the PPO Actor-Critic Neural Network."""
    inputs = layers.Input(shape=(input_dim,))
    
    # Shared Backbone
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)
    
    # Actor Head (Policy)
    # Outputs logits for all possible actions
    actor = layers.Dense(action_space_size, activation=None, name="actor_logits")(x)
    
    # Critic Head (Value)
    # Outputs estimated win probability/score (-1 to 1 or raw score)
    critic = layers.Dense(1, activation=None, name="critic_value")(x)
    
    model = keras.Model(inputs=inputs, outputs=[actor, critic])
    return model


class PPOAgent:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
        
    def select_action(self, obs_vec, mask):
        """
        Select an action using the model's policy.
        Returns: action_index, log_prob, value
        """
        obs_tensor = tf.convert_to_tensor([obs_vec], dtype=tf.float32)
        logits, value = self.model(obs_tensor)
        
        # Apply mask (set invalid actions to -inf)
        mask_tensor = tf.convert_to_tensor([mask], dtype=tf.float32)
        inf_mask = (1.0 - mask_tensor) * -1e9
        masked_logits = logits + inf_mask
        
        # Sample from categorical distribution
        action_idx = tf.random.categorical(masked_logits, 1)[0, 0].numpy()
        
        # Calculate log probability of the selected action
        log_probs = tf.nn.log_softmax(masked_logits)
        action_log_prob = log_probs[0, action_idx]
        
        return action_idx, action_log_prob, value[0, 0]

    def train_step(self, states, actions, old_log_probs, returns, advantages):
        """Perform one PPO update step."""
        states = tf.cast(states, dtype=tf.float32)
        actions = tf.cast(actions, dtype=tf.int32)
        old_log_probs = tf.cast(old_log_probs, dtype=tf.float32)
        returns = tf.cast(returns, dtype=tf.float32)
        advantages = tf.cast(advantages, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            logits, values = self.model(states)
            values = tf.squeeze(values)
            
            # Re-calculate log probs
            log_probs_all = tf.nn.log_softmax(logits)
            indices = tf.range(len(actions))
            gathered_indices = tf.stack([indices, actions], axis=1)
            new_log_probs = tf.gather_nd(log_probs_all, gathered_indices)
            
            # Ratio
            ratio = tf.exp(new_log_probs - old_log_probs)
            
            # Surrogate Loss
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            # Value Loss
            value_loss = tf.reduce_mean(tf.square(returns - values))
            
            # Entropy Loss (exploration bonus)
            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_sum(probs * log_probs_all, axis=1)
            entropy_loss = -tf.reduce_mean(entropy)
            
            total_loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss
            
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return total_loss, policy_loss, value_loss


def compute_gae(rewards, values, next_value, dones):
    """Compute Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_val = values[t + 1]
            
        delta = rewards[t] + GAMMA * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
        
    returns = advantages + values
    return advantages, returns


def run_self_play_episode(env: GameEnv, agent: PPOAgent, processor: DataProcessor):
    """
    Run a single game of self-play.
    Returns trajectories for all players.
    """
    obs = env.reset()
    done = False
    
    # Storage for each player
    trajectories = {i: {'states': [], 'actions': [], 'log_probs': [], 'values': [], 'rewards': []} 
                   for i in range(env.num_players)}
    
    while not done:
        actions = {}
        
        # 1. Select actions for all players
        for pid in range(env.num_players):
            # Encode state
            state_vec = processor.encode_observation(obs, pid)
            mask = processor.get_action_mask(env, pid)
            
            # Predict
            action_idx, log_prob, value = agent.select_action(state_vec, mask)
            
            # Store
            trajectories[pid]['states'].append(state_vec)
            trajectories[pid]['actions'].append(action_idx)
            trajectories[pid]['log_probs'].append(log_prob)
            trajectories[pid]['values'].append(value)
            
            # Map to string for env
            actions[pid] = processor.index_to_action(action_idx)
            
        # 2. Step environment
        next_obs, rewards, done, _ = env.step(actions)
        
        # 3. Store rewards (intermediate rewards usually 0, final reward at end)
        for pid in range(env.num_players):
            trajectories[pid]['rewards'].append(rewards[pid])
            
        obs = next_obs

    # 4. Calculate final rewards (Win/Loss/Score)
    # We use normalized score or rank as reward
    from seven_wonders.scoring import calculate_scores
    final_scores = calculate_scores(env)
    max_score = max(final_scores.values())
    
    for pid in range(env.num_players):
        # Reward: 1.0 if winner, 0.0 otherwise (or normalized score)
        # Simple binary reward for AlphaZero style
        is_winner = (final_scores[pid] == max_score)
        final_reward = 1.0 if is_winner else -1.0
        
        # Assign final reward to the last step
        trajectories[pid]['rewards'][-1] = final_reward
        
    return trajectories


def arena_battle(env: GameEnv, candidate_agent: PPOAgent, best_agent: PPOAgent, processor: DataProcessor, num_games: int):
    """
    Pit Candidate Agent vs Best Agent.
    Candidate plays as Player 0, Best plays as others.
    """
    candidate_wins = 0
    
    for _ in range(num_games):
        obs = env.reset()
        done = False
        
        while not done:
            actions = {}
            for pid in range(env.num_players):
                state_vec = processor.encode_observation(obs, pid)
                mask = processor.get_action_mask(env, pid)
                
                if pid == 0:
                    # Candidate
                    idx, _, _ = candidate_agent.select_action(state_vec, mask)
                else:
                    # Best / Opponent
                    idx, _, _ = best_agent.select_action(state_vec, mask)
                
                actions[pid] = processor.index_to_action(idx)
            
            obs, _, done, _ = env.step(actions)
            
        # Check winner
        from seven_wonders.scoring import get_winner
        winner_id = get_winner(env)
        if winner_id == 0:
            candidate_wins += 1
            
    return candidate_wins / num_games


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
    if os.path.exists(latest_model_path):
        print(f"Loading existing latest model ({latest_model_path})...")
        model = keras.models.load_model(latest_model_path)
    else:
        model = create_actor_critic_model(processor.input_dim, processor.action_space_size)

    agent = PPOAgent(model, processor)
    
    # Best model
    if os.path.exists(best_model_path):
        print(f"Loading existing best model ({best_model_path})...")
        best_model = keras.models.load_model(best_model_path)
    else:
        best_model = keras.models.clone_model(model)
        best_model.set_weights(model.get_weights())

    best_agent = PPOAgent(best_model, processor)
    
    print("Starting Training Loop...")
    
    for iteration in range(1, 101):
        print(f"\n=== Iteration {iteration} ===")
        
        # 1. Self-Play Data Collection
        all_states, all_actions, all_log_probs, all_returns, all_advantages = [], [], [], [], []
        
        for g in range(SELF_PLAY_GAMES):
            trajectories = run_self_play_episode(env, agent, processor)
            
            # Process trajectories for each player
            for pid, traj in trajectories.items():
                # Compute GAE
                rewards = np.array(traj['rewards'])
                values = np.array(traj['values'])
                dones = np.zeros_like(rewards)
                dones[-1] = 1  # Last step is terminal
                
                advantages, returns = compute_gae(rewards, values, 0.0, dones)
                
                all_states.extend(traj['states'])
                all_actions.extend(traj['actions'])
                all_log_probs.extend(traj['log_probs'])
                all_returns.extend(returns)
                all_advantages.extend(advantages)
                
        # Normalize advantages
        all_advantages = np.array(all_advantages, dtype=np.float32)
        all_advantages = (all_advantages - np.mean(all_advantages)) / (np.std(all_advantages) + 1e-8)
        
        # Ensure correct types for dataset
        all_states = np.array(all_states, dtype=np.float32)
        all_log_probs = np.array(all_log_probs, dtype=np.float32)
        all_returns = np.array(all_returns, dtype=np.float32)
        all_actions = np.array(all_actions, dtype=np.int32)
        
        # 2. Training
        print(f"Training on {len(all_states)} steps...")
        dataset = tf.data.Dataset.from_tensor_slices((
            all_states, all_actions, all_log_probs, all_returns, all_advantages
        )).shuffle(len(all_states)).batch(BATCH_SIZE)
        
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
        
        print(f"Loss: {avg_total:.4f} (P: {avg_policy:.4f}, V: {avg_value:.4f})")
        
        # 3. Arena Evaluation
        if iteration % EVAL_INTERVAL == 0:
            print("Evaluating against best model...")
            win_rate = arena_battle(env, agent, best_agent, processor, ARENA_GAMES)
            print(f"Candidate Win Rate: {win_rate:.2%}")
            
            if win_rate >= 0.55:
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

if __name__ == "__main__":
    main()