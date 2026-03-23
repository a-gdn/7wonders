import os
import numpy as np
import tensorflow as tf
from tensorflow import keras # type: ignore
from tensorflow.keras import layers, optimizers # type: ignore
import time
import random

# --- Import your existing environment ---
from seven_wonders.environment import GameEnv
from seven_wonders.scoring import calculate_scores

# Metal acceleration for Mac
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ==========================================
#             CONFIGURATION
# ==========================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_PLAYERS = 4
ITERATIONS = 3000
CURRICULUM_STEPS = 1 # Smooth transition from random to self-play

GAMES_PER_UPDATE = 200  
EPOCHS_PER_UPDATE = 4  
BATCH_SIZE = 1024      

GAMMA = 0.99           
GAE_LAMBDA = 0.95      
CLIP_RATIO = 0.2       

INITIAL_LEARNING_RATE = 5e-5 

# Increased entropy to force exploration against harder opponents
INITIAL_ENTROPY = 0.12       
FINAL_ENTROPY = 0.01      
MAX_GRAD_NORM = 0.5    

POOL_UPDATE_FREQ = 100  
MAX_POOL_SIZE = 100     

# ==========================================
#             DATA PROCESSOR
# ==========================================
class DataProcessor:
    def __init__(self, env: GameEnv):
        self.card_to_id = {}
        self.id_to_card = {}
        self.wonder_to_id = {}
        self._build_vocab(env)
        self.num_cards = len(self.card_to_id)
        self.num_wonders = len(self.wonder_to_id)
        self.action_space_size = self.num_cards * 3
        
        self.global_dim = 2 + 6  # +6 for edifice state (2 features per age × 3 ages)
        self.player_feat_dim = self.num_wonders + 18 + self.num_cards  # base + science + expansion features
        self.private_dim = 3 * self.num_cards 
        self.input_dim = self.global_dim + (env.num_players * self.player_feat_dim) + self.private_dim

    def _build_vocab(self, env: GameEnv):
        unique_names = set(c["name"] for c in env.cards_data)
        for idx, name in enumerate(sorted(list(unique_names))):
            self.card_to_id[name] = idx
            self.id_to_card[idx] = name
        unique_wonders = set(w["name"] for w in env.wonder_data)
        self.wonder_to_id = {name: i for i, name in enumerate(sorted(list(unique_wonders)))}

    def encode_observation(self, obs, player_id, legal_actions=None):
        features = []
        features.append(obs["current_age"] / 3.0)
        features.append(obs["current_turn"] / 6.0)
        
        # Add edifice state (selected projects and completion)
        edifice_state = obs.get("edifice_state", {})
        for age in [1, 2, 3]:
            proj_info = edifice_state.get(age, {})
            project_name = proj_info.get("project_name")
            is_built = 1.0 if proj_info.get("built", False) else 0.0
            # Use card_to_id if project is a known card, else use hash
            if project_name and project_name in self.card_to_id:
                proj_encoding = self.card_to_id[project_name] / max(self.num_cards, 1)
            else:
                proj_encoding = 0.0 if not project_name else (hash(project_name) % self.num_cards) / max(self.num_cards, 1)
            features.append(proj_encoding)
            features.append(is_built)
        
        for i in range(NUM_PLAYERS):
            target_pid = (player_id + i) % NUM_PLAYERS
            p = obs["players"][target_pid]
            w_vec = np.zeros(self.num_wonders)
            if p["wonder_name"] in self.wonder_to_id:
                w_vec[self.wonder_to_id[p["wonder_name"]]] = 1.0
            features.extend(w_vec)
            features.extend([
                1.0 if p["wonder_side"] == "day" else 0.0,
                1.0 if p["wonder_side"] == "night" else 0.0,
                p["wonder_stage_progress"] / 4.0,
                min(p["coins"] / 20.0, 1.0), 
            ])
            prod = p["production"]
            features.extend([
                prod.get("wood", 0)/4.0, prod.get("stone", 0)/4.0, prod.get("ore", 0)/4.0, 
                prod.get("clay", 0)/4.0, prod.get("glass", 0)/3.0, prod.get("papyrus", 0)/3.0, 
                prod.get("textile", 0)/3.0
            ])
            sci = p["science"]
            features.extend([
                sci.get("compass", 0)/4.0, sci.get("gear", 0)/4.0, sci.get("tablet", 0)/4.0, 
                min(1.0, p["shields"]/15.0), 
                min(1.0, (p["military_tokens_score"] + 10) / 30.0),
                min(1.0, p["diplomacy_tokens"] / 5.0),
                min(1.0, max(p["debt_tokens"], -10) / 10.0),
                1.0 if p["edifice_participated"].get(1, False) else 0.0,
                1.0 if p["edifice_participated"].get(2, False) else 0.0,
                1.0 if p["edifice_participated"].get(3, False) else 0.0
            ])
            built = np.zeros(self.num_cards)
            for c in p["built_card_names"]:
                if c in self.card_to_id: built[self.card_to_id[c]] = 1.0
            features.extend(built)

        self_p = obs["players"][player_id]
        hand, mem, can_build_mask = np.zeros(self.num_cards), np.zeros(self.num_cards), np.zeros(self.num_cards)
        for c in self_p["current_hand"]:
            if c in self.card_to_id: hand[self.card_to_id[c]] = 1.0
        for c in self_p.get("memory_known_cards", []):
            if c in self.card_to_id: mem[self.card_to_id[c]] = 1.0
        if legal_actions:
            for action in legal_actions:
                if action in self.card_to_id: can_build_mask[self.card_to_id[action]] = 1.0
        
        features.extend(hand); features.extend(mem); features.extend(can_build_mask) 
        return np.array(features, dtype=np.float32)

    def get_action_mask(self, env, player_id):
        mask = np.zeros(self.action_space_size, dtype=np.float32)
        legal = env.get_legal_actions(player_id)
        for act in legal:
            idx = self.action_to_index(act)
            if idx is not None: mask[idx] = 1.0
        return mask

    def action_to_index(self, action_str):
        N = self.num_cards
        if action_str.startswith("wonder_stage_"):
            name = action_str.replace("wonder_stage_", "")
            return N + self.card_to_id.get(name, 0)
        elif action_str.startswith("discard_"):
            name = action_str.replace("discard_", "")
            return 2 * N + self.card_to_id.get(name, 0)
        elif action_str in self.card_to_id: return self.card_to_id[action_str]
        return None

    def index_to_action(self, idx):
        N = self.num_cards
        if idx < N: return self.id_to_card.get(idx, "discard") 
        elif idx < 2*N: return f"wonder_stage_{self.id_to_card.get(idx-N, '')}"
        else: return f"discard_{self.id_to_card.get(idx-2*N, '')}"

# ==========================================
#             PPO AGENT
# ==========================================
def create_ppo_model(input_dim, action_size):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.LayerNormalization()(x)
    return keras.Model(inputs=inputs, outputs=[layers.Dense(action_size)(x), layers.Dense(1)(x)])

class PPOAgent:
    def __init__(self, model):
        self.model = model
        self.optimizer = optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)

    @tf.function
    def get_action_and_value(self, state, mask):
        logits, value = self.model(state, training=False)
        masked_logits = logits + (1.0 - mask) * -1e9
        action = tf.random.categorical(masked_logits, 1)[0, 0]
        probs = tf.nn.softmax(masked_logits)
        logprob = tf.math.log(probs[0, action] + 1e-10)
        return action, logprob, value[0, 0]

    @tf.function
    def train_step(self, states, actions, old_logprobs, returns, advantages, masks, current_entropy_coef):
        with tf.GradientTape() as tape:
            logits, values = self.model(states, training=True)
            values = tf.squeeze(values, -1)
            masked_logits = logits + (1.0 - masks) * -1e9
            probs = tf.nn.softmax(masked_logits)
            
            action_masks = tf.one_hot(actions, depth=tf.shape(logits)[1])
            new_logprobs = tf.math.log(tf.reduce_sum(probs * action_masks, axis=1) + 1e-10)
            
            ratio = tf.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO) * advantages
            
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            value_loss = tf.reduce_mean(tf.square(returns - values))
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1))
            
            total_loss = policy_loss + 0.5 * value_loss - current_entropy_coef * entropy

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return policy_loss, value_loss, entropy

def _safe_sum_debt(debt_list):
    """Flatten and sum debt tokens, handling nested lists."""
    if not debt_list:
        return 0
    total = 0
    for item in debt_list:
        if isinstance(item, list):
            total += sum(item)
        else:
            total += item
    return total

def env_to_dict(env):
    edifice_state = {}
    if hasattr(env, 'active_edifices'):
        for age in [1, 2, 3]:
            edifice = getattr(env, 'active_edifices', {}).get(age)
            complete = getattr(env, 'edifice_completed', {}).get(age, False)
            edifice_state[age] = {
                "project_name": edifice.get("name") if edifice else None,
                "built": complete
            }
    
    return {
        "current_age": env.current_age, "current_turn": env.current_turn,
        "edifice_state": edifice_state,
        "players": [{
            "wonder_name": p.wonder_name, "wonder_side": p.wonder_side, "wonder_stage_progress": p.current_wonder_stage,
            "coins": p.coins, "production": p.production, "science": p.science, "shields": sum(p.military_tokens),
            "military_tokens_score": sum(p.military_tokens), "built_card_names": [c.name for c in p.built_cards],
            "current_hand": [c.name for c in p.current_hand],
            "memory_known_cards": [c if isinstance(c, str) else c.name for c in p.memory_known_cards],
            "diplomacy_tokens": getattr(p, "diplomacy_tokens", 0),
            "debt_tokens": _safe_sum_debt(getattr(p, "edifice_debt_tokens", [])) + _safe_sum_debt(getattr(p, "cities_debt_tokens", [])),
            "edifice_participated": getattr(p, "participated_in_edifice", {1: False, 2: False, 3: False})
        } for p in env.players]
    }

def get_intermediate_reward(action_str):
    # FIX 1: Scaled down by 10x to prevent agent from ignoring the terminal game score.
    if action_str.startswith("wonder_stage_"):
        return 0.010 
    elif action_str.startswith("discard_"):
        return -0.005
    else:
        return 0.005 

# ==========================================
#             MAIN LOOP
# ==========================================
def main():
    env = GameEnv(num_players=NUM_PLAYERS)
    proc = DataProcessor(env)
    
    # Hero Model
    model = create_ppo_model(proc.input_dim, proc.action_space_size)
    agent = PPOAgent(model)
    
    # FIX 2: Create a distinct model for each opponent to support heterogeneous self-play
    opponent_models = [create_ppo_model(proc.input_dim, proc.action_space_size) for _ in range(NUM_PLAYERS - 1)]
    opponent_agents = [PPOAgent(m) for m in opponent_models]
    
    weights_path = "ppo_7wonders_latest.keras"
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"✅ Loaded existing model weights from {weights_path}! Continuing training...")

    opponent_pool = [model.get_weights()]
    
    log_dir = f"logs/7wonders_ppo/{int(time.time())}"
    os.makedirs("logs/7wonders_ppo", exist_ok=True)
    summary_writer = tf.summary.create_file_writer(log_dir)

    print(f"🚀 PPO Training Started | Input: {proc.input_dim} | Actions: {proc.action_space_size}")

    for iter_idx in range(1, ITERATIONS + 1):
        start_time = time.time()
        
        # --- POOL UPDATE ---
        if iter_idx % POOL_UPDATE_FREQ == 0:
            opponent_pool.append(model.get_weights())
            if len(opponent_pool) > MAX_POOL_SIZE:
                opponent_pool.pop(0) 
        
        # --- SCHEDULE UPDATES ---
        progress = min((iter_idx - 1) / ITERATIONS, 1.0)
        current_entropy = INITIAL_ENTROPY - progress * (INITIAL_ENTROPY - FINAL_ENTROPY)
        current_entropy_tf = tf.constant(current_entropy, dtype=tf.float32)
        
        current_lr = INITIAL_LEARNING_RATE - progress * (INITIAL_LEARNING_RATE - 1e-5)
        agent.optimizer.learning_rate.assign(current_lr)
        
        # --- CURRICULUM LOGIC ---
        p_self_play = min(iter_idx / CURRICULUM_STEPS, 1.0)

        b_states, b_actions, b_logprobs, b_returns, b_advs, b_masks = [], [], [], [], [], []
        batch_scores = []
        hero_wins = 0 

        for _ in range(GAMES_PER_UPDATE):
            env.reset()
            done = False
            
            # Decide if this game uses random opponents or the trained pool
            use_pool = random.random() < p_self_play
            if use_pool:
                # FIX 2 (cont): Sample an independent historical weight for each opponent model
                for opp_model in opponent_models:
                    opp_weights = random.choice(opponent_pool)
                    opp_model.set_weights(opp_weights)
            
            hero_traj = {"states": [], "actions": [], "logprobs": [], "rewards": [], "values": [], "masks": []} 
            
            while not done:
                actions_str = {}
                obs = env_to_dict(env)
                
                for pid in range(NUM_PLAYERS):
                    state = proc.encode_observation(obs, pid, env.get_legal_actions(pid))
                    mask = proc.get_action_mask(env, pid)
                    
                    if pid == 0: # Hero
                        state_tf = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
                        mask_tf = tf.convert_to_tensor(mask[None, :], dtype=tf.float32)
                        act_idx, lp, val = agent.get_action_and_value(state_tf, mask_tf)
                        chosen_action = proc.index_to_action(act_idx.numpy())
                        actions_str[pid] = chosen_action
                        
                        hero_traj["states"].append(state)
                        hero_traj["actions"].append(act_idx.numpy())
                        hero_traj["logprobs"].append(lp.numpy())
                        hero_traj["values"].append(val.numpy())
                        hero_traj["masks"].append(mask)
                        
                        step_reward = get_intermediate_reward(chosen_action)
                        hero_traj["rewards"].append(step_reward) 
                    else: # Opponents
                        if use_pool:
                            opp_idx = pid - 1 # Opponent models are 0-indexed (0, 1, 2)
                            state_tf = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
                            mask_tf = tf.convert_to_tensor(mask[None, :], dtype=tf.float32)
                            act_idx, _, _ = opponent_agents[opp_idx].get_action_and_value(state_tf, mask_tf)
                            actions_str[pid] = proc.index_to_action(act_idx.numpy())
                        else:
                            # Random legal action
                            valid_indices = np.where(mask == 1.0)[0]
                            random_act_idx = np.random.choice(valid_indices)
                            actions_str[pid] = proc.index_to_action(random_act_idx)

                _, _, done, _ = env.step(actions_str)
                    
            scores = calculate_scores(env)
            batch_scores.append(scores[0]) 
            
            scores_list = sorted(list(scores.values()), reverse=True)
            hero_rank = scores_list.index(scores[0]) 
            
            if hero_rank == 0:
                hero_wins += 1
            
            rank_rewards = [1.0, 0.5, 0.0, -0.5]
            rank_bonus = rank_rewards[hero_rank]
            
            # FIX 3: With tiny intermediate rewards, this terminal reward correctly dominates the training signal
            absolute_reward = scores[0] / 50.0 
            hero_traj["rewards"][-1] += (absolute_reward + rank_bonus)
            
            rewards = np.array(hero_traj["rewards"])
            values = np.array(hero_traj["values"])
            advs = np.zeros_like(rewards, dtype=np.float32)
            
            last_adv = 0.0
            last_val = 0.0 
            
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + GAMMA * last_val - values[t]
                advs[t] = last_adv = delta + GAMMA * GAE_LAMBDA * last_adv
                last_val = values[t]
                
            returns = advs + values
            
            b_states.extend(hero_traj["states"])
            b_actions.extend(hero_traj["actions"])
            b_logprobs.extend(hero_traj["logprobs"])
            b_returns.extend(returns)
            b_advs.extend(advs)
            b_masks.extend(hero_traj["masks"])

        b_advs = np.array(b_advs)
        b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)

        dataset = tf.data.Dataset.from_tensor_slices((
            np.array(b_states), np.array(b_actions), np.array(b_logprobs), 
            np.array(b_returns).astype('float32'), b_advs.astype('float32'), np.array(b_masks)
        )).shuffle(2048).batch(BATCH_SIZE)
        
        l_a, l_c, l_e = [], [], []
        for _ in range(EPOCHS_PER_UPDATE):
            for batch in dataset:
                b_s, b_a, b_lp, b_ret, b_adv, b_m = batch
                pa, pc, pe = agent.train_step(b_s, b_a, b_lp, b_ret, b_adv, b_m, current_entropy_tf)
                
                l_a.append(pa.numpy())
                l_c.append(pc.numpy())
                l_e.append(pe.numpy())

        avg_score, max_score_batch = np.mean(batch_scores), np.max(batch_scores)
        win_rate = (hero_wins / GAMES_PER_UPDATE) * 100

        with summary_writer.as_default():
            tf.summary.scalar('Game/Hero_Avg_Score', avg_score, step=iter_idx)
            tf.summary.scalar('Game/Hero_Max_Score', max_score_batch, step=iter_idx)
            tf.summary.scalar('Game/Win_Rate', win_rate, step=iter_idx)
            tf.summary.scalar('Curriculum/Self_Play_Ratio', p_self_play, step=iter_idx)
            tf.summary.scalar('Metrics/Entropy_Actual', np.mean(l_e), step=iter_idx)

        if iter_idx % 20 == 0: 
            print(f"Iter {iter_idx:3d} | Win Rate: {win_rate:5.1f}% | Score: {avg_score:4.1f} | SP_Ratio: {p_self_play:4.2f} | Entropy: {np.mean(l_e):4.2f} | Time: {time.time()-start_time:.1f}s")
        
        if iter_idx % 20 == 0 or iter_idx == ITERATIONS: 
            model.save("ppo_7wonders_latest.keras")

if __name__ == "__main__":
    main()