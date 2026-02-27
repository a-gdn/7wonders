import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import time
import random

from seven_wonders.environment import GameEnv
from seven_wonders.scoring import calculate_scores

# ==========================================
#            GPU (Mac Metal)
# ==========================================
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==========================================
#             CONFIGURATION
# ==========================================
NUM_PLAYERS = 4
ITERATIONS = 2000

GAMES_PER_UPDATE = 200
EPOCHS_PER_UPDATE = 4
BATCH_SIZE = 1024

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2

INITIAL_LEARNING_RATE = 3e-4
INITIAL_ENTROPY = 0.12
FINAL_ENTROPY = 0.01

MAX_GRAD_NORM = 0.5

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

        self.global_dim = 2
        self.player_feat_dim = self.num_wonders + 16 + self.num_cards
        self.private_dim = 3 * self.num_cards
        self.input_dim = (
            self.global_dim
            + (env.num_players * self.player_feat_dim)
            + self.private_dim
        )

    def _build_vocab(self, env: GameEnv):
        unique_names = set(c["name"] for c in env.cards_data)
        for idx, name in enumerate(sorted(unique_names)):
            self.card_to_id[name] = idx
            self.id_to_card[idx] = name

        unique_wonders = set(w["name"] for w in env.wonder_data)
        self.wonder_to_id = {
            name: i for i, name in enumerate(sorted(unique_wonders))
        }

    def encode_observation(self, obs, player_id, legal_actions=None):
        features = []

        features.append(obs["current_age"] / 3.0)
        features.append(obs["current_turn"] / 6.0)

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
                prod.get("wood", 0)/4.0,
                prod.get("stone", 0)/4.0,
                prod.get("ore", 0)/4.0,
                prod.get("clay", 0)/4.0,
                prod.get("glass", 0)/3.0,
                prod.get("papyrus", 0)/3.0,
                prod.get("textile", 0)/3.0
            ])

            sci = p["science"]
            features.extend([
                sci.get("compass", 0)/4.0,
                sci.get("gear", 0)/4.0,
                sci.get("tablet", 0)/4.0,
                min(1.0, p["shields"]/15.0),
                min(1.0, (p["military_tokens_score"] + 10)/30.0)
            ])

            built = np.zeros(self.num_cards)
            for c in p["built_card_names"]:
                if c in self.card_to_id:
                    built[self.card_to_id[c]] = 1.0
            features.extend(built)

        self_p = obs["players"][player_id]

        hand = np.zeros(self.num_cards)
        mem = np.zeros(self.num_cards)
        can_build = np.zeros(self.num_cards)

        for c in self_p["current_hand"]:
            if c in self.card_to_id:
                hand[self.card_to_id[c]] = 1.0

        if legal_actions:
            for action in legal_actions:
                if action in self.card_to_id:
                    can_build[self.card_to_id[action]] = 1.0

        features.extend(hand)
        features.extend(mem)
        features.extend(can_build)

        return np.array(features, dtype=np.float32)

    def get_action_mask(self, env, player_id):
        mask = np.zeros(self.action_space_size, dtype=np.float32)
        legal = env.get_legal_actions(player_id)
        for act in legal:
            idx = self.action_to_index(act)
            if idx is not None:
                mask[idx] = 1.0
        return mask

    def action_to_index(self, action_str):
        N = self.num_cards
        if action_str.startswith("wonder_stage_"):
            name = action_str.replace("wonder_stage_", "")
            return N + self.card_to_id.get(name, 0)
        elif action_str.startswith("discard_"):
            name = action_str.replace("discard_", "")
            return 2 * N + self.card_to_id.get(name, 0)
        elif action_str in self.card_to_id:
            return self.card_to_id[action_str]
        return None

    def index_to_action(self, idx):
        N = self.num_cards
        if idx < N:
            return self.id_to_card.get(idx, "discard")
        elif idx < 2*N:
            return f"wonder_stage_{self.id_to_card.get(idx-N, '')}"
        else:
            return f"discard_{self.id_to_card.get(idx-2*N, '')}"

# ==========================================
#             PPO MODEL
# ==========================================
def create_ppo_model(input_dim, action_size):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.LayerNormalization()(x)
    policy_logits = layers.Dense(action_size)(x)
    value = layers.Dense(1)(x)
    return keras.Model(inputs=inputs, outputs=[policy_logits, value])

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
    def train_step(self, states, actions, old_logprobs,
                   returns, advantages, masks, entropy_coef):

        with tf.GradientTape() as tape:
            logits, values = self.model(states, training=True)
            values = tf.squeeze(values, -1)

            masked_logits = logits + (1.0 - masks) * -1e9
            probs = tf.nn.softmax(masked_logits)

            action_masks = tf.one_hot(actions, depth=tf.shape(logits)[1])
            new_logprobs = tf.math.log(
                tf.reduce_sum(probs * action_masks, axis=1) + 1e-10
            )

            ratio = tf.exp(new_logprobs - old_logprobs)

            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(
                ratio,
                1.0 - CLIP_RATIO,
                1.0 + CLIP_RATIO
            ) * advantages

            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            value_loss = tf.reduce_mean(tf.square(returns - values))
            entropy = -tf.reduce_mean(
                tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1)
            )

            total_loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

# ==========================================
#             TRAINING LOOP
# ==========================================
def main():
    env = GameEnv(num_players=NUM_PLAYERS)
    proc = DataProcessor(env)

    model = create_ppo_model(proc.input_dim, proc.action_space_size)
    agent = PPOAgent(model)

    print("🚀 Training vs RANDOM opponents only")

    for iteration in range(1, ITERATIONS + 1):

        progress = min((iteration - 1) / ITERATIONS, 1.0)
        entropy_coef = INITIAL_ENTROPY - progress * (INITIAL_ENTROPY - FINAL_ENTROPY)
        entropy_tf = tf.constant(entropy_coef, dtype=tf.float32)

        batch_states = []
        batch_actions = []
        batch_logprobs = []
        batch_returns = []
        batch_advs = []
        batch_masks = []
        scores_batch = []

        for _ in range(GAMES_PER_UPDATE):

            env.reset()
            done = False

            hero_traj = {
                "states": [],
                "actions": [],
                "logprobs": [],
                "values": [],
                "rewards": [],
                "masks": []
            }

            while not done:
                actions = {}

                obs = {
                    "current_age": env.current_age,
                    "current_turn": env.current_turn,
                    "players": [{
                        "wonder_name": p.wonder_name,
                        "wonder_side": p.wonder_side,
                        "wonder_stage_progress": p.current_wonder_stage,
                        "coins": p.coins,
                        "production": p.production,
                        "science": p.science,
                        "shields": sum(p.military_tokens),
                        "military_tokens_score": sum(p.military_tokens),
                        "built_card_names": [c.name for c in p.built_cards],
                        "current_hand": [c.name for c in p.current_hand],
                        "memory_known_cards": []
                    } for p in env.players]
                }

                for pid in range(NUM_PLAYERS):

                    state = proc.encode_observation(
                        obs,
                        pid,
                        env.get_legal_actions(pid)
                    )
                    mask = proc.get_action_mask(env, pid)

                    if pid == 0:
                        state_tf = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
                        mask_tf = tf.convert_to_tensor(mask[None, :], dtype=tf.float32)

                        act_idx, logprob, value = agent.get_action_and_value(
                            state_tf,
                            mask_tf
                        )

                        action = proc.index_to_action(act_idx.numpy())

                        hero_traj["states"].append(state)
                        hero_traj["actions"].append(act_idx.numpy())
                        hero_traj["logprobs"].append(logprob.numpy())
                        hero_traj["values"].append(value.numpy())
                        hero_traj["masks"].append(mask)
                        hero_traj["rewards"].append(0.0)

                        actions[pid] = action

                    else:
                        valid_indices = np.where(mask == 1.0)[0]
                        rand_idx = np.random.choice(valid_indices)
                        actions[pid] = proc.index_to_action(rand_idx)

                _, _, done, _ = env.step(actions)

            scores = calculate_scores(env)
            scores_batch.append(scores[0])

            hero_traj["rewards"][-1] += scores[0] / 50.0

            rewards = np.array(hero_traj["rewards"])
            values = np.array(hero_traj["values"])
            advs = np.zeros_like(rewards)

            last_adv = 0.0
            last_val = 0.0

            for t in reversed(range(len(rewards))):
                delta = rewards[t] + GAMMA * last_val - values[t]
                advs[t] = last_adv = delta + GAMMA * GAE_LAMBDA * last_adv
                last_val = values[t]

            returns = advs + values

            batch_states.extend(hero_traj["states"])
            batch_actions.extend(hero_traj["actions"])
            batch_logprobs.extend(hero_traj["logprobs"])
            batch_returns.extend(returns)
            batch_advs.extend(advs)
            batch_masks.extend(hero_traj["masks"])

        batch_advs = np.array(batch_advs)
        batch_advs = (
            (batch_advs - batch_advs.mean())
            / (batch_advs.std() + 1e-8)
        )

        dataset = tf.data.Dataset.from_tensor_slices((
            np.array(batch_states),
            np.array(batch_actions),
            np.array(batch_logprobs),
            np.array(batch_returns).astype("float32"),
            batch_advs.astype("float32"),
            np.array(batch_masks)
        )).shuffle(2048).batch(BATCH_SIZE)

        for _ in range(EPOCHS_PER_UPDATE):
            for batch in dataset:
                agent.train_step(*batch, entropy_tf)

        print(
            f"Iter {iteration:4d} | "
            f"Avg Score: {np.mean(scores_batch):5.2f} | "
            f"Max Score: {np.max(scores_batch):5.2f}"
        )

        if iteration % 20 == 0:
            model.save("ppo_7wonders_random_only.keras")


if __name__ == "__main__":
    main()