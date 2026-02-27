import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

from seven_wonders.environment import GameEnv
from seven_wonders.scoring import calculate_scores
from train import DataProcessor 

# ==========================================
#            GPU (Mac Metal)
# ==========================================
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==========================================
#            CONFIGURATION
# ==========================================
NUM_PLAYERS = 4
NUM_GAMES = 1000

HERO_MODEL_PATH = "ppo_7wonders_latest.keras"
RANDOM_AI_MODEL_PATH = "ppo_7wonders_random_only.keras"

# ==========================================
#            HELPER FUNCTIONS
# ==========================================
def get_model_action(model, proc, env, obs, pid):
    state = proc.encode_observation(obs, pid, env.get_legal_actions(pid))
    mask = proc.get_action_mask(env, pid)

    state_tf = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
    mask_tf = tf.convert_to_tensor(mask[None, :], dtype=tf.float32)

    logits, _ = model(state_tf, training=False)
    masked_logits = logits + (1.0 - mask_tf) * -1e9
    
    act_idx = tf.argmax(masked_logits, axis=1)[0].numpy()
    return proc.index_to_action(act_idx)

def get_random_action(env, pid):
    return random.choice(env.get_legal_actions(pid))

# ==========================================
#            EVALUATION LOOP
# ==========================================
def main():
    print("Loading models...")
    hero_model = keras.models.load_model(HERO_MODEL_PATH, compile=False)
    random_ai_model = keras.models.load_model(RANDOM_AI_MODEL_PATH, compile=False)
    
    env = GameEnv(num_players=NUM_PLAYERS)
    proc = DataProcessor(env)

    # Dictionary to track statistics
    stats = {
        "Hero (Self-Play)": {"wins": 0, "total_score": 0.0, "count_in_game": 1},
        "Baseline (Random-Trained)": {"wins": 0, "total_score": 0.0, "count_in_game": 2},
        "Pure Random": {"wins": 0, "total_score": 0.0, "count_in_game": 1}
    }

    print(f"\n🚀 Starting Evaluation: {NUM_GAMES} Games")

    for game in range(1, NUM_GAMES + 1):
        env.reset()
        done = False
        
        # 0: Hero, 1: Baseline, 2: Random
        seating_order = [0, 1, 1, 1]
        random.shuffle(seating_order)

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
                p_type = seating_order[pid]
                if p_type == 0:
                    actions[pid] = get_model_action(hero_model, proc, env, obs, pid)
                elif p_type == 1:
                    actions[pid] = get_model_action(random_ai_model, proc, env, obs, pid)
                else:
                    actions[pid] = get_random_action(env, pid)

            _, _, done, _ = env.step(actions)

        # SCORING LOGIC
        raw_scores = calculate_scores(env)
        
        # Handle different return types (dict vs list) from scoring.py
        if isinstance(raw_scores, dict):
            final_scores = [float(raw_scores[i]) for i in range(NUM_PLAYERS)]
        else:
            final_scores = [float(s) for s in raw_scores]

        max_score = max(final_scores)
        
        # Sanity check for the first few games
        if game <= 5:
            print(f"Game {game} Scores: {final_scores} | Seating: {seating_order}")

        for pid in range(NUM_PLAYERS):
            p_type = seating_order[pid]
            score = final_scores[pid]
            
            if p_type == 0:
                label = "Hero (Self-Play)"
            elif p_type == 1:
                label = "Baseline (Random-Trained)"
            else:
                label = "Pure Random"
            
            stats[label]["total_score"] += score
            if score == max_score:
                stats[label]["wins"] += 1

        if game % 100 == 0:
            print(f"Completed {game}/{NUM_GAMES} games...")

    # ==========================================
    #               FINAL OUTPUT
    # ==========================================
    print("\n" + "="*50)
    print(f"{'PLAYER TYPE':<28} | {'WIN %':<8} | {'AVG SCORE':<10}")
    print("-" * 50)
    
    for label, data in stats.items():
        # Divide by total occurrences of that agent type across all games
        denominator = NUM_GAMES * data["count_in_game"]
        win_rate = (data["wins"] / denominator) * 100
        avg_score = data["total_score"] / denominator
        
        print(f"{label:<28} | {win_rate:>6.1f}% | {avg_score:>9.1f}")
    print("="*50)

if __name__ == "__main__":
    main()