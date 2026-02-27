# 7 Wonders RL Environment

A complete Python implementation of the 7 Wonders board game, designed as a Reinforcement Learning (RL) environment. This project includes scripts to train an AI agent using Proximal Policy Optimization (PPO).

## Features

- **Complete Game Logic:** Implements the official 7 Wonders rules for 3-7 players, including simultaneous turns, resource management, wonder construction, military conflicts, and scoring.
- **Reinforcement Learning Ready:** Provides a `GameEnv` class compatible with RL training libraries.
- **PPO Training Scripts:** Includes scripts for training agents using self-play and against random opponents.
- **Model Evaluation:** A script to evaluate and compare the performance of different trained models.

## Project Structure

```
.
├── seven_wonders/
│   ├── environment.py      # Main GameEnv class for the 7 Wonders game
│   ├── models.py           # Data classes for Card and WonderStage
│   ├── player.py           # Player state definition
│   ├── resource.py         # Resource trading and cost logic
│   ├── scoring.py          # Final score calculation
│   ├── setup.py            # Game initialization and deck management
│   └── db/                 # JSON databases for cards and wonders
├── train.py                # PPO training script with self-play
├── train_against_random.py # PPO training script against random opponents
├── eval.py                 # Script to evaluate trained models
├── test_game.py            # Usage examples and tests
└── README.md
```

## Getting Started

### Installation

This project requires TensorFlow for the PPO agent.

```bash
pip install tensorflow
```

### Basic Usage

You can interact with the game environment directly, as shown below:

```python
from seven_wonders.environment import GameEnv
from seven_wonders import scoring
import random

# Initialize environment for 4 players
env = GameEnv(num_players=4, random_seed=42)
obs = env.reset()

# Main Game Loop
while not env.is_done():
    actions = {}
    
    # Get actions for all players (Simultaneous turns)
    for player_id in range(env.num_players):
        legal_moves = env.get_legal_actions(player_id)
        # Random agent
        actions[player_id] = random.choice(legal_moves)
    
    # Step the environment
    obs, rewards, done, info = env.step(actions)

# Calculate final scores
final_scores = scoring.calculate_scores(env)
winner = max(final_scores, key=final_scores.get)
print(f"Game Over. Winner: Player {winner} with {final_scores[winner]} points.")
```

## Training the AI

This project provides two main scripts for training the AI agent.

### Training with Self-Play

The `train.py` script uses a curriculum learning approach. The agent starts by playing against random opponents and gradually transitions to playing against a pool of its own past selves.

To start the training:
```bash
python train.py
```
The trained model will be saved as `ppo_7wonders_latest.keras`.

### Training Against Random Opponents

The `train_against_random.py` script trains the agent exclusively against random opponents. This is useful for establishing whether the model can achieve high scores.

To start this training:
```bash
python train_against_random.py
```
This script saves the model as `ppo_7wonders_random_only.keras`.

## Evaluating Models

The `eval.py` script is used to compare the performance of different models. It runs a specified number of games and reports the win rates and average scores for each model. By default, it compares:
- The self-play model (`ppo_7wonders_latest.keras`)
- The baseline random-trained model (`ppo_7wonders_random_only.keras`)
- A pure random agent

To run the evaluation:
```bash
python eval.py
```

## Environment Details

### Observation Space

The `get_observation()` method returns a dictionary representing the global game state and individual player states.

```python
{
    "current_age": 0,
    "current_turn": 0,
    "num_players": 4,
    "discard_pile": ["CardName", ...],
    "players": [
        {
            "player_id": 0,
            "coins": 3,
            "wonder_name": "Gizeh",
            "wonder_side": "day",
            "production": { "wood": 1, ... },
            "science": { "compass": 1, ... },
            "shields": 0,
            "military_tokens_score": 0,
            "wonder_stage_progress": 0,
            "built_card_names": ["Chantier", ...],
            "current_hand": ["Card A", "Card B", ...],
            "memory_known_cards": ["Card A", "Card C", ...]
        },
        // ... other players
    ]
}
```

### Action Space

Actions are strings passed in a dictionary `{player_id: action_string}`.

| Action Type | Format | Description |
|---|---|---|
| **Build Structure** | `"{CardName}"` | Build the card (e.g., `"Baths"`). |
| **Build Wonder** | `"wonder_stage_{CardName}"` | Use a card to build the next wonder stage. |
| **Discard** | `"discard_{CardName}"` | Discard a card for 3 coins. |

Use `env.get_legal_actions(player_id)` to get a list of valid moves.

### Rewards

The base environment returns a reward of `0` at each step. The final score should be used as the primary reward signal.

The `train.py` script uses a shaped reward function, which includes small positive rewards for building structures and negative rewards for discarding, in addition to the final score. This can help guide the agent during training.
