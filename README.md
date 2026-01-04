# 7 Wonders RL Environment

A complete Python implementation of the 7 Wonders board game, designed as a Reinforcement Learning (RL) environment.

## Overview

This project implements the official rules of 7 Wonders for 3-7 players. It supports simultaneous turns, resource management, wonder construction, military conflicts, and full scoring. It includes a PPO training script (`train.py`) demonstrating how to train agents using self-play.

## Project Structure

The codebase is organized as a Python package `seven_wonders`.

```
.
├── seven_wonders/
│   ├── environment.py      # Main GameEnv class
│   ├── player.py           # Player state definition
│   ├── resource.py         # Resource trading and cost logic
│   ├── scoring.py          # Final score calculation
│   ├── setup.py            # Game initialization and deck management
│   └── db/                 # JSON databases for cards and wonders
├── train.py                # PPO Training script (AlphaZero-style)
├── test_game.py            # Usage examples and testing
└── README.md
```

## Getting Started

### Basic Usage

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

## Environment Details

### Observation Space

The `get_observation()` method returns a dictionary representing the global game state and individual player states.

```python
{
    "current_age": 0,          # 0, 1, 2 (representing Ages I, II, III)
    "current_turn": 0,         # 0-5 (6 turns per age)
    "num_players": 4,
    "discard_pile": ["CardName", ...],
    "players": [
        {
            "player_id": 0,
            "coins": 3,
            "wonder_name": "Gizeh",
            "wonder_side": "day",
            "production": { "wood": 0, "stone": 0, "ore": 0, "clay": 0, "glass": 0, "papyrus": 0, "textile": 0 },
            "science": { "compass": 0, "gear": 0, "tablet": 0 },
            "shields": 0,
            "military_tokens_score": 0,
            "wonder_stage_progress": 0,
            "cards_played": 0,
            "built_card_names": ["Chantier", ...],
            "current_hand": ["Card A", "Card B", ...],  # Cards currently held
            "memory_known_cards": ["Card A", "Card C", ...] # Cards seen in hand this age
        },
        # ... other players
    ]
}
```

### Action Space

Actions are strings passed in a dictionary `{player_id: action_string}`.

| Action Type | Format | Description |
|-------------|--------|-------------|
| **Build Structure** | `"{CardName}"` | Build the card (e.g., `"Baths"`). Requires resources/coins. |
| **Build Wonder** | `"wonder_stage_{CardName}"` | Use card as a marker to build next wonder stage. |
| **Discard** | `"discard_{CardName}"` | Discard card for 3 coins. |
| **Auto Discard** | `"discard"` | Discards the first available card (fallback). |

Use `env.get_legal_actions(player_id)` to retrieve valid moves for the current state, which accounts for costs, prerequisites (chains), and duplicates.

### Rewards

The environment returns `0` rewards during the game steps. 
- **RL Implementation**: You should implement reward shaping (e.g., delta in VP) or use the final game score.
- **Scoring**: Use `seven_wonders.scoring.calculate_scores(env)` at the end of the game to get official 7 Wonders scores.

## Features Implemented

- **Core Mechanics**: 3 Ages, card drafting (hand rotation), simultaneous actions.
- **Economy**: Resource production, trading with neighbors (2 coins or 1 with trading posts).
- **Construction**: Resource costs, coin costs, and **free construction chains**.
- **Wonders**: All base game wonders (A & B sides) with unique effects.
- **Scoring Categories**:
  - Military Conflicts (Victory/Defeat tokens)
  - Treasury (3 coins = 1 VP)
  - Wonder Stages
  - Civilian Structures (Blue)
  - Scientific Structures (Green) - Set and symbol scoring
  - Commercial Structures (Yellow)
  - Guilds (Purple) - Neighbor-dependent scoring

## Training

A PPO (Proximal Policy Optimization) training script is included in `train.py`. It demonstrates:
1.  **DataProcessor**: Converting dictionary observations to tensor representations.
2.  **Self-Play**: Agents playing against themselves to generate data.
3.  **Arena**: Evaluating new models against previous best models.

To run training:
```bash
python train.py
```

## Testing

Run the test suite to verify game mechanics and play a human-vs-bot game:

```bash
python test_game.py
```