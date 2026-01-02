# 7 Wonders Game Environment

A complete implementation of the 7 Wonders board game as a Reinforcement Learning (RL) environment.

## Overview

This environment implements all the official rules of 7 Wonders for 3-7 players. It is designed to be compatible with RL training frameworks.

## Game Features Implemented

### Core Mechanics

- **3 Ages**: Games are divided into 3 ages, each with 6 turns
- **Simultaneous Actions**: All players select cards simultaneously each turn
- **Card Types**: 
  - Brown cards (raw materials)
  - Grey cards (manufactured goods)
  - Blue cards (civilian structures)
  - Yellow cards (commercial structures)
  - Red cards (military structures)
  - Green cards (scientific structures)
  - Purple cards (guilds)

### Resource Management

- **Production**: Players produce resources from wonders, brown cards, and grey cards
- **Commerce**: Buy missing resources from neighbors for 2 coins (or 1 with trading cards)
- **Coins**: Used to pay for structures, purchase resources, and score VP at end of game

### Wonder Construction

- **Wonder Stages**: Each wonder has multiple stages (2-4 depending on the wonder and side)
- **Stage Building**: Pay resource cost to build stages in order
- **Stage Effects**: Stages can provide various effects (VP, production, coins, military shields, science symbols, etc.)

### Military System

- **Military Shields**: Red cards provide shields
- **Conflict Resolution**: At end of each age, compare shields with neighbors
- **Tokens**: Gain +1/+3/+5 VP tokens or -1 defeat tokens

### Scoring System

Points are calculated in order:
1. **Military Conflicts**: Sum of victory/defeat tokens (-1, 1, 3, 5)
2. **Treasury**: 3 coins = 1 VP (rounded down)
3. **Wonder**: VP from completed stages
4. **Civilian Structures**: VP from blue cards
5. **Scientific Structures**: VP from green cards (symbol sets)
6. **Commercial Structures**: VP from yellow cards
7. **Guilds**: VP from purple cards based on neighbor configurations

### Scientific Scoring

- **Identical Symbols**: n identical symbols = n² VP
- **Different Symbols**: 3 different symbols = 7 VP (can score multiple sets)
- **Symbols**: Compass, Gear, Tablet

### Free Construction

- **Chain Prerequisites**: Some cards can be built free if you built their prerequisite
- **Wonder Stages**: Built in order, can be built in any age

## Usage

### Basic Setup

```python
from GameEnv import GameEnv

# Create environment with 3 players
env = GameEnv(num_players=3, random_seed=42)

# Reset to initial state
observation = env.reset()
```

### Playing the Game

```python
# Get game state
obs = env.get_observation()

# Get legal actions for each player
legal_actions = {}
for player_id in range(env.num_players):
    legal_actions[player_id] = env.get_legal_actions(player_id)

# Execute one turn
actions = {0: "Chantier", 1: "discard", 2: "Autel"}
observation, rewards, done, info = env.step(actions)

# Check if game is over
if done:
    scores = env.calculate_scores()
```

## Observation Format

The observation returned by `get_observation()` contains:

```python
{
    "current_age": 0,          # 0-2 (ages 1-3)
    "current_turn": 0,         # 0-5 (turn 1-6 per age)
    "num_players": 3,
    "players": [
        {
            "player_id": 0,
            "coins": 3,
            "production": {
                "wood": 0, "stone": 0, "ore": 0, "clay": 1,
                "glass": 0, "papyrus": 0, "textile": 0
            },
            "science": {"compass": 0, "gear": 0, "tablet": 0},
            "military_shields": 0,
            "wonder_stage_progress": 0,
            "max_wonder_stages": 3,
            "cards_played": 0,
            "current_hand_size": 7,
            "military_tokens_score": 0,
            "built_card_names": []
        },
        # ... more players
    ]
}
```

## Action Format

Actions are strings representing player choices:

- **Build Structure**: Card name (e.g., "Chantier", "Autel")
- **Build Wonder Stage**: Unused card can be used as marker
- **Discard**: "discard" - gain 3 coins

## Legal Actions

`get_legal_actions(player_id)` returns list of valid actions for a player:
- Can only build cards you don't already have
- Can only build structures you can afford (resources + coins)
- Can only build wonder stages in order
- Can always discard

## Integration with RL

### State Representation

Convert observations to feature vectors:

```python
def get_rl_features(obs, player_id):
    player_obs = obs["players"][player_id]
    
    features = [
        player_obs["coins"],
        sum(player_obs["production"].values()),
        sum(player_obs["science"].values()),
        player_obs["military_shields"],
        player_obs["wonder_stage_progress"],
        len(player_obs["built_card_names"]),
        obs["current_turn"],
    ]
    return features
```

### Reward Function

Currently returns empty dict. Implement based on your RL objective:

```python
def calculate_rewards(env, last_obs, current_obs):
    rewards = {}
    for player_id in range(env.num_players):
        # Example: reward military advancement
        old_shields = last_obs["players"][player_id]["military_shields"]
        new_shields = current_obs["players"][player_id]["military_shields"]
        rewards[player_id] = new_shields - old_shields
    return rewards
```

### Terminal Conditions

Game ends after Age 3 (turn 6). Check with `env.is_done()`.

## Card Database

Cards are loaded from `db/cards.json` with structure:

```json
{
  "name": "Chantier",
  "age": 1,
  "player_requirement": 3,
  "color": "brown",
  "cost": {"coins": 0},
  "chain_from": null,
  "effect": {"production": {"wood": 1}}
}
```

## Wonder Database

Wonder boards loaded from `db/wonder_boards.json` with structure:

```json
{
  "name": "Alexandrie",
  "start_resource": "glass",
  "sides": {
    "day": [
      {
        "stage": 1,
        "cost": {"stone": 2},
        "effect": {"vp": 3}
      },
      ...
    ],
    "night": [...]
  }
}
```

## Wonders Available

- Alexandrie (glass)
- Babylon (wood)
- Éphesos (papyrus)
- Gizah (stone)
- Halikarnassos (textile)
- Olympe (ore)
- Rhodes (clay)

## Rules Implementation Status

### ✅ Fully Implemented

- Card types and colors
- Resource production and management
- Coin system
- Wonder construction and stages
- Military shields and conflicts
- Scientific symbol tracking
- Card effects (production, VP, military, science)
- Chain prerequisites (free construction)
- Commerce system basics
- Simultaneous card selection
- Hand rotation (age-dependent direction)
- Age progression (I → II → III)
- Final scoring (all 7 categories)
- Legal action validation

### 📝 Partially Implemented

- Commerce (resources can be bought at fixed cost, but neighbor preference not prioritized)
- Immediate coin effects (basic version, more complex cases may need tuning)

### ⚠️ Notes

- Commercial structures and guild effects are computed at game end
- Some card effects may require tuning for complex interactions
- Trading with neighbors uses fixed cost (2 coins standard, 1 with certain yellow cards)

## Testing

Run test suite:

```bash
python test_game.py
```

Tests include:
- Basic game completion (3 players)
- Game mechanics validation
- RL integration example

## Architecture

### Main Classes

- **GameEnv**: Main environment class managing game state and transitions
- **PlayerCity**: Represents a single player's city with resources, cards, wonder
- **Card**: Card data structure with effects and costs
- **WonderStage**: Wonder stage with cost and effects

### Enums

- **CardColor**: Brown, grey, blue, yellow, red, green, purple
- **ScienceSymbol**: Compass, gear, tablet
- **GameState**: Setup, age active, military resolution, game over

## File Structure

```
/Users/alexis/code/7wonders/
├── GameEnv.py              # Main environment implementation
├── GameSetup.py            # Original setup utilities (legacy)
├── test_game.py            # Example usage and tests
└── db/
    ├── cards.json          # Card definitions
    └── wonder_boards.json   # Wonder board definitions
```

## Parameters

### GameEnv Constructor

```python
GameEnv(num_players=3, random_seed=None)
```

- **num_players**: 3-7 players required
- **random_seed**: For reproducible games

## Future Enhancements

1. Trading costs reduction via yellow cards
2. Complex guild effects (variable based on board state)
3. Wonder destruction/degradation (if implementing expansions)
4. More sophisticated commerce simulation
5. Multi-threaded simultaneous action resolution
6. Render modes (ASCII, visual)
