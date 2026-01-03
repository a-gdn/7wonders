"""
Example usage of the 7 Wonders Game Environment.

This demonstrates how to use the GameEnv class for RL training.
"""

from seven_wonders.environment import GameEnv
from seven_wonders.constants import CardColor
from seven_wonders import scoring
import random


def test_basic_game():
    """Test a basic game with random actions."""
    print("Testing 7 Wonders Game Environment\n")
    
    # Create game with 3 players
    env = GameEnv(num_players=3, random_seed=42)
    
    # Reset to initial state
    observation = env.reset()
    
    print(f"Game initialized with {env.num_players} players")
    print(f"Starting observation: {observation}\n")
    
    # Play until game is over
    turn_count = 0
    while not env.is_done():
        # Get legal actions for each player
        actions = {}
        for player_id in range(env.num_players):
            legal_actions = env.get_legal_actions(player_id)
            # Random action selection
            action = random.choice(legal_actions)
            actions[player_id] = action
            print(f"Player {player_id} chose: {action}")
        
        # Execute step
        observation, rewards, done, info = env.step(actions)
        turn_count += 1
        
        print(f"After turn {turn_count}, current age: {observation['current_age']}, "
              f"turn: {observation['current_turn']}")
        print()
    
    # Game is over - calculate final scores
    scores = scoring.calculate_scores(env)
    print("\n" + "="*50)
    print("GAME OVER - FINAL SCORES")
    print("="*50)
    
    for player in env.players:
        score = scores[player.player_id]
        print(f"\nPlayer {player.player_id} ({player.wonder_name}):")
        print(f"  Total Score: {score}")
        print(f"  Military Tokens: {sum(player.military_tokens)}")
        print(f"  Treasury: {player.coins} coins ({player.coins // 3} VP)")
        print(f"  Wonder Stages: {player.current_wonder_stage}")
        print(f"  Built Cards: {len(player.built_cards)}")
        print(f"  Science Symbols: {player.science}")
    
    winner_id = max(scores, key=scores.get)
    print(f"\n🏆 Winner: Player {winner_id} with {scores[winner_id]} points!")


def test_game_mechanics():
    """Test specific game mechanics."""
    print("\nTesting Game Mechanics\n")
    
    env = GameEnv(num_players=3, random_seed=123)
    observation = env.reset()
    
    player = env.players[0]
    print(f"Player 0 wonder: {player.wonder_name}")
    print(f"Starting resource: wood" if player.production["wood"] > 0 else "No starting resource")
    print(f"Coins: {player.coins}")
    print(f"Initial hand size: {len(player.current_hand)}\n")
    
    # Test legal actions
    legal_actions = env.get_legal_actions(0)
    print(f"Legal actions for Player 0: {legal_actions[:5]}...")  # Show first 5
    
    # Simulate a few turns
    for turn in range(3):
        print(f"\n--- Turn {turn + 1} ---")
        env.render()
        
        # Get actions
        actions = {}
        for player_id in range(env.num_players):
            legal = env.get_legal_actions(player_id)
            actions[player_id] = random.choice(legal)
        
        env.step(actions)


def test_rl_integration():
    """
    Example of how to integrate with RL training.
    """
    print("\nRL Integration Example\n")
    
    def get_rl_observation(env_obs):
        """Convert game observation to RL-friendly format."""
        player_obs = env_obs["players"][0]  # Focus on player 0
        
        # Example features for RL agent
        features = [
            player_obs["coins"],
            sum(player_obs["production"].values()),
            sum(player_obs["science"].values()),
            player_obs["military_shields"],
            player_obs["wonder_stage_progress"],
            len(player_obs["built_card_names"]),
            env_obs["current_turn"],
        ]
        return features
    
    env = GameEnv(num_players=3)
    obs = env.reset()
    
    # Convert to RL format
    rl_obs = get_rl_observation(obs)
    print(f"RL observation vector (first 5 components): {rl_obs[:5]}")
    print(f"Full observation vector: {rl_obs}")
    
    # Get action space
    legal_actions = env.get_legal_actions(0)
    print(f"Action space size: {len(set(legal_actions))}")


def _get_colored_text(text, color_name):
    """Apply ANSI color to text based on card color name."""
    colors = {
        "brown": "\033[38;5;94m",   # Brown (Raw Materials)
        "grey": "\033[90m",         # Grey (Manufactured Goods)
        "gray": "\033[90m",
        "blue": "\033[94m",         # Blue (Civilian)
        "green": "\033[92m",        # Green (Science)
        "yellow": "\033[93m",       # Yellow (Commercial)
        "red": "\033[91m",          # Red (Military)
        "purple": "\033[95m",       # Purple (Guilds)
    }
    code = colors.get(str(color_name).lower(), "\033[0m")
    return f"{code}{text}\033[0m"


def play_human_vs_bots():
    """Play a game interactively as Player 0 against random bots."""
    print("Starting Interactive 7 Wonders Game")
    print("You are Player 0.")
    
    num_players = 3
    env = GameEnv(num_players=num_players)
    
    # Build card color map
    card_colors = {}
    if hasattr(env, "cards_data"):
        for card in env.cards_data:
            card_colors[card["name"]] = card["color"]

    obs = env.reset()
    
    while not env.is_done():
        print(f"\n{'='*20} AGE {obs['current_age'] + 1} - TURN {obs['current_turn'] + 1} {'='*20}")
        
        # Display all players info
        for pid in range(num_players):
            p_obs = obs["players"][pid]
            p_obj = env.players[pid]
            wonder_side = p_obs.get("wonder_side", "day")
            
            role = "You" if pid == 0 else f"Bot {pid + 1}"
            print(f"\n--- Player {pid + 1} ({role}) ---")
            print(f"Wonder: {p_obj.wonder_name} ({wonder_side}) (Stage {p_obs['wonder_stage_progress']})")
            print(f"Coins: {p_obs['coins']} | Shields: {p_obs['military_shields']}")
            print(f"Resources: {p_obs['production']}")
            print(f"Science: {p_obs['science']}")
            print(f"Built Cards: {p_obs['built_card_names']}")
            print(f"Hand: {p_obs['current_hand']}")

        actions = {}
        
        # Human Player (Player 0)
        legal_actions = env.get_legal_actions(0)
        legal_actions.sort()
        
        print(f"\n--- Your Turn (Player 1) ---")
        print("Available Actions:")
        for i, act in enumerate(legal_actions):
            if act.startswith("discard_"):
                c_name = act.replace("discard_", "")
            elif act.startswith("wonder_stage_"):
                c_name = act.replace("wonder_stage_", "")
            else:
                c_name = act
            
            color_name = card_colors.get(c_name, "default")
            print(f"  {i}: {_get_colored_text(act, color_name)}")
        
        valid = False
        while not valid:
            try:
                choice = input("\nEnter action index: ")
                idx = int(choice)
                if 0 <= idx < len(legal_actions):
                    actions[0] = legal_actions[idx]
                    valid = True
                else:
                    print("Invalid index. Try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Bot Players
        for pid in range(1, num_players):
            bot_legal = env.get_legal_actions(pid)
            actions[pid] = random.choice(bot_legal)
        
        # Step
        obs, rewards, done, info = env.step(actions)
        
        print("\nActions taken:")
        for pid, act in actions.items():
            player_type = "You" if pid == 0 else f"Bot {pid + 1}"
            print(f"  {player_type}: {act}")

    # Game Over
    print("\n" + "="*50)
    print("GAME OVER")
    scores = scoring.calculate_scores(env)
    for pid in range(num_players):
        print(f"Player {pid + 1}: {scores[pid]} points")
        
    winner = max(scores, key=scores.get)
    if winner == 0:
        print("\n🏆 You Won! (Player 1)")
    else:
        print(f"\nPlayer {winner + 1} Won. Better luck next time!")


if __name__ == "__main__":
    # Run tests
    # test_basic_game()
    # test_game_mechanics()
    # test_rl_integration()
    play_human_vs_bots()
