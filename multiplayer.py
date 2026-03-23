"""
7 Wonders Multiplayer Game Interface
Play with real human players - everyone manually chooses their actions
"""

import os
import random
from typing import Dict, List, Tuple, Optional

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from seven_wonders.environment import GameEnv
from seven_wonders.scoring import calculate_scores

# ==========================================
#          GAME MANAGER
# ==========================================
class MultiplayerGame:
    def __init__(self, num_players: int):
        self.num_players = num_players
        self.env = GameEnv(num_players=num_players, expansions=[])
        self.player_names = []
        self.wonder_selections = {}
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def setup_players(self):
        """Setup player names and wonder selections"""
        print("\n" + "="*70)
        print("🎲 7 WONDERS - MULTIPLAYER SETUP")
        print("="*70)
        
        # Get player names
        print("\nEnter player names:")
        for i in range(self.num_players):
            while True:
                name = input(f"  Player {i + 1} name: ").strip()
                if name:
                    self.player_names.append(name)
                    break
        
        # Get wonder selections for each player
        wonders = self.get_available_wonders()
        available = wonders.copy()
        
        print("\n" + "-"*70)
        print("🏛️  Wonder Selection")
        print("-"*70)
        
        for i in range(self.num_players):
            self.clear_screen()
            print(f"\n🎲 {self.player_names[i]}'s turn to pick a wonder\n")
            print("Available wonders:")
            for idx, wonder in enumerate(available, 1):
                print(f"  {idx}. {wonder}")
            
            while True:
                try:
                    choice = int(input(f"\nChoose wonder (1-{len(available)}): "))
                    if 1 <= choice <= len(available):
                        chosen_wonder = available.pop(choice - 1)
                        break
                except ValueError:
                    pass
                print("Invalid choice. Try again.")
            
            # Choose side
            print("\nChoose side:")
            print("  1. Day")
            print("  2. Night")
            while True:
                try:
                    side_choice = int(input("Choose side (1-2): "))
                    if side_choice in [1, 2]:
                        side = "day" if side_choice == 1 else "night"
                        break
                except ValueError:
                    pass
                print("Invalid choice. Try again.")
            
            self.wonder_selections[i] = (chosen_wonder, side)
            self.env.players[i].wonder_name = chosen_wonder
            self.env.players[i].wonder_side = side
        
        print("\n✅ Setup complete!")
        print("\nWonder selections:")
        for i, (wonder, side) in self.wonder_selections.items():
            print(f"  {self.player_names[i]}: {wonder} ({side})")
    
    def get_available_wonders(self) -> List[str]:
        """Get list of unique wonders"""
        unique_wonders = set(w["name"] for w in self.env.wonder_data)
        return sorted(list(unique_wonders))
    
    def env_to_dict(self):
        """Convert environment to observation dict"""
        return {
            "current_age": self.env.current_age,
            "current_turn": self.env.current_turn,
            "players": [{
                "player_name": self.player_names[pid],
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
                "memory_known_cards": [c if isinstance(c, str) else c.name for c in p.memory_known_cards]
            } for pid, p in enumerate(self.env.players)]
        }
    
    def display_game_state(self, current_player_id: int = None):
        """Display full game state"""
        obs = self.env_to_dict()
        
        print("\n" + "="*70)
        print(f"Age {obs['current_age'] + 1}, Turn {obs['current_turn'] + 1}")
        print("="*70)
        
        for pid, p in enumerate(obs["players"]):
            marker = "👤 YOU" if pid == current_player_id else f"   "
            print(f"\n{marker} {p['player_name']}: {p['wonder_name']} ({p['wonder_side'].upper()})")
            print(f"   Coins: {p['coins']} | Shields: {p['shields']} | Wonder stage: {p['wonder_stage_progress']}")
            
            built_str = ", ".join(p["built_card_names"]) if p["built_card_names"] else "none"
            print(f"   Built cards: {built_str}")
            
            prod = p["production"]
            prod_str = ", ".join([f"{v}{k[0]}" for k, v in prod.items() if v > 0])
            if prod_str:
                print(f"   Production: {prod_str}")
    
    def display_hand(self, player_id: int):
        """Display player's hand"""
        obs = self.env_to_dict()
        player = self.env.players[player_id]
        
        print(f"\n🎴 Your hand ({len(player.current_hand)} cards):")
        for i, card in enumerate(player.current_hand, 1):
            print(f"  {i}. {card.name}")
    
    def get_legal_actions_display(self, player_id: int) -> Dict[int, str]:
        """Display and return legal actions"""
        legal_actions = self.env.get_legal_actions(player_id)
        action_map = {}
        
        print(f"\n🎯 Available Actions:")
        for idx, action in enumerate(legal_actions, 1):
            action_map[idx] = action
            
            if action.startswith("wonder_stage_"):
                card_name = action.replace("wonder_stage_", "")
                print(f"  {idx}. Build on wonder with: {card_name}")
            elif action.startswith("discard_"):
                card_name = action.replace("discard_", "")
                print(f"  {idx}. Discard: {card_name} (gain 3 coins)")
            else:
                print(f"  {idx}. Build: {action}")
        
        return action_map
    
    def get_player_action(self, player_id: int) -> str:
        """Get action from human player"""
        self.clear_screen()
        
        print("\n" + "="*70)
        print(f"🎮 {self.player_names[player_id]}'s turn")
        print("="*70)
        
        self.display_game_state(player_id)
        self.display_hand(player_id)
        
        action_map = self.get_legal_actions_display(player_id)
        
        while True:
            try:
                choice = int(input(f"\nEnter action number (1-{len(action_map)}): "))
                if choice in action_map:
                    return action_map[choice]
            except ValueError:
                pass
            print("Invalid choice. Try again.")
    
    def play(self):
        """Main game loop"""
        self.clear_screen()
        self.setup_players()
        
        # Reset environment with selected wonders
        self.env.reset()
        for i, (wonder, side) in self.wonder_selections.items():
            self.env.players[i].wonder_name = wonder
            self.env.players[i].wonder_side = side
        
        print("\n✅ Game starting!")
        input("Press Enter to begin...")
        
        # Main game loop
        turn_count = 0
        while not self.env.is_done():
            actions = {}
            
            # Get action from each player
            for pid in range(self.num_players):
                action = self.get_player_action(pid)
                actions[pid] = action
                print(f"\n✅ {self.player_names[pid]} chose: {action}")
                input("Press Enter for next player...")
            
            # Execute turn
            observation, rewards, done, info = self.env.step(actions)
            turn_count += 1
        
        # Game over - show results
        self.display_results()
    
    def display_results(self):
        """Display final game results"""
        scores = calculate_scores(self.env)
        
        self.clear_screen()
        print("\n" + "="*70)
        print("🏁 GAME OVER - FINAL RESULTS")
        print("="*70)
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (player_id, score) in enumerate(sorted_scores, 1):
            player = self.env.players[player_id]
            print(f"\n{rank}. 👑 {self.player_names[player_id]}: {score} points")
            print(f"   Wonder: {player.wonder_name} ({player.wonder_side})")
            print(f"   Built: {len(player.built_cards)} cards | Coins: {player.coins}")
        
        print("\n" + "="*70)
        print(f"🎉 {self.player_names[sorted_scores[0][0]]} wins!")
        print("="*70 + "\n")

# ==========================================
#          MAIN ENTRY POINT
# ==========================================
def main():
    print("\n" + "="*70)
    print("🎲 7 WONDERS - MULTIPLAYER")
    print("="*70)
    
    # Get number of players
    while True:
        try:
            num_players = int(input("\nHow many players? (3-7): "))
            if 3 <= num_players <= 7:
                break
        except ValueError:
            pass
        print("Invalid. Enter 3-7.")
    
    # Create and play game
    game = MultiplayerGame(num_players=num_players)
    game.play()

if __name__ == "__main__":
    main()
