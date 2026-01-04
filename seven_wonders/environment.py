"""
7 Wonders Game Environment for Reinforcement Learning

This module implements a complete game environment for the 7 Wonders board game,
following all official game rules. It is designed to be compatible with RL training.
"""

import json
import random
import os
from typing import Dict, List, Tuple, Optional, Set
from .constants import CardColor, GameState
from .models import Card, WonderStage
from .player import PlayerCity
from . import setup
from . import scoring
from . import resource as resource_manager

class GameEnv:
    """
    7 Wonders Game Environment.
    
    Implements the complete 7 Wonders board game rules for RL training.
    Supports 3-7 players.
    """
    
    def __init__(self, num_players: int = 4, random_seed: Optional[int] = None):
        """
        Initialize the game environment.
        
        Args:
            num_players: Number of players (3-7)
            random_seed: Optional seed for reproducibility
        """
        if not 3 <= num_players <= 7:
            raise ValueError(f"Number of players must be between 3 and 7, got {num_players}")
        
        self.num_players = num_players
        if random_seed is not None:
            random.seed(random_seed)
        
        # Load game data
        self.cards_data = setup.load_json("db/cards.json")
        self.wonder_data = setup.load_json("db/wonder_boards.json")

        # Game state
        self.state = GameState.SETUP
        self.current_age = 0
        self.current_turn = 0  # 0-5 per age
        self.players: List[PlayerCity] = []
        self.decks: Dict[int, List[Card]] = {}
        self.discard_pile: List[Card] = []
        
        # Coin bank (unlimited coins available per rules)
        # Coins can be exchanged as 1-coin or 3-coin denominations
        self.coin_bank_unlimited = True
        
        # Track cards selected this turn (by all players simultaneously)
        self.selected_cards: Dict[int, Optional[Card]] = {i: None for i in range(num_players)}
        self.selected_action_types: Dict[int, str] = {i: "discard" for i in range(num_players)}
    
    def reset(self) -> Dict:
        """
        Reset the game to initial state.
        
        Returns:
            Initial game observation for RL training
        """
        # Setup players with wonders
        self.players = setup.setup_players(self.num_players, self.wonder_data)
        
        # Setup and shuffle decks
        self.decks = setup.setup_decks(self.num_players, self.cards_data)
        
        # Initialize game state
        self.state = GameState.AGE_ACTIVE
        self.current_age = 0
        self.current_turn = 0
        self.discard_pile = []
        self.selected_cards = {i: None for i in range(self.num_players)}
        self.selected_action_types = {i: "discard" for i in range(self.num_players)}
        
        for p in self.players:
            p.memory_known_cards.clear()

        # Deal initial hands for Age I
        setup.deal_age_hand(self.players, self.decks, 0)
        
        return self.get_observation()
    
    def step(self, actions: Dict[int, str]) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one game step (one turn).
        
        Args:
            actions: Dictionary mapping player_id to action (card selection)
                    Action format: card_name or "discard" or "wonder_stage_X"
        
        Returns:
            observation, reward, done, info
        """
        if self.state != GameState.AGE_ACTIVE:
            raise ValueError(f"Cannot execute step in state {self.state}")
        
        is_final_turn = self.current_turn == 5
        
        # Phase 0: Update memory with cards currently in hand (seen this turn)
        for player in self.players:
            for card in player.current_hand:
                player.memory_known_cards.add(card.name)

        # Phase 1: Players select cards simultaneously
        self._select_cards(actions, is_final_turn)
        
        # Phase 2: Execute card actions simultaneously
        rewards = self._execute_card_actions()
        
        # Phase 2.5: Update memory - remove cards that are no longer in circulation (visible to all)
        visible_cards = set()
        for p in self.players:
            visible_cards.update(p.built_card_names)
        visible_cards.update(c.name for c in self.discard_pile)
        
        for p in self.players:
            p.memory_known_cards.difference_update(visible_cards)

        # Phase 3: Rotate hands (unless it's the final turn)
        if not is_final_turn:
            self._rotate_hands()
        
        # Move to next turn
        self.current_turn += 1
        
        done = False
        if self.current_turn >= 6:
            # End of age: military resolution
            self._resolve_military_conflicts()
            
            # Move to next age or end game
            self.current_age += 1
            if self.current_age >= 3:
                done = True
                self.state = GameState.GAME_OVER
            else:
                self.current_turn = 0
                # Reset memory for new age
                for p in self.players:
                    p.memory_known_cards.clear()
                setup.deal_age_hand(self.players, self.decks, self.current_age)
                # State remains AGE_ACTIVE
        
        return self.get_observation(), rewards, done, {}
    
    def _select_cards(self, actions: Dict[int, str], is_final_turn: bool = False):
        """
        Players select cards from their hands.
        
        For turn 6 of an age (final turn), players only receive 2 cards
        and must choose 1 (the other is discarded without coins).
        
        Per rules:
        - Turn 6: Players receive 2-card hand
        - Select 1 card to play
        - Discard 1 card WITHOUT gaining 3 coins
        
        Args:
            actions: Dictionary of player_id -> action (card name or "discard")
            is_final_turn: True if this is turn 6 (final turn of age)
        """
        for player_id, action in actions.items():
            player = self.players[player_id]
            hand = player.current_hand
            
            # Parse action intent and target card
            intent = "build_structure"
            target_name = action
            
            if action == "discard":
                intent = "discard"
                target_name = None # Will pick first available
            elif action.startswith("discard_"):
                intent = "discard"
                target_name = action.replace("discard_", "")
            elif action.startswith("wonder_stage_"):
                intent = "build_wonder"
                target_name = action.replace("wonder_stage_", "")
            
            # Find and select the card
            selected_card = None
            for card in hand:
                if target_name and card.name == target_name:
                    selected_card = card
                    break
                elif not target_name and intent == "discard":
                    # Fallback for generic discard: pick first card
                    selected_card = card
                    break
            
            if selected_card:
                self.selected_cards[player_id] = selected_card
                self.selected_action_types[player_id] = intent
                player.current_hand.remove(selected_card)
                
                # On turn 6 (final turn): discard remaining card without coins
                if is_final_turn and player.current_hand:
                    remaining_card = player.current_hand.pop(0)
                    self.discard_pile.append(remaining_card)
            else:
                # If action is invalid, discard
                if hand:
                    self.selected_cards[player_id] = hand[0]
                    self.selected_action_types[player_id] = "discard"
                    player.current_hand.pop(0)
                    
                    # On turn 6: discard remaining card without coins
                    if is_final_turn and player.current_hand:
                        remaining_card = player.current_hand.pop(0)
                        self.discard_pile.append(remaining_card)
    
    def _execute_card_actions(self) -> Dict[int, float]:
        """
        Execute the selected actions for all players simultaneously.
        
        Returns:
            Rewards for each player (empty dict for now, calculated at end of game)
        """
        rewards = {i: 0.0 for i in range(self.num_players)}
        deferred_credits = {}  # Map player_id -> amount (Commerce income available next turn)
        
        # Pass 1: Determine valid actions for all players based on START of turn state
        actions_to_execute = []
        
        for player_id, card in self.selected_cards.items():
            if card is None:
                continue
            
            player = self.players[player_id]
            intent = self.selected_action_types.get(player_id, "discard")
            
            # Execute based on intent, with fallbacks to discard if impossible
            if intent == "build_structure" and self._can_build_structure(player, card):
                actions_to_execute.append((player, "build_structure", card))
            elif intent == "build_wonder" and self._can_build_wonder_stage(player, card):
                actions_to_execute.append((player, "build_wonder", card))
            else:
                # Intent was discard OR build failed
                actions_to_execute.append((player, "discard", card))
        
        # Pass 2: Execute actions (using deferred credits for commerce)
        for player, action_type, card in actions_to_execute:
            if action_type == "build_structure":
                self._build_structure(player, card, deferred_credits)
            elif action_type == "build_wonder":
                self._build_wonder_stage(player, card, deferred_credits)
            else:
                self._discard_card(player, card)
        
        # Pass 3: Distribute commerce income
        for player_id, amount in deferred_credits.items():
            self.players[player_id].coins += amount
            
        return rewards
    
    def _can_build_structure(self, player: PlayerCity, card: Card) -> bool:
        """Check if a player can build a structure (card)."""
        # Can't build duplicate structures
        if card.name in player.built_card_names:
            return False
        
        # Check if it's free via chain
        if resource_manager.has_chain_prerequisite(player, card):
            return True
        
        # Check resource cost
        return resource_manager.can_afford_resources(self, player, card.cost)
    
    def _build_structure(self, player: PlayerCity, card: Card, deferred_credits: Dict[int, int] = None):
        """Build a structure for the player."""
        # Check if this is a free construction via chain
        is_free = resource_manager.has_chain_prerequisite(player, card)
        
        if not is_free:
            # Pay resource and coin costs
            cost = card.cost
            
            # Check and pay coins first
            if "coins" in cost:
                player.coins -= cost["coins"]
            
            # For resources: resources are NOT spent, but must be paid for if buying from neighbors
            resources_needed = {k: v for k, v in cost.items() if k != "coins"}
            if resources_needed:
                resource_manager.pay_resource_cost(self, player, resources_needed, deferred_credits)
        
        # Add card to built cards
        player.built_cards.append(card)
        player.built_card_names.add(card.name)
        
        # Apply card effects
        self._apply_card_effects(player, card)
    
    def _apply_card_effects(self, player: PlayerCity, card: Card):
        """Apply the effects of a built card."""
        # Normalize effects to a list to handle both single dict (legacy) and list of effects
        effects = card.effect if isinstance(card.effect, list) else [card.effect]
        
        for effect in effects:
            if "production" in effect:
                prod = effect["production"]
                for resource, count in prod.items():
                    player.production[resource] += count
            
            if "production_choice" in effect:
                # For RL, we'll pick the first choice for now
                # In a real implementation, this would be an action
                choice_effect = effect["production_choice"]
                options = choice_effect["options"]
                chosen_resource = options[0]
                player.production[chosen_resource] += 1
            
            if "vp" in effect:
                player.total_vp_from_cards += effect["vp"]
            
            if "shield" in effect:
                player.shields += effect["shield"]
            
            if "science" in effect:
                symbol = effect["science"]
                player.science[symbol] += 1
            
            if "coins" in effect:
                player.coins += effect["coins"]
            
            if "immediate_coins" in effect:
                coins_earned = self._calculate_immediate_coins(player, effect["immediate_coins"])
                player.coins += coins_earned
            
            if "trading" in effect:
                # Yellow card trading benefits - modifies trade costs
                # Stored for later use
                pass
        
        # Note: Variable VP effects (vp_per_card, vp_per_wonder_stage) are 
        # calculated at the end of the game in scoring.py, not here.
    
    def _calculate_immediate_coins(self, player: PlayerCity, effect: Dict) -> int:
        """Calculate coins earned immediately (e.g., from Vineyard)."""
        coin_type = effect.get("type")
        
        if coin_type == "coins_per_card":
            target_color = effect.get("color")
            multiplier = effect.get("multiplier", 1)
            target = effect.get("target", "self")
            
            count = 0
            if target == "neighbors_and_self":
                # Count own cards
                count += sum(1 for card in player.built_cards if card.color.value == target_color)
                # Count neighbor cards
                neighbors = self._get_neighbors(player.player_id)
                for neighbor_id in neighbors:
                    count += sum(1 for card in self.players[neighbor_id].built_cards 
                               if card.color.value == target_color)
            elif target == "self":
                count = sum(1 for card in player.built_cards if card.color.value == target_color)
            
            return count * multiplier
        
        return 0
    
    def _can_build_wonder_stage(self, player: PlayerCity, card: Card) -> bool:
        """Check if player can build the next wonder stage."""
        if player.current_wonder_stage >= len(player.wonder_stages):
            return False
        
        stage = player.wonder_stages[player.current_wonder_stage]
        if stage.built:
            return False
        
        # Check if can afford the stage cost
        cost = stage.cost
        
        # Check coin cost
        if "coins" in cost:
            if player.coins < cost["coins"]:
                return False
        
        # Check resource cost (can use own or buy from neighbors)
        own_production = player.production
        resources_needed = {k: v for k, v in cost.items() if k != "coins"}
        
        for resource, required in resources_needed.items():
            available_own = own_production.get(resource, 0)
            
            if available_own >= required:
                continue
            
            # Need to buy the difference
            needed_to_buy = required - available_own
            cost_per_resource = 2
            if resource_manager.has_trading_bonus(player):
                cost_per_resource = 1
            
            total_cost = needed_to_buy * cost_per_resource
            
            # Check if any neighbor can sell and if player has enough coins
            neighbor_can_provide = False
            for neighbor_id in self._get_neighbors(player.player_id):
                neighbor = self.players[neighbor_id]
                sellable = resource_manager.get_sellable_resources(self, neighbor)
                if resource in sellable:
                    neighbor_can_provide = True
                    break
            
            if not neighbor_can_provide or player.coins < total_cost:
                return False
        
        return True
    
    def _build_wonder_stage(self, player: PlayerCity, card: Card, deferred_credits: Dict[int, int] = None):
        """Build a wonder stage using the selected card as marker."""
        stage = player.wonder_stages[player.current_wonder_stage]
        
        # Pay the cost (coins only - resources are NOT spent)
        cost = stage.cost
        if "coins" in cost:
            player.coins -= cost["coins"]
        
        # Pay resource costs by buying from neighbors if needed (resources are NOT spent)
        resources_needed = {k: v for k, v in cost.items() if k != "coins"}
        if resources_needed:
            resource_manager.pay_resource_cost(self, player, resources_needed, deferred_credits)

        # Mark stage as built
        stage.built = True
        player.current_wonder_stage += 1
        
        # Remove from memory as it's consumed (private knowledge for this player)
        player.memory_known_cards.discard(card.name)
        
        # Apply stage effects
        self._apply_wonder_stage_effects(player, stage)
        
        # The card used as marker is hidden under the board.
        # It does NOT go to the discard pile (which is for 3-coin discards).
    
    def _apply_wonder_stage_effects(self, player: PlayerCity, stage: WonderStage):
        """Apply effects from completing a wonder stage."""
        # Normalize effects to a list
        effects = stage.effect if isinstance(stage.effect, list) else [stage.effect]
        
        for effect in effects:
            if "vp" in effect:
                player.total_vp_from_cards += effect["vp"]
            
            if "production" in effect:
                prod = effect["production"]
                if "options" in prod:
                    # TO DO: Choice of production - pick first for now
                    chosen = prod["options"][0]
                    player.production[chosen] += 1
                else:
                    for resource, count in prod.items():
                        player.production[resource] += count
            
            if "coins" in effect:
                player.coins += effect["coins"]
    
    def _discard_card(self, player: PlayerCity, card: Card):
        """Discard a card and gain 3 coins."""
        player.coins += 3
        self.discard_pile.append(card)
    
    def _rotate_hands(self):
        """Rotate hands to neighbors (direction depends on age)."""
        # Age 1 and 3: pass left (clockwise) - index increases
        # Age 2: pass right (counter-clockwise) - index decreases
        
        if self.current_age + 1 in [1, 3]:  # Age 1 or 3 (1-indexed)
            # Pass to left (next player)
            hands = [player.current_hand for player in self.players]
            for i, player in enumerate(self.players):
                player.current_hand = hands[(i - 1) % self.num_players]
        else:  # Age 2
            # Pass to right (previous player)
            hands = [player.current_hand for player in self.players]
            for i, player in enumerate(self.players):
                player.current_hand = hands[(i + 1) % self.num_players]
    
    def _resolve_military_conflicts(self):
        """Resolve military conflicts at the end of an age."""
        age_token_values = {1: 1, 2: 3, 3: 5}
        token_value = age_token_values[self.current_age + 1]
        
        for i, player in enumerate(self.players):
            left_neighbor = self.players[(i - 1) % self.num_players]
            right_neighbor = self.players[(i + 1) % self.num_players]
            
            player_shields = player.shields
            left_shields = left_neighbor.shields
            right_shields = right_neighbor.shields
            
            # Compare with left neighbor
            if player_shields > left_shields:
                player.military_tokens.append(token_value)
            elif player_shields < left_shields:
                player.military_tokens.append(-1)
            
            # Compare with right neighbor
            if player_shields > right_shields:
                player.military_tokens.append(token_value)
            elif player_shields < right_shields:
                player.military_tokens.append(-1)
    
    def _get_neighbors(self, player_id: int) -> List[int]:
        """Get the IDs of a player's neighbors."""
        left = (player_id - 1) % self.num_players
        right = (player_id + 1) % self.num_players
        return [left, right]
    
    def exchange_coins_with_bank(self, player_id: int, coins: int) -> bool:
        """
        Exchange coins with the bank.
        Per rules: "players may 'make change' between the value '3' and '1' coins as needed"
        The bank has unlimited coins in both 1-coin and 3-coin denominations.
        
        Args:
            player_id: The player exchanging coins
            coins: Number of coins to add/subtract
        
        Returns:
            True if exchange was successful
        """
        player = self.players[player_id]
        new_total = player.coins + coins
        
        # Coins can never go negative
        if new_total < 0:
            return False
        
        player.coins = new_total
        return True
    
    def get_observation(self) -> Dict:
        """
        Get current game observation for RL training.
        
        Returns:
            Dictionary containing game state suitable for RL agent
        """
        observation = {
            "current_age": self.current_age,
            "current_turn": self.current_turn,
            "num_players": self.num_players,
            "discard_pile": [c.name for c in self.discard_pile],
            "players": []
        }
        
        for player in self.players:
            player_obs = {
                "player_id": player.player_id,
                "coins": player.coins,
                "wonder_name": player.wonder_name,
                "wonder_side": player.wonder_side,
                "production": dict(player.production),
                "science": dict(player.science),
                "shields": player.shields,
                "wonder_stage_progress": player.current_wonder_stage,
                "max_wonder_stages": len(player.wonder_stages),
                "cards_played": len(player.built_cards),
                "current_hand_size": len(player.current_hand),
                "current_hand": [c.name for c in player.current_hand],
                "military_tokens_score": sum(player.military_tokens),
                "built_card_names": list(player.built_card_names),
                "memory_known_cards": list(player.memory_known_cards)
            }
            observation["players"].append(player_obs)
        
        return observation
    
    def get_legal_actions(self, player_id: int) -> List[str]:
        """
        Get list of legal actions for a player.
        
        Returns:
            List of action strings (card names, "discard", etc.)
        """
        player = self.players[player_id]
        legal_actions = []
        
        for card in player.current_hand:
            # Can always choose to discard
            legal_actions.append(f"discard_{card.name}")
            
            # Can build if conditions are met
            if self._can_build_structure(player, card):
                legal_actions.append(card.name)
            
            # Can build wonder stage if conditions are met
            if self._can_build_wonder_stage(player, card):
                legal_actions.append(f"wonder_stage_{card.name}")
        
        # Always have discard as fallback
        if not legal_actions:
            legal_actions.append("discard")
        
        return list(set(legal_actions))  # Remove duplicates
    
    def is_done(self) -> bool:
        """Check if the game is over."""
        return self.state == GameState.GAME_OVER
    
    def render(self):
        """Render the current game state (for debugging)."""
        print(f"\n=== Age {self.current_age + 1}, Turn {self.current_turn + 1} ===")
        for player in self.players:
            print(f"\nPlayer {player.player_id} ({player.wonder_name}):")
            print(f"  Coins: {player.coins}")
            print(f"  Production: {player.production}")
            print(f"  Science: {player.science}")
            print(f"  Shields: {player.shields}")
            print(f"  Wonder: {player.current_wonder_stage}/{len(player.wonder_stages)} stages")
            print(f"  Cards built: {len(player.built_cards)}")
