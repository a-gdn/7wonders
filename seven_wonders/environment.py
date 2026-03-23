"""
7 Wonders Game Environment for Reinforcement Learning

This module implements a complete game environment for the 7 Wonders board game,
following all official game rules. It is designed to be compatible with RL training.
"""

import json
import random
import os
import copy
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
    
    def __init__(self, num_players: int = 4, random_seed: Optional[int] = None, expansions: Optional[List[str]] = None):
        """
        Initialize the game environment.
        
        Args:
            num_players: Number of players (3-7)
            random_seed: Optional seed for reproducibility
            expansions: List of expansions to enable, defaults to ["cities", "edifice"]
        """
        if not 3 <= num_players <= 7:
            raise ValueError(f"Number of players must be between 3 and 7, got {num_players}")
        
        self.num_players = num_players
        if random_seed is not None:
            random.seed(random_seed)
            
        if expansions is None:
            expansions = ["cities", "edifice"]
        self.expansions = expansions
        
        # Load game data
        self.cards_data = setup.load_json("db/cards.json")
        self.wonder_data = setup.load_json("db/wonder_boards.json")
        
        if "cities" in self.expansions:
            try:
                self.cards_data.extend(setup.load_json("db/cards_cities.json"))
                self.wonder_data.extend(setup.load_json("db/wonder_boards_cities.json"))
            except FileNotFoundError:
                pass
                
        if "edifice" in self.expansions:
            try:
                self.wonder_data.extend(setup.load_json("db/wonder_boards_edifice.json"))
                self.edifice_data = setup.load_json("db/projects_edifice.json")
            except FileNotFoundError:
                self.edifice_data = []
        else:
            self.edifice_data = []

        # Game state
        self.state = GameState.SETUP
        self.current_age = 0
        self.current_turn = 0  # 0-5 per age
        self.players: List[PlayerCity] = []
        self.decks: Dict[int, List[Card]] = {}
        self.discard_pile: List[Card] = []
        
        # Edifice expansion state (only if enabled)
        self.active_edifices: Dict[int, Dict] = {}  # Maps age (1,2,3) to edifice card data
        self.edifice_pawns_on_card: Dict[int, int] = {}  # Maps age to # pawns on the edifice
        self.edifice_pawn_box: Dict[int, int] = {}  # Maps age to # pawns in box (unused)
        self.remaining_edifice_cards: Dict[int, List[Dict]] = {}  # Maps age to unused edifice cards
        self.edifice_completed: Dict[int, bool] = {}  # Tracks which edifices were successfully completed
        
        # Cities expansion state (only if enabled)
        self.pending_coin_losses: Dict[int, int] = {}  # Maps player_id to coins lost this turn
        
        # Coin bank (unlimited coins available per rules)
        # Coins can be exchanged as 1-coin or 3-coin denominations
        self.coin_bank_unlimited = True
        
        # Track cards selected this turn (by all players simultaneously)
        self.selected_cards: Dict[int, Optional[Card]] = {i: None for i in range(num_players)}
        self.selected_action_types: Dict[int, str] = {i: "discard" for i in range(num_players)}
    
    def reset(self) -> Dict[int, Dict]:
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

        # Setup Edifice expansion if enabled
        if "edifice" in self.expansions:
            self._setup_edifices()

        # Setup Cities expansion if enabled
        if "cities" in self.expansions:
            self._setup_cities()

        # Deal initial hands for Age I (pass expansions for card count)
        setup.deal_age_hand(self.players, self.decks, 0, self.expansions)
        
        return self.get_observation()
    
    def step(self, actions: Dict[int, str]) -> Tuple[Dict[int, Dict], Dict[int, float], bool, Dict]:
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
        
        # Cities expansion: 7 turns per age (0-6); Base game: 6 turns (0-5)
        is_final_turn = self.current_turn == 6 if "cities" in self.expansions else self.current_turn == 5
        
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
        # Cities expansion: 7 turns per age (age ends when turn >= 7)
        # Base game: 6 turns per age (age ends when turn >= 6)
        age_end_turn = 7 if "cities" in self.expansions else 6
        if self.current_turn >= age_end_turn:
            # End of age: edifice resolution (if Edifice expansion enabled)
            if "edifice" in self.expansions:
                self._resolve_edifice_completion()
            
            # Military resolution
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
                    # Reset participation flag for new age
                    if "edifice" in self.expansions:
                        p.participated_in_edifice[self.current_age + 1] = False
                setup.deal_age_hand(self.players, self.decks, self.current_age, self.expansions)
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
            actions: Dictionary of player_id -> action (card name or "discard" or "wonder_stage_X")
                    Can also include "wonder_stage_X_edifice" for edifice participation
            is_final_turn: True if this is turn 6 (final turn of age)
        """
        for player_id, action in actions.items():
            player = self.players[player_id]
            hand = player.current_hand
            
            # Parse action intent and target card
            intent = "build_structure"
            target_name = action
            edifice_participate = False
            
            if action == "discard":
                intent = "discard"
                target_name = None # Will pick first available
            elif action.startswith("discard_"):
                intent = "discard"
                target_name = action.replace("discard_", "")
            elif action.startswith("wonder_stage_"):
                intent = "build_wonder"
                target_name = action.replace("wonder_stage_", "")
                # Check for edifice participation variant
                if target_name.endswith("_edifice"):
                    edifice_participate = True
                    target_name = target_name.replace("_edifice", "")
            
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
                # Store edifice participation intent
                if edifice_participate:
                    self.selected_action_types[player_id] = "build_wonder_edifice"
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
                actions_to_execute.append((player, "build_structure", card, False))
            elif intent == "build_wonder" and self._can_build_wonder_stage(player, card):
                actions_to_execute.append((player, "build_wonder", card, False))
            elif intent == "build_wonder_edifice" and self._can_build_wonder_stage(player, card):
                # Include edifice participation flag
                actions_to_execute.append((player, "build_wonder", card, True))
            else:
                # Intent was discard OR build failed
                actions_to_execute.append((player, "discard", card, False))
        
        # Pass 2: Execute actions (using deferred credits for commerce)
        for player, action_type, card, edifice_param in actions_to_execute:
            if action_type == "build_structure":
                self._build_structure(player, card, deferred_credits)
            elif action_type == "build_wonder":
                self._build_wonder_stage(player, card, deferred_credits, participate_in_edifice=edifice_param)
            else:
                self._discard_card(player, card)
        
        # Pass 2.5: Process Cities expansion coin loss effects
        if "cities" in self.expansions:
            self._process_coin_losses()
        
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
            
            if "production_choice_daily" in effect:
                # Daily production choice - pick first option
                choice_effect = effect["production_choice_daily"]
                options = choice_effect.get("options", [])
                if options:
                    player.production_choice_daily = options
                    chosen_resource = options[0]
                    player.production[chosen_resource] += choice_effect.get("choice_count", 1)
            
            if "production_choice_missing_resource" in effect:
                # Can produce one resource that player doesn't have
                player.production_choice_missing_resource = True
                # Pick first missing resource
                all_resources = ["wood", "stone", "clay", "ore", "glass", "papyrus", "textile"]
                for resource in all_resources:
                    if player.production.get(resource, 0) == 0:
                        player.production[resource] += 1
                        break
            
            if "vp" in effect:
                player.total_vp_from_cards += effect["vp"]
            
            if "shield" in effect:
                player.shields += effect["shield"]
            
            if "science" in effect:
                symbol = effect["science"]
                player.science[symbol] += 1
            
            if "science_wildcard" in effect:
                # Player can choose any science symbol
                player.science_has_wildcard = True
            
            if "coins" in effect:
                player.coins += effect["coins"]
            
            if "immediate_coins" in effect:
                coins_earned = self._calculate_immediate_coins(player, effect["immediate_coins"])
                player.coins += coins_earned
            
            if "trading" in effect:
                # Yellow card trading benefits - modifies trade costs
                # Stored for later use
                pass
            
            if "trading_discount" in effect:
                # Reduce trading costs with neighbors
                discount_data = effect["trading_discount"]
                target = discount_data.get("target", "both")
                discount_amount = discount_data.get("discount", 1)
                
                if target == "both":
                    player.trading_discounts["left"] += discount_amount
                    player.trading_discounts["right"] += discount_amount
                elif target == "left":
                    player.trading_discounts["left"] += discount_amount
                elif target == "right":
                    player.trading_discounts["right"] += discount_amount
            
            # Cities expansion: Diplomacy token
            if "diplomacy" in effect:
                player.diplomacy_tokens += effect["diplomacy"]
            
            # Cities expansion: Neighbors get debt tokens
            if "neighbors_debt_tokens" in effect:
                debt_value = effect["neighbors_debt_tokens"]
                player.neighbors_debt_tokens_value = debt_value
            
            # End-of-game: Copy neighbor's science symbols
            if "end_game_science_copy_neighbor" in effect:
                player.science_copy_neighbor_count = effect["end_game_science_copy_neighbor"]
            
            # End-of-game: Coins per black card
            if "coins_per_black_card" in effect:
                player.coins_per_black_card = effect["coins_per_black_card"]
            
            # End-of-game: Coins per military token
            if "coins_per_military_token" in effect:
                player.coins_per_military_token = effect["coins_per_military_token"]
            
            # End-of-game: Coins per military loss token
            if "coins_per_military_loss_token" in effect:
                player.coins_per_military_loss_token = effect["coins_per_military_loss_token"]
            
            # End-of-game: Others lose coins per this player's military tokens
            if "others_lose_coins_per_military_token" in effect:
                player.others_lose_coins_per_military_token = effect["others_lose_coins_per_military_token"]
            
            # End-of-game: Others lose coins per this player's wonder stages
            if "others_lose_coins_per_wonder_stage" in effect:
                player.others_lose_coins_per_wonder_stage = effect["others_lose_coins_per_wonder_stage"]
            
            # Wonder stages: Don't cost resources
            if "free_wonder_stages_resources" in effect:
                player.free_wonder_resources = True
            
            # End-of-game: Copy purple card effects
            if "copy_purple_card_effect" in effect:
                player.copy_purple_count = effect["copy_purple_card_effect"]
            
            # Special action: Play from discard pile
            if "action" in effect:
                if effect["action"] == "play_from_discard_end_of_turn":
                    player.special_action_play_from_discard = True
        
        # Note: Variable VP effects (vp_per_card, vp_per_wonder_stage) are 
        # calculated at the end of the game in scoring.py, not here.
        # Note: Coin loss effects (others_lose_coins, neighbors_lose_coins) are
        # applied after all cards are played to avoid retroactive changes
    
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
        # If player has free_wonder_resources, skip resource cost check
        if not player.free_wonder_resources:
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
    
    def _build_wonder_stage(self, player: PlayerCity, card: Card, deferred_credits: Dict[int, int] = None, participate_in_edifice: bool = False):
        """Build a wonder stage using the selected card as marker.
        
        Args:
            player: The player building
            card: The card used as marker
            deferred_credits: Dict mapping player_id to deferred coin income
            participate_in_edifice: Whether to participate in the age's edifice
        """
        stage = player.wonder_stages[player.current_wonder_stage]
        
        # Pay the cost (coins only - resources are NOT spent)
        cost = stage.cost
        if "coins" in cost:
            player.coins -= cost["coins"]
        
        # Pay resource costs by buying from neighbors if needed (resources are NOT spent)
        # Skip if player has free_wonder_resources effect
        resources_needed = {k: v for k, v in cost.items() if k != "coins"}
        if resources_needed and not player.free_wonder_resources:
            resource_manager.pay_resource_cost(self, player, resources_needed, deferred_credits)

        # Handle Edifice participation (per rules: can participate with wonder stage construction)
        if participate_in_edifice and self._can_participate_in_edifice(player):
            self._participate_in_edifice(player, deferred_credits)

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
    
    def _process_coin_losses(self):
        """
        Process Cities expansion coin loss effects.
        
        After all cards are played, check for card effects that cause:
        - others_lose_coins: all other players lose coins
        - neighbors_lose_coins: neighbors lose coins
        - neighbors_gain_coins: neighbors gain coins
        - others_gain_coins: all other players gain coins
        
        If a player can't pay coins, they take a -1 debt token instead (per rules).
        """
        # Collect all coin modifications needed
        coin_changes = {i: 0 for i in range(self.num_players)}  # player_id -> net coin change
        
        for player in self.players:
            for card in player.built_cards:
                effects = card.effect if isinstance(card.effect, list) else [card.effect]
                
                for effect in effects:
                    # Others lose coins
                    if "others_lose_coins" in effect:
                        amount = effect["others_lose_coins"]
                        for other_id in range(self.num_players):
                            if other_id != player.player_id:
                                coin_changes[other_id] -= amount
                    
                    # Neighbors lose coins
                    if "neighbors_lose_coins" in effect:
                        amount = effect["neighbors_lose_coins"]
                        for neighbor_id in self._get_neighbors(player.player_id):
                            coin_changes[neighbor_id] -= amount
                    
                    # Others gain coins
                    if "others_gain_coins" in effect:
                        amount = effect["others_gain_coins"]
                        for other_id in range(self.num_players):
                            if other_id != player.player_id:
                                coin_changes[other_id] += amount
                    
                    # Neighbors gain coins
                    if "neighbors_gain_coins" in effect:
                        amount = effect["neighbors_gain_coins"]
                        for neighbor_id in self._get_neighbors(player.player_id):
                            coin_changes[neighbor_id] += amount
                    
                    # Neighbors get debt tokens (Cities expansion)
                    if "neighbors_debt_tokens" in effect:
                        debt_value = effect["neighbors_debt_tokens"]
                        for neighbor_id in self._get_neighbors(player.player_id):
                            neighbor = self.players[neighbor_id]
                            neighbor.cities_debt_tokens.append(debt_value)
        
        # Apply coin changes and handle debts
        for player_id, coin_change in coin_changes.items():
            if coin_change == 0:
                continue
            
            player = self.players[player_id]
            
            if coin_change > 0:
                # Gaining coins - straightforward
                player.coins += coin_change
            else:
                # Losing coins
                coins_to_lose = abs(coin_change)
                if player.coins >= coins_to_lose:
                    # Can pay
                    player.coins -= coins_to_lose
                else:
                    # Can't pay full amount - take debt tokens for unpaid coins
                    player.coins = 0
                    unpaid = coins_to_lose - player.coins
                    for _ in range(unpaid):
                        player.cities_debt_tokens.append(-1)
    
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
        """
        Resolve military conflicts at the end of an age.
        
        Cities expansion: Players with diplomacy tokens don't participate in conflicts.
        Their neighbors are considered adjacent to each other for conflict purposes.
        """
        age_token_values = {1: 1, 2: 3, 3: 5}
        token_value = age_token_values[self.current_age + 1]
        
        for i, player in enumerate(self.players):
            left_neighbor = self.players[(i - 1) % self.num_players]
            right_neighbor = self.players[(i + 1) % self.num_players]
            
            player_shields = player.shields
            left_shields = left_neighbor.shields
            right_shields = right_neighbor.shields
            
            # Cities expansion: Check for diplomacy tokens
            has_diplomacy = "cities" in self.expansions and player.diplomacy_tokens > 0
            
            if has_diplomacy:
                # Player with diplomacy token doesn't participate
                # Discard one diplomacy token (per rules, must use it this round)
                player.diplomacy_tokens -= 1
                
                # Neighbors are now considered adjacent to each other
                # They will compare in separate conflicts
                continue
            
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
    
    def _setup_edifices(self):
        """Setup Edifice expansion: initialize edifices, participating pawns, and debt tokens."""
        edifice_participation_counts = {3: 2, 4: 3, 5: 3, 6: 4, 7: 5}
        pawns_per_age = edifice_participation_counts.get(self.num_players, 3)
        
        # Separate edifice cards by age and shuffle
        edifice_by_age = {1: [], 2: [], 3: []}
        for edifice in self.edifice_data:
            age = edifice.get("age", 1)
            edifice_by_age[age].append(edifice)
        
        # Shuffle each age's edifices and select 1 per age
        for age in [1, 2, 3]:
            if edifice_by_age[age]:
                random.shuffle(edifice_by_age[age])
                self.active_edifices[age] = edifice_by_age[age][0]  # Select first
                self.remaining_edifice_cards[age] = edifice_by_age[age][1:]  # Rest unused
                self.edifice_pawns_on_card[age] = pawns_per_age
                self.edifice_pawn_box[age] = 0
            else:
                self.active_edifices[age] = None
                self.remaining_edifice_cards[age] = []
                self.edifice_pawns_on_card[age] = 0
                self.edifice_pawn_box[age] = pawns_per_age
        
        # Initialize edifice completion tracking
        self.edifice_completed = {1: False, 2: False, 3: False}
        
        # Initialize player edifice state
        for player in self.players:
            player.edifice_participation_pawns = {1: 0, 2: 0, 3: 0}
            player.edifice_debt_tokens = []
            player.participated_in_edifice = {1: False, 2: False, 3: False}
    
    def _setup_cities(self):
        """Setup Cities expansion: ensure all players have cities fields properly initialized."""
        for player in self.players:
            # Ensure cities_debt_tokens is a list
            if not isinstance(player.cities_debt_tokens, list):
                player.cities_debt_tokens = []
            # Ensure diplomacy_tokens is an int
            if not isinstance(player.diplomacy_tokens, int):
                player.diplomacy_tokens = 0
    
    def _can_participate_in_edifice(self, player: PlayerCity) -> bool:
        """
        Check if a player can participate in the current age's edifice.
        Per rules: can participate once per age, only if pawns available.
        """
        if "edifice" not in self.expansions:
            return False
        
        age_idx = self.current_age + 1  # Convert 0-indexed to 1-indexed age
        
        # Check if already participated this age
        if player.participated_in_edifice.get(age_idx, False):
            return False
        
        # Check if edifice exists and has pawns
        edifice = self.active_edifices.get(age_idx)
        if not edifice:
            return False
        
        if self.edifice_pawns_on_card.get(age_idx, 0) > 0:
            return True
        
        return False
    
    def _participate_in_edifice(self, player: PlayerCity, deferred_credits: Dict[int, int] = None) -> bool:
        """
        Execute participation in current age's edifice.
        Player must pay the participation cost alongside wonder stage cost.
        
        Returns:
            True if participation was successful
        """
        if "edifice" not in self.expansions or not self._can_participate_in_edifice(player):
            return False
        
        age_idx = self.current_age + 1
        edifice = self.active_edifices.get(age_idx)
        
        if not edifice:
            return False
        
        # Get participation cost
        participation_cost = edifice.get("cost", {})
        
        # Check if player can afford it (coins only - simplified)
        coin_cost = participation_cost.get("coins", 0)
        
        # Check for resource costs - for now, simplified to just check coins
        # Full implementation would need to handle resource trading
        resource_cost = {k: v for k, v in participation_cost.items() if k != "coins"}
        total_coin_equivalent = coin_cost
        
        # Rough estimate: assume each resource costs 2 coins to buy from neighbor
        if resource_cost and not player.production:
            total_coin_equivalent += sum(resource_cost.values()) * 2
        
        if player.coins < total_coin_equivalent:
            return False
        
        # Deduct cost
        if coin_cost > 0:
            player.coins -= coin_cost
        
        # Pay resource costs if needed (simplified - just deduct coins equivalent)
        if resource_cost:
            resource_manager.pay_resource_cost(self, player, resource_cost, deferred_credits)
        
        # Take a participation pawn
        current_pawns = self.edifice_pawns_on_card.get(age_idx, 0)
        
        if current_pawns > 0:
            # Take from card
            self.edifice_pawns_on_card[age_idx] -= 1
            player.edifice_participation_pawns[age_idx] += 1
        else:
            # Take from box
            box_pawns = self.edifice_pawn_box.get(age_idx, 0)
            if box_pawns > 0:
                self.edifice_pawn_box[age_idx] -= 1
                player.edifice_participation_pawns[age_idx] += 1
            else:
                # No pawns available - shouldn't happen if can_participate is correct
                return False
        
        # Mark as participated this age
        player.participated_in_edifice[age_idx] = True
        
        # Check if edifice is now complete (all pawns taken from card)
        if self.edifice_pawns_on_card.get(age_idx, 0) == 0 and not any(
            p.edifice_participation_pawns.get(age_idx, 0) == 0 for p in self.players
        ):
            # Edifice is complete - will be resolved at end of age
            pass
        
        return True
    
    def _resolve_edifice_completion(self):
        """
        Resolve edifice completion at end of age.
        
        For each age's edifice:
        - If all pawns taken: distribute rewards to participants
        - If pawns remain on card: apply penalties to those without pawns and didn't participate
        """
        age_idx = self.current_age + 1  # Convert to 1-indexed age
        edifice = self.active_edifices.get(age_idx)
        
        if not edifice:
            return
        
        pawns_remaining = self.edifice_pawns_on_card.get(age_idx, 0)
        
        if pawns_remaining == 0:
            # Edifice completed successfully - distribute rewards
            self.edifice_completed[age_idx] = True
            self._apply_edifice_reward(edifice)
        else:
            # Edifice construction failed - apply penalties
            self.edifice_completed[age_idx] = False
            self._apply_edifice_penalty(edifice, age_idx)
    
    def _apply_edifice_reward(self, edifice: Dict):
        """Apply rewards to players who participated in the completed edifice."""
        age_idx = self.current_age + 1
        reward = edifice.get("reward", {})
        
        # Distribute reward to those with participation pawns
        for player in self.players:
            if player.edifice_participation_pawns.get(age_idx, 0) > 0:
                self._apply_edifice_effect(player, reward)
    
    def _apply_edifice_penalty(self, edifice: Dict, age_idx: int):
        """Apply penalties to players based on participation status."""
        penalty = edifice.get("penalty", {})
        
        debt_values = {1: -2, 2: -3, 3: -5}
        debt_value = debt_values.get(age_idx, -2)
        
        for player in self.players:
            # If player has NO participation pawns for this age, apply penalty
            if player.edifice_participation_pawns.get(age_idx, 0) == 0:
                # Try to apply penalty
                if not self._apply_edifice_penalty_to_player(player, penalty):
                    # If can't pay, take a debt token instead
                    player.edifice_debt_tokens.append(debt_value)
    
    def _apply_edifice_penalty_to_player(self, player: PlayerCity, penalty: Dict) -> bool:
        """
        Apply penalty to a player. Returns True if penalty was fully applied, False if debt taken.
        
        Penalties can include:
        - coins: lose coins (if negative)
        - discard_card_color: discard a card of specific color
        """
        # Coins penalty (if negative)
        if "coins" in penalty and penalty["coins"] < 0:
            coin_penalty = abs(penalty["coins"])
            if player.coins >= coin_penalty:
                player.coins -= coin_penalty
            else:
                # Can't pay full penalty - would need debt token
                return False
        
        # Card discard penalty
        if "discard_card_color" in penalty:
            color = penalty["discard_card_color"]
            count = penalty.get("count", 1)
            
            # Find cards of this color and discard them
            cards_to_discard = [
                card for card in player.built_cards 
                if card.color.value == color
            ]
            cards_to_discard = cards_to_discard[:count]
            
            for card in cards_to_discard:
                player.built_cards.remove(card)
                player.built_card_names.discard(card.name)
        
        return True
    
    def _apply_edifice_effect(self, player: PlayerCity, effect: Dict):
        """Apply an edifice reward effect to a player."""
        if "vp" in effect:
            player.total_vp_from_cards += effect["vp"]
        
        if "vp_per_color" in effect:
            # Count cards of specified colors and apply VP
            per_color = effect["vp_per_color"]
            colors = per_color.get("colors", [])
            vp = per_color.get("vp", 0)
            count = sum(1 for card in player.built_cards if card.color.value in colors)
            player.total_vp_from_cards += count * vp
        
        if "vp_per_wonder_stage" in effect:
            vp_per_stage = effect["vp_per_wonder_stage"]
            player.total_vp_from_cards += player.current_wonder_stage * vp_per_stage
        
        if "coins" in effect:
            player.coins += effect["coins"]
        
        if "shield" in effect:
            player.shields += effect["shield"]
        
        if "immediate_military_tokens" in effect:
            tokens = effect["immediate_military_tokens"]
            for token_value in tokens:
                player.military_tokens.append(token_value)
        
        if "discard_military_loss_tokens" in effect and effect["discard_military_loss_tokens"]:
            # Remove all -1 military loss tokens
            player.military_tokens = [t for t in player.military_tokens if t != -1]
        
        if "production_choice" in effect:
            # For RL, pick first choice
            prod_choice = effect["production_choice"]
            options = prod_choice.get("options", [])
            if options:
                player.production[options[0]] += prod_choice.get("choice_count", 1)
    
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
    
    def get_observation(self) -> Dict[int, Dict]:
        """
        Get current game observation for all players.
        
        Returns:
            Dictionary mapping player_id -> their specific observation dict
        """
        observation = {
            "current_age": self.current_age,
            "current_turn": self.current_turn,
            "num_players": self.num_players,
            "discard_pile": [c.name for c in self.discard_pile],
            "players": []
        }
        
        # Add Edifice expansion state if enabled
        if "edifice" in self.expansions:
            observation["edifice_state"] = {}
            for age in [1, 2, 3]:
                edifice = self.active_edifices.get(age)
                observation["edifice_state"][age] = {
                    "name": edifice.get("name") if edifice else None,
                    "pawns_on_card": self.edifice_pawns_on_card.get(age, 0),
                    "pawns_in_box": self.edifice_pawn_box.get(age, 0)
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
                "cards_played": len(player.built_cards),
                "current_hand": [c.name for c in player.current_hand],
                "military_tokens_score": sum(player.military_tokens),
                "built_card_names": list(player.built_card_names),
                "memory_known_cards": list(player.memory_known_cards)
            }
            
            # Add Edifice expansion state if enabled
            if "edifice" in self.expansions:
                player_obs["edifice_pawns"] = dict(player.edifice_participation_pawns)
                player_obs["edifice_debt_tokens"] = list(player.edifice_debt_tokens)
                player_obs["edifice_debt_value"] = sum(player.edifice_debt_tokens)
            
            # Add Cities expansion state if enabled
            if "cities" in self.expansions:
                player_obs["diplomacy_tokens"] = player.diplomacy_tokens
                # Ensure cities_debt_tokens is a proper list before converting/summing
                if isinstance(player.cities_debt_tokens, list):
                    player_obs["cities_debt_tokens"] = list(player.cities_debt_tokens)
                    # Flatten if it contains nested lists (defensive)
                    flattened = []
                    for item in player.cities_debt_tokens:
                        if isinstance(item, list):
                            flattened.extend(item)
                        else:
                            flattened.append(item)
                    player_obs["cities_debt_value"] = sum(flattened) if flattened else 0
                else:
                    player_obs["cities_debt_tokens"] = []
                    player_obs["cities_debt_value"] = 0
            
            observation["players"].append(player_obs)
        
        # Create individual views with private information masked
        player_views = {}
        for pid in range(self.num_players):
            # Deepcopy to ensure no shared mutable state in views
            view = copy.deepcopy(observation)
            
            # Mask opponents' private info
            for p_obs in view["players"]:
                if p_obs["player_id"] != pid:
                    p_obs["current_hand"] = []
                    p_obs["memory_known_cards"] = []
            player_views[pid] = view
            
        return player_views
    
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
                
                # Can also participate in edifice during wonder stage (if conditions met)
                if "edifice" in self.expansions and self._can_participate_in_edifice(player):
                    legal_actions.append(f"wonder_stage_{card.name}_edifice")
        
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
