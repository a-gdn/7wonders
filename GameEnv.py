"""
7 Wonders Game Environment for Reinforcement Learning

This module implements a complete game environment for the 7 Wonders board game,
following all official game rules. It is designed to be compatible with RL training.
"""

import json
import random
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
from copy import deepcopy


class CardColor(Enum):
    """Card colors representing different card types."""
    BROWN = "brown"      # Raw materials
    GREY = "grey"        # Manufactured goods
    BLUE = "blue"        # Civilian structures
    YELLOW = "yellow"    # Commercial structures
    RED = "red"          # Military structures
    GREEN = "green"      # Scientific structures
    PURPLE = "purple"    # Guilds


class ScienceSymbol(Enum):
    """Science symbols for scoring."""
    COMPASS = "compass"
    GEAR = "gear"
    TABLET = "tablet"


@dataclass
class Card:
    """Represents a single card in the game."""
    name: str
    age: int
    color: CardColor
    cost: Dict[str, int]  # Resource or coin costs
    chain_from: Optional[List[str]]  # Cards that unlock free construction
    effect: Dict
    player_requirement: int  # Minimum players for this card variant


@dataclass
class WonderStage:
    """Represents a stage of a wonder."""
    stage: int
    cost: Dict[str, int]
    effect: Dict
    built: bool = False


@dataclass
class PlayerCity:
    """Represents a player's city with all their built structures and resources."""
    player_id: int
    wonder_name: str
    wonder_side: str  # "day" or "night"
    coins: int = 3  # Total coin value (can be represented as 1-coins or 3-coins)
    
    # Built structures
    built_cards: List[Card] = field(default_factory=list)
    built_card_names: Set[str] = field(default_factory=set)
    
    # Wonder progress
    wonder_stages: List[WonderStage] = field(default_factory=list)
    current_wonder_stage: int = 0
    
    # Resources produced
    production: Dict[str, int] = field(default_factory=lambda: {
        "wood": 0, "stone": 0, "ore": 0, "clay": 0,
        "glass": 0, "papyrus": 0, "textile": 0
    })
    
    # Science
    science: Dict[str, int] = field(default_factory=lambda: {
        "compass": 0, "gear": 0, "tablet": 0
    })
    
    # Military
    military_shields: int = 0
    military_tokens: List[int] = field(default_factory=list)  # -1, 1, 3, 5
    
    # Hand of cards
    current_hand: List[Card] = field(default_factory=list)
    
    # Stats for scoring
    total_vp_from_cards: int = 0
    
    def exchange_coins(self, coins_to_exchange: int) -> bool:
        """
        Exchange coins between 1-coin and 3-coin denominations.
        Per rules: "players may 'make change' between the value '3' and '1' coins as needed"
        
        This is a utility method for tracking coin exchanges internally.
        In practice, the coin value is treated as generic integers.
        
        Args:
            coins_to_exchange: Positive value to add 3-coins, negative to convert to 1-coins
        
        Returns:
            True if exchange was successful
        """
        # This is implicit in the coin system - coins are stored as total value
        # and can be freely represented as 1-coins or 3-coins
        # This method exists for clarity and potential future expansion
        return True


class GameState(Enum):
    """Game states."""
    SETUP = "setup"
    AGE_ACTIVE = "age_active"
    CARD_SELECTION = "card_selection"
    CARD_ACTION = "card_action"
    MILITARY_RESOLUTION = "military_resolution"
    GAME_OVER = "game_over"


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
        self.cards_data = self._load_json("db/cards.json")
        self.wonder_data = self._load_json("db/wonder_boards.json")
        
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
        
    def _load_json(self, path: str) -> any:
        """Load JSON file from relative path."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Try absolute path from current working directory
            import os
            abs_path = os.path.join(os.path.dirname(__file__), path)
            with open(abs_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    def reset(self) -> Dict:
        """
        Reset the game to initial state.
        
        Returns:
            Initial game observation for RL training
        """
        # Setup players with wonders
        self.players = self._setup_players()
        
        # Setup and shuffle decks
        self.decks = self._setup_decks()
        
        # Initialize game state
        self.state = GameState.AGE_ACTIVE
        self.current_age = 0
        self.current_turn = 0
        self.discard_pile = []
        self.selected_cards = {i: None for i in range(self.num_players)}
        
        # Deal initial hands for Age I
        self._deal_age_hand()
        
        return self.get_observation()
    
    def _setup_players(self) -> List[PlayerCity]:
        """Initialize all players with randomly selected wonders."""
        players = []
        available_wonders = list(self.wonder_data)
        
        for i in range(self.num_players):
            # Pick a random wonder and side
            wonder = available_wonders.pop(random.randint(0, len(available_wonders) - 1))
            side = random.choice(["day", "night"])
            
            # Create player city
            player = PlayerCity(
                player_id=i,
                wonder_name=wonder["name"],
                wonder_side=side,
                coins=3
            )
            
            # Initialize wonder stages
            player.wonder_stages = [
                WonderStage(
                    stage=j + 1,
                    cost=stage["cost"],
                    effect=stage["effect"]
                )
                for j, stage in enumerate(wonder["sides"][side])
            ]
            
            # Set starting resource production
            starting_resource = wonder.get("start_resource")
            if starting_resource:
                player.production[starting_resource] = 1
            
            players.append(player)
        
        return players
    
    def _setup_decks(self) -> Dict[int, List[Card]]:
        """
        Setup and shuffle age decks based on player count and guild selection.
        
        Per rules: Remove all cards not used based on player count.
        For Age III, keep num_players + 2 guild cards.
        
        Returns:
            Dictionary with age (1, 2, 3) as keys and card lists as values
        """
        decks = {1: [], 2: [], 3: []}
        
        # Filter cards by exact player requirement and age
        for card_data in self.cards_data:
            # Only include cards matching the exact player count requirement
            if card_data["player_requirement"] == self.num_players:
                if card_data["color"] != "purple":  # Handle purple separately
                    card = self._create_card_from_data(card_data)
                    decks[card_data["age"]].append(card)
        
        # Select guild cards for age 3: keep num_players + 2 guilds per rules
        purple_cards = [c for c in self.cards_data if c["color"] == "purple"]
        num_guilds_to_keep = self.num_players + 2
        
        # Filter guilds by player requirement first
        available_guilds = [g for g in purple_cards if g["player_requirement"] == self.num_players]
        
        # If not enough available after filtering, include more liberal criteria
        if len(available_guilds) < num_guilds_to_keep:
            available_guilds = [g for g in purple_cards if g["player_requirement"] <= self.num_players]
        
        # Randomly select the required number of guilds
        selected_guilds = random.sample(available_guilds, min(num_guilds_to_keep, len(available_guilds)))
        for guild_data in selected_guilds:
            card = self._create_card_from_data(guild_data)
            decks[3].append(card)
        
        # Shuffle all decks
        for age in decks:
            random.shuffle(decks[age])
        
        return decks
    
    def _create_card_from_data(self, card_data: Dict) -> Card:
        """Convert card data from JSON to Card object."""
        return Card(
            name=card_data["name"],
            age=card_data["age"],
            color=CardColor(card_data["color"]),
            cost=card_data.get("cost", {}),
            chain_from=card_data.get("chain_from"),
            effect=card_data.get("effect", {}),
            player_requirement=card_data["player_requirement"]
        )
    
    def _deal_age_hand(self):
        """Deal 7 cards to each player for the current age."""
        age_deck = self.decks[self.current_age + 1]
        
        for player in self.players:
            hand_size = min(7, len(age_deck))
            player.current_hand = [age_deck.pop() for _ in range(hand_size)]
    
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
        
        # Phase 1: Players select cards simultaneously
        self._select_cards(actions, is_final_turn)
        
        # Phase 2: Execute card actions simultaneously
        rewards = self._execute_card_actions()
        
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
                self._deal_age_hand()
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
            
            # Find and select the card
            selected_card = None
            for card in hand:
                if card.name == action or action == "discard":
                    selected_card = card
                    break
            
            if selected_card:
                self.selected_cards[player_id] = selected_card
                player.current_hand.remove(selected_card)
                
                # On turn 6 (final turn): discard remaining card without coins
                if is_final_turn and player.current_hand:
                    remaining_card = player.current_hand.pop(0)
                    self.discard_pile.append(remaining_card)
            else:
                # If action is invalid, discard
                if hand:
                    self.selected_cards[player_id] = hand[0]
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
        
        for player_id, card in self.selected_cards.items():
            if card is None:
                continue
            
            player = self.players[player_id]
            
            # Action 1: Try to build the structure
            if self._can_build_structure(player, card):
                self._build_structure(player, card)
            # Action 2: Try to build a wonder stage
            elif self._can_build_wonder_stage(player, card):
                self._build_wonder_stage(player, card)
            # Action 3: Forced or chosen discard
            else:
                self._discard_card(player, card)
        
        return rewards
    
    def _can_build_structure(self, player: PlayerCity, card: Card) -> bool:
        """Check if a player can build a structure (card)."""
        # Can't build duplicate structures
        if card.name in player.built_card_names:
            return False
        
        # Check if it's free via chain
        if self._has_chain_prerequisite(player, card):
            return True
        
        # Check resource cost
        return self._can_afford_resources(player, card)
    
    def _has_chain_prerequisite(self, player: PlayerCity, card: Card) -> bool:
        """Check if player has built a prerequisite card for free construction."""
        if card.chain_from is None:
            return False
        
        chain_from = card.chain_from
        if isinstance(chain_from, str):
            chain_from = [chain_from]
        
        for prereq in chain_from:
            if prereq in player.built_card_names:
                return True
        
        return False
    
    def _can_afford_resources(self, player: PlayerCity, card: Card) -> bool:
        """
        Check if player can afford the resource cost of a card.
        
        Per rules: Can use own resources AND/OR buy from neighbors.
        Must have coins at START of turn to buy.
        """
        cost = card.cost
        
        # Check coin cost (coins are actually spent, unlike resources)
        if "coins" in cost:
            if player.coins < cost["coins"]:
                return False
        
        # Check resource cost (can use own or buy from neighbors)
        own_production = player.production
        resources_needed = {k: v for k, v in cost.items() if k != "coins"}
        
        for resource, required in resources_needed.items():
            available_own = own_production.get(resource, 0)
            
            # If we produce enough, we're good
            if available_own >= required:
                continue
            
            # Need to buy the difference
            needed_to_buy = required - available_own
            cost_per_resource = 2
            if self._has_trading_bonus(player):
                cost_per_resource = 1
            
            total_cost = needed_to_buy * cost_per_resource
            
            # Check if any neighbor can sell and if player has enough coins
            neighbor_can_provide = False
            for neighbor_id in self._get_neighbors(player.player_id):
                neighbor = self.players[neighbor_id]
                sellable = self._get_sellable_resources(neighbor)
                if resource in sellable:
                    neighbor_can_provide = True
                    break
            
            # Player must have coins at START of turn for purchase
            if not neighbor_can_provide or player.coins < total_cost:
                return False
        
        return True
    
    def _build_structure(self, player: PlayerCity, card: Card):
        """Build a structure for the player."""
        # Check if this is a free construction via chain
        is_free = self._has_chain_prerequisite(player, card)
        
        if not is_free:
            # Pay resource and coin costs
            cost = card.cost
            
            # Check and pay coins first
            if "coins" in cost:
                player.coins -= cost["coins"]
            
            # For resources: resources are NOT spent, but must be paid for if buying from neighbors
            resources_needed = {k: v for k, v in cost.items() if k != "coins"}
            if resources_needed:
                self._pay_resource_cost(player, resources_needed)
        
        # Add card to built cards
        player.built_cards.append(card)
        player.built_card_names.add(card.name)
        
        # Apply card effects
        self._apply_card_effects(player, card)
    
    def _pay_resource_cost(self, player: PlayerCity, resources_needed: Dict[str, int]):
        """
        Pay resource cost by using own resources and/or buying from neighbors.
        
        Per rules:
        - Resources are NOT spent
        - Player can buy from neighbors at 2 coins per resource (1 with yellow card bonus)
        - Player must have coins at START of turn (not earned coins this turn)
        - Can buy from both neighbors in same turn
        """
        resources_to_buy = {}
        
        # Identify what needs to be bought
        for resource, needed in resources_needed.items():
            available_own = player.production.get(resource, 0)
            
            if available_own < needed:
                # Need to buy the difference
                resources_to_buy[resource] = needed - available_own
        
        # Buy from neighbors
        if resources_to_buy:
            self._buy_resources_from_neighbors(player, resources_to_buy)
    
    def _buy_resources_from_neighbors(self, player: PlayerCity, resources_to_buy: Dict[str, int]):
        """
        Buy resources from neighboring cities.
        
        Per rules:
        - 2 coins per resource (standard)
        - 1 coin per resource (with yellow card trading bonus)
        - Can buy from both neighbors in same turn
        - Players can never refuse to sell
        
        Args:
            player: The player buying resources
            resources_to_buy: Dict of {resource: quantity} to buy
        """
        neighbors = self._get_neighbors(player.player_id)
        
        for resource, quantity_needed in resources_to_buy.items():
            # Try to buy from neighbors
            for neighbor_id in neighbors:
                if quantity_needed <= 0:
                    break
                
                neighbor = self.players[neighbor_id]
                
                # Check what neighbor can sell
                can_sell = self._get_sellable_resources(neighbor)
                
                if resource in can_sell:
                    # Determine cost (2 coins normally, 1 with yellow card bonus)
                    cost_per_resource = 2
                    if self._has_trading_bonus(player):
                        cost_per_resource = 1
                    
                    # Buy as much as needed/affordable
                    can_buy = min(quantity_needed, player.coins // cost_per_resource)
                    
                    if can_buy > 0:
                        player.coins -= can_buy * cost_per_resource
                        neighbor.coins += can_buy * cost_per_resource
                        quantity_needed -= can_buy
    
    def _get_sellable_resources(self, player: PlayerCity) -> Set[str]:
        """
        Get resources that a player can sell to neighbors.
        
        Per rules: Can sell resources from:
        - Wonder board (initial production)
        - Brown cards (raw materials)
        - Grey cards (manufactured goods)
        
        Cannot sell from:
        - Some yellow cards (reserved to owner)
        - Some wonders (reserved to owner)
        
        Returns:
            Set of resource names the player can sell
        """
        sellable = set()
        
        # Add wonder starting resource
        for wonder in self.wonder_data:
            if wonder["name"] == player.wonder_name:
                start_resource = wonder.get("start_resource")
                if start_resource:
                    sellable.add(start_resource)
                break
        
        # Add resources from brown cards (raw materials)
        for card in player.built_cards:
            if card.color == CardColor.BROWN:
                if "production" in card.effect:
                    for resource in card.effect["production"].keys():
                        sellable.add(resource)
                if "production_choice" in card.effect:
                    # For choice cards, player can sell any chosen resource
                    for resource in card.effect["production_choice"]["options"]:
                        sellable.add(resource)
        
        # Add resources from grey cards (manufactured goods)
        for card in player.built_cards:
            if card.color == CardColor.GREY:
                if "production" in card.effect:
                    for resource in card.effect["production"].keys():
                        sellable.add(resource)
                if "production_choice" in card.effect:
                    # For choice cards, player can sell any chosen resource
                    for resource in card.effect["production_choice"]["options"]:
                        sellable.add(resource)
        
        return sellable
    
    def _has_trading_bonus(self, player: PlayerCity) -> bool:
        """
        Check if player has yellow cards that reduce trading cost to 1 coin.
        
        Yellow cards can have trading effects that reduce costs.
        
        Returns:
            True if player has trading bonus, False otherwise
        """
        for card in player.built_cards:
            if card.color == CardColor.YELLOW:
                if "trading" in card.effect:
                    # Trading effect exists - player gets discount
                    return True
        
        return False
    
    def _apply_card_effects(self, player: PlayerCity, card: Card):
        """Apply the effects of a built card."""
        effect = card.effect
        
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
        
        if "military" in effect:
            player.military_shields += effect["military"]
        
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
        
        if "vp_per_card" in effect:
            # Scoring cards - calculate VP
            vp = self._calculate_card_scoring(player, effect["vp_per_card"])
            player.total_vp_from_cards += vp
        
        if "vp_per_wonder_stage" in effect:
            # Arène: 1 VP per wonder stage
            vp = player.current_wonder_stage * effect["vp_per_wonder_stage"].get("multiplier", 1)
            player.total_vp_from_cards += vp
    
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
    
    def _calculate_card_scoring(self, player: PlayerCity, effect: Dict) -> int:
        """Calculate VP from scoring cards (used at end of game)."""
        # TO DO
        # This is used for cards that score during the game
        # Most scoring happens at the end
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
            if self._has_trading_bonus(player):
                cost_per_resource = 1
            
            total_cost = needed_to_buy * cost_per_resource
            
            # Check if any neighbor can sell and if player has enough coins
            neighbor_can_provide = False
            for neighbor_id in self._get_neighbors(player.player_id):
                neighbor = self.players[neighbor_id]
                sellable = self._get_sellable_resources(neighbor)
                if resource in sellable:
                    neighbor_can_provide = True
                    break
            
            if not neighbor_can_provide or player.coins < total_cost:
                return False
        
        return True
    
    def _build_wonder_stage(self, player: PlayerCity, card: Card):
        """Build a wonder stage using the selected card as marker."""
        stage = player.wonder_stages[player.current_wonder_stage]
        
        # Pay the cost (coins only - resources are NOT spent)
        cost = stage.cost
        if "coins" in cost:
            player.coins -= cost["coins"]
        
        # Pay resource costs by buying from neighbors if needed (resources are NOT spent)
        resources_needed = {k: v for k, v in cost.items() if k != "coins"}
        if resources_needed:
            self._pay_resource_cost(player, resources_needed)
        
        # Mark stage as built
        stage.built = True
        player.current_wonder_stage += 1
        
        # Apply stage effects
        self._apply_wonder_stage_effects(player, stage)
        
        # The card used as marker goes to discard (hidden)
        self.discard_pile.append(card)
    
    def _apply_wonder_stage_effects(self, player: PlayerCity, stage: WonderStage):
        """Apply effects from completing a wonder stage."""
        effect = stage.effect
        
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
            
            player_shields = player.military_shields
            left_shields = left_neighbor.military_shields
            right_shields = right_neighbor.military_shields
            
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
    
    def calculate_scores(self) -> Dict[int, int]:
        """
        Calculate final scores for all players.
        
        Per rules: "In case of a tie, the player with the most coins in his or her treasury is the winner."
        
        Scoring order:
        1. Military conflicts (tokens)
        2. Treasury (coins / 3)
        3. Wonder
        4. Civilian structures (blue)
        5. Scientific structures (green)
        6. Commercial structures (yellow)
        7. Guilds (purple)
        
        Returns:
            Dictionary mapping player_id to total VP
        """
        scores = {}
        
        for player in self.players:
            score = 0
            
            # 1. Military tokens
            score += sum(player.military_tokens)
            
            # 2. Treasury (3 coins = 1 VP)
            score += player.coins // 3
            
            # 3. Wonder VP
            for stage in player.wonder_stages:
                if stage.built:
                    if "vp" in stage.effect:
                        score += stage.effect["vp"]
            
            # 4. Civilian structures (blue)
            score += sum(card.effect.get("vp", 0) 
                        for card in player.built_cards 
                        if card.color == CardColor.BLUE)
            
            # 5. Scientific structures (green)
            science_score = self._calculate_science_score(player)
            score += science_score
            
            # 6. Commercial structures (yellow)
            commercial_score = self._calculate_commercial_score(player)
            score += commercial_score
            
            # 7. Guilds (purple)
            guild_score = self._calculate_guild_score(player)
            score += guild_score
            
            scores[player.player_id] = score
        
        return scores
    
    def get_winner(self) -> int:
        """
        Determine the winner of the game.
        
        Per rules:
        - Primary winner: Highest total VP
        - Tiebreaker 1: Most coins in treasury
        - Tiebreaker 2: No further breaking (tie stands)
        
        Returns:
            Player ID of the winner (or lowest ID if complete tie)
        """
        if self.state != GameState.GAME_OVER:
            raise ValueError("Cannot determine winner until game is over")
        
        scores = self.calculate_scores()
        
        # Find max score
        max_score = max(scores.values())
        tied_players = [pid for pid, score in scores.items() if score == max_score]
        
        # If no tie, return the winner
        if len(tied_players) == 1:
            return tied_players[0]
        
        # Apply tiebreaker: most coins
        tied_player_coins = {pid: self.players[pid].coins for pid in tied_players}
        max_coins = max(tied_player_coins.values())
        coin_winners = [pid for pid, coins in tied_player_coins.items() if coins == max_coins]
        
        # Return the first/lowest player ID in case of complete tie (per rules: no further breaking)
        return min(coin_winners)
    
    def _calculate_science_score(self, player: PlayerCity) -> int:
        """
        Calculate VP from scientific structures.
        
        Per rules:
        - Sets of identical symbols: n symbols = n² VP
        - Sets of 3 different symbols: each complete set = 7 VP
        - Both scoring methods are cumulative
        
        Special cases:
        - Wonder of Babylon: can increase max symbols beyond 4
        - Scientific Guild: can increase max symbols beyond 4
        
        Returns:
            Total VP from science cards
        """
        science_counts = player.science
        score = 0
        
        # Score from sets of identical symbols
        for symbol, count in science_counts.items():
            if count >= 1:
                score += count * count  # 1->1, 2->4, 3->9, 4->16, 5->25, 6->36, etc.
        
        # Score from sets of 3 different symbols
        # Count how many complete sets of 3 different symbols exist
        min_count = min(science_counts.values())
        if min_count >= 1:
            # Each complete set of 3 different symbols = 7 VP
            num_complete_sets = min_count
            score += num_complete_sets * 7
        
        return score
    
    def _calculate_commercial_score(self, player: PlayerCity) -> int:
        """
        Calculate VP from commercial structures (yellow cards).
        
        Per rules: "Some commercial structures from Age III grant victory points."
        
        Returns:
            Total VP from yellow cards
        """
        score = 0
        
        # Check for yellow cards with VP scoring effects
        for card in player.built_cards:
            if card.color == CardColor.YELLOW:
                effect = card.effect
                
                # VP per card of a specific color
                if "vp_per_card" in effect:
                    vp_effect = effect["vp_per_card"]
                    color = vp_effect.get("color")
                    multiplier = vp_effect.get("multiplier", 1)
                    
                    # Count cards of that color in own city
                    count = sum(1 for c in player.built_cards if c.color.value == color)
                    score += count * multiplier
                
                # VP per wonder stage (e.g., Arena)
                if "vp_per_wonder_stage" in effect:
                    multiplier = effect["vp_per_wonder_stage"].get("multiplier", 1)
                    score += player.current_wonder_stage * multiplier
                
                # Direct VP from the card
                if "vp" in effect:
                    score += effect["vp"]
        
        return score
    
    def _calculate_guild_score(self, player: PlayerCity) -> int:
        """
        Calculate VP from guilds (purple cards).
        
        Per rules: "Each Guild is worth a number of victory points depending on the 
        configuration of the player's city and/or that of the two neighboring cities"
        
        Returns:
            Total VP from guild cards
        """
        score = 0
        neighbors = self._get_neighbors(player.player_id)
        
        for card in player.built_cards:
            if card.color == CardColor.PURPLE:
                effect = card.effect
                
                # VP per card of a specific color
                if "vp_per_card" in effect:
                    vp_effect = effect["vp_per_card"]
                    color = vp_effect.get("color")
                    multiplier = vp_effect.get("multiplier", 1)
                    target = vp_effect.get("target", "neighbors")
                    
                    # Count cards in target cities
                    if target == "neighbors":
                        # Count in both neighboring cities
                        for neighbor_id in neighbors:
                            count = sum(1 for c in self.players[neighbor_id].built_cards 
                                      if c.color.value == color)
                            score += count * multiplier
                    elif target == "self":
                        # Count in own city
                        count = sum(1 for c in player.built_cards if c.color.value == color)
                        score += count * multiplier
                    elif target == "self_and_neighbors":
                        # Count in own city + neighbors
                        count = sum(1 for c in player.built_cards if c.color.value == color)
                        score += count * multiplier
                        for neighbor_id in neighbors:
                            count = sum(1 for c in self.players[neighbor_id].built_cards 
                                      if c.color.value == color)
                            score += count * multiplier
                
                # VP per wonder stage
                if "vp_per_wonder_stage" in effect:
                    multiplier = effect["vp_per_wonder_stage"].get("multiplier", 1)
                    target = effect.get("target", "self")
                    
                    if target == "self":
                        score += player.current_wonder_stage * multiplier
                    elif target == "neighbors":
                        for neighbor_id in neighbors:
                            score += self.players[neighbor_id].current_wonder_stage * multiplier
                    elif target == "all":
                        score += player.current_wonder_stage * multiplier
                        for neighbor_id in neighbors:
                            score += self.players[neighbor_id].current_wonder_stage * multiplier
                
                # Direct VP from the guild
                if "vp" in effect:
                    score += effect["vp"]
        
        return score
    
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
            "players": []
        }
        
        for player in self.players:
            player_obs = {
                "player_id": player.player_id,
                "coins": player.coins,
                "production": dict(player.production),
                "science": dict(player.science),
                "military_shields": player.military_shields,
                "wonder_stage_progress": player.current_wonder_stage,
                "max_wonder_stages": len(player.wonder_stages),
                "cards_played": len(player.built_cards),
                "current_hand_size": len(player.current_hand),
                "military_tokens_score": sum(player.military_tokens),
                "built_card_names": list(player.built_card_names)
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
            legal_actions.append("discard")
            
            # Can build if conditions are met
            if self._can_build_structure(player, card):
                legal_actions.append(card.name)
            
            # Can build wonder stage if conditions are met
            if self._can_build_wonder_stage(player, card):
                legal_actions.append(f"wonder_stage_{card.name}")
        
        # Always have discard as fallback
        if "discard" not in legal_actions:
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
            print(f"  Military: {player.military_shields} shields")
            print(f"  Wonder: {player.current_wonder_stage}/{len(player.wonder_stages)} stages")
            print(f"  Cards built: {len(player.built_cards)}")
