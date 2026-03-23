from typing import List, Set, Dict
from dataclasses import dataclass, field
from .models import Card, WonderStage

@dataclass
class PlayerCity:
    player_id: int
    wonder_name: str
    wonder_side: str
    coins: int = 3
    
    built_cards: List[Card] = field(default_factory=list)
    built_card_names: Set[str] = field(default_factory=set)
    
    wonder_stages: List[WonderStage] = field(default_factory=list)
    current_wonder_stage: int = 0
    
    production: Dict[str, int] = field(default_factory=lambda: {
        "wood": 0, "stone": 0, "ore": 0, "clay": 0,
        "glass": 0, "papyrus": 0, "textile": 0
    })
    
    science: Dict[str, int] = field(default_factory=lambda: {
        "compass": 0, "gear": 0, "tablet": 0
    })
    
    shields: int = 0
    military_tokens: List[int] = field(default_factory=list)
    current_hand: List[Card] = field(default_factory=list)
    total_vp_from_cards: int = 0
    memory_known_cards: Set[str] = field(default_factory=set)
    
    # Edifice expansion tracking
    edifice_participation_pawns: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0})
    edifice_debt_tokens: List[int] = field(default_factory=list)  # Edifice expansion debt (values: -2, -3, -5)
    participated_in_edifice: Dict[int, bool] = field(default_factory=lambda: {1: False, 2: False, 3: False})
    
    # Cities expansion tracking
    diplomacy_tokens: int = 0  # Diplomacy tokens - prevent military conflicts
    cities_debt_tokens: List[int] = field(default_factory=list)  # Cities expansion debt (values: -1 per unpaid coin)
    
    # Effect tracking for end-of-game calculations
    coins_per_black_card: int = 0  # Multiplier for coins per black card built
    coins_per_military_token: int = 0  # Coins earned per military token value
    coins_per_military_loss_token: int = 0  # Coins earned per -1 military token
    science_copy_neighbor_count: int = 0  # Number of neighbor science symbols to copy
    free_wonder_resources: bool = False  # Whether wonder stages don't cost resources
    science_has_wildcard: bool = False  # Whether science can choose any symbol
    trading_discounts: Dict[str, int] = field(default_factory=lambda: {"both": 0, "left": 0, "right": 0})  # Trading cost reductions
    production_choice_daily: List[str] = field(default_factory=list)  # Daily production choice options
    production_choice_missing_resource: bool = False  # Can produce one missing resource
    neighbors_debt_tokens_value: int = 0  # Give -1 debt tokens to neighbors
    others_lose_coins_per_military_token: int = 0  # Others lose coins per this player's military tokens
    others_lose_coins_per_wonder_stage: int = 0  # Others lose coins per this player's wonder stages
    copy_purple_count: int = 0  # Number of purple card effects to copy
    special_action_play_from_discard: bool = False  # Can play card from discard pile at end of turn

    def exchange_coins(self, coins_to_exchange: int) -> bool:
        # Internal utility logic
        return True