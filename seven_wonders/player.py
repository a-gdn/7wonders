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
    
    military_shields: int = 0
    military_tokens: List[int] = field(default_factory=list)
    current_hand: List[Card] = field(default_factory=list)
    total_vp_from_cards: int = 0
    memory_known_cards: Set[str] = field(default_factory=set)

    def exchange_coins(self, coins_to_exchange: int) -> bool:
        # Internal utility logic
        return True