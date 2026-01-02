from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .constants import CardColor

@dataclass
class Card:
    name: str
    age: int
    color: CardColor
    cost: Dict[str, int]
    chain_from: Optional[List[str]]
    effect: Dict
    player_requirement: int

@dataclass
class WonderStage:
    stage: int
    cost: Dict[str, int]
    effect: Dict
    built: bool = False