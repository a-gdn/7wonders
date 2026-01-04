import json
import random
import os
from typing import List, Dict
from .models import Card, WonderStage
from .player import PlayerCity
from .constants import CardColor

def load_json(path: str) -> any:
    """Load JSON file from relative path."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        import os
        abs_path = os.path.join(os.path.dirname(__file__), path)
        with open(abs_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def setup_players(num_players: int, wonder_data: List[Dict]) -> List[PlayerCity]:
    """Initialize players with randomly selected wonders."""
    players = []
    available_wonders = list(wonder_data)
    for i in range(num_players):
        wonder = available_wonders.pop(random.randint(0, len(available_wonders) - 1))
        side = random.choice(["day", "night"])
        
        player = PlayerCity(player_id=i, wonder_name=wonder["name"], wonder_side=side)
        player.wonder_stages = [
            WonderStage(stage=j + 1, cost=s["cost"], effect=s["effect"])
            for j, s in enumerate(wonder["sides"][side])
        ]
        if wonder.get("start_resource"):
            player.production[wonder.get("start_resource")] = 1
        players.append(player)
    return players

def setup_decks(num_players: int, cards_data: List[Dict]) -> Dict[int, List[Card]]:
    """Filter, create Card objects, and shuffle age decks."""
    # Separate candidates by age/type
    candidates = {1: [], 2: [], 3: []}
    guild_candidates = []

    for data in cards_data:
        if data["color"] == "purple":
            guild_candidates.append(data)
        elif data["player_requirement"] <= num_players:
            candidates[data["age"]].append(data)

    decks = {1: [], 2: [], 3: []}
    target_size = 7 * num_players

    # Age 1 and 2
    for age in [1, 2]:
        available = candidates[age]
        if len(available) < target_size:
            raise ValueError(f"Not enough cards for Age {age}. Required: {target_size}, Found: {len(available)}")
        # Randomly sample if we have more cards than needed (e.g. from expansions)
        selected = random.sample(available, target_size) if len(available) > target_size else available
        decks[age] = [create_card_from_data(d) for d in selected]

    # Age 3: Guilds + Regular
    # 1. Select Guilds (N + 2)
    num_guilds = num_players + 2
    if len(guild_candidates) < num_guilds:
        raise ValueError(f"Not enough Guild cards. Required: {num_guilds}, Found: {len(guild_candidates)}")
    selected_guilds = random.sample(guild_candidates, num_guilds)
    
    # 2. Fill remainder with Age 3 cards
    remaining_slots = target_size - len(selected_guilds)
    available_age3 = candidates[3]
    if len(available_age3) < remaining_slots:
        raise ValueError(f"Not enough regular cards for Age 3. Required: {remaining_slots}, Found: {len(available_age3)}")
    selected_age3 = random.sample(available_age3, remaining_slots) if len(available_age3) > remaining_slots else available_age3
    
    decks[3] = [create_card_from_data(d) for d in selected_guilds + selected_age3]

    for age in decks:
        random.shuffle(decks[age])

    return decks

def create_card_from_data(card_data: Dict) -> Card:
    """Helper to convert dictionary to Card object."""
    return Card(
        name=card_data["name"],
        age=card_data["age"],
        color=CardColor(card_data["color"]),
        cost=card_data.get("cost", {}),
        chain_from=card_data.get("chain_from"),
        effect=card_data.get("effect", {}),
        player_requirement=card_data["player_requirement"]
    )

def deal_age_hand(players: list, decks: dict, current_age: int):
    """
    Deals 7 cards to each player for the current age.
    
    Args:
        players: List of PlayerCity objects
        decks: Dictionary of age decks {1: [...], 2: [...], 3: [...]}
        current_age: The current age (0, 1, or 2)
    """
    # age_deck uses 1-based indexing from setup_decks (1, 2, 3)
    age_deck = decks[current_age + 1]
    
    for player in players:
        # Per rules, each player starts an age with 7 cards
        # We take cards from the end of the shuffled list
        hand_size = min(7, len(age_deck))
        player.current_hand = [age_deck.pop() for _ in range(hand_size)]