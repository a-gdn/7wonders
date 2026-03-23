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
    """
    Filter, create Card objects, and shuffle age decks.
    
    For Cities expansion: separates Black cards, selects N per age,
    selects Purple cards N+2, and includes them in the respective age decks.
    
    Deck composition per age:
    - Base game: 7 * num_players cards per age
    - Cities expansion: 7 * num_players cards + N black cards where N = num_players
    """
    # Separate cards by type
    candidates = {1: [], 2: [], 3: []}
    black_cards = {1: [], 2: [], 3: []}
    guild_candidates = []
    
    # Determine if Cities expansion is being used by checking for black cards
    has_cities = any(card.get("color") == "black" for card in cards_data)

    for data in cards_data:
        if data["color"] == "purple":
            guild_candidates.append(data)
        elif data["color"] == "black":
            # Cities expansion: separate black cards by age
            age = data.get("age", 1)
            black_cards[age].append(data)
        elif data.get("player_requirement", 3) <= num_players:
            age = data.get("age", 1)
            candidates[age].append(data)

    decks = {1: [], 2: [], 3: []}
    
    # Base deck size is always 7 cards per player
    base_target_size = 7 * num_players

    # Age 1 and 2: Select base cards + optionally add black cards
    for age in [1, 2]:
        available = candidates[age]
        if len(available) < base_target_size:
            raise ValueError(f"Not enough base cards for Age {age}. Required: {base_target_size}, Found: {len(available)}")
        
        # Randomly sample base cards (exactly 7 per player)
        selected = random.sample(available, base_target_size) if len(available) > base_target_size else available
        decks[age] = [create_card_from_data(d) for d in selected]
        
        # Cities expansion: add black cards for this age (N black cards where N = num_players)
        if has_cities and black_cards[age]:
            num_black = min(num_players, len(black_cards[age]))
            selected_black = random.sample(black_cards[age], num_black)
            decks[age].extend([create_card_from_data(d) for d in selected_black])

    # Age 3: Guilds + Regular cards + Black cards
    # 1. Select Guilds (N + 2)
    num_guilds = num_players + 2
    if len(guild_candidates) < num_guilds:
        raise ValueError(f"Not enough Guild cards. Required: {num_guilds}, Found: {len(guild_candidates)}")
    selected_guilds = random.sample(guild_candidates, num_guilds)
    
    # 2. Select base Age 3 cards (to make up base_target_size cards)
    remaining_slots = base_target_size - len(selected_guilds)
    available_age3 = candidates[3]
    
    if len(available_age3) < remaining_slots:
        raise ValueError(f"Not enough regular cards for Age 3. Required: {remaining_slots}, Found: {len(available_age3)}")
    selected_age3 = random.sample(available_age3, remaining_slots) if len(available_age3) > remaining_slots else available_age3
    
    decks[3] = [create_card_from_data(d) for d in selected_guilds + selected_age3]
    
    # 3. Cities expansion: add black cards to Age 3
    if has_cities and black_cards[3]:
        num_black_age3 = min(num_players, len(black_cards[3]))
        selected_black_age3 = random.sample(black_cards[3], num_black_age3)
        decks[3].extend([create_card_from_data(d) for d in selected_black_age3])

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
        player_requirement=card_data.get("player_requirement", 3)
    )

def deal_age_hand(players: list, decks: dict, current_age: int, expansions: List[str] = None):
    """
    Deals cards to each player for the current age.
    
    Args:
        players: List of PlayerCity objects
        decks: Dictionary of age decks {1: [...], 2: [...], 3: [...]}
        current_age: The current age (0, 1, or 2)
        expansions: List of enabled expansions (cities, edifice, etc)
    """
    if expansions is None:
        expansions = []
    
    # age_deck uses 1-based indexing from setup_decks (1, 2, 3)
    age_deck = decks[current_age + 1]
    
    # Cities expansion: 8 cards per age; Base game: 7 cards per age
    hand_size = 8 if "cities" in expansions else 7
    
    for player in players:
        # Deal cards from the end of the shuffled list
        num_cards = min(hand_size, len(age_deck))
        player.current_hand = [age_deck.pop() for _ in range(num_cards)]