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
    decks = {1: [], 2: [], 3: []}
    for data in cards_data:
        if data["player_requirement"] == num_players and data["color"] != "purple":
            decks[data["age"]].append(create_card_from_data(data))
    
    # Handle Guilds (Purple) for Age III
    purple_cards = [c for c in cards_data if c["color"] == "purple"]
    num_guilds = num_players + 2
    selected_guilds = random.sample(purple_cards, min(num_guilds, len(purple_cards)))
    for g_data in selected_guilds:
        decks[3].append(create_card_from_data(g_data))
        
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