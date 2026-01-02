from typing import Dict, List, Set
from .constants import CardColor

def can_afford_resources(env, player, card_cost: Dict[str, int]) -> bool:
    """
    Check if player can afford the resource cost of a card or wonder stage.
    
    Per rules: Can use own production and/or buy from neighbors.
    Must have enough coins at the START of the turn to facilitate trades.
    """
    # 1. Check direct coin cost
    if "coins" in card_cost:
        if player.coins < card_cost["coins"]:
            return False
    
    # 2. Check resource production
    own_production = player.production
    resources_needed = {k: v for k, v in card_cost.items() if k != "coins"}
    
    for resource, required in resources_needed.items():
        available_own = own_production.get(resource, 0)
        
        # If own production covers it, move to next resource
        if available_own >= required:
            continue
            
        # 3. Calculate trade requirements
        needed_to_buy = required - available_own
        cost_per_unit = 1 if has_trading_bonus(player) else 2
        total_trade_cost = needed_to_buy * cost_per_unit
        
        # Check if neighbors can provide the resource
        neighbor_can_provide = False
        for neighbor_id in env._get_neighbors(player.player_id):
            neighbor = env.players[neighbor_id]
            sellable = get_sellable_resources(env, neighbor)
            if resource in sellable:
                neighbor_can_provide = True
                break
        
        # Player must have neighbor availability AND sufficient coins
        if not neighbor_can_provide or player.coins < total_trade_cost:
            return False
            
    return True

def pay_resource_cost(env, player, resources_needed: Dict[str, int]):
    """
    Pay resource cost by using own resources and/or buying from neighbors.
    """
    resources_to_buy = {}
    
    for resource, needed in resources_needed.items():
        available_own = player.production.get(resource, 0)
        if available_own < needed:
            resources_to_buy[resource] = needed - available_own
    
    if resources_to_buy:
        buy_resources_from_neighbors(env, player, resources_to_buy)

def buy_resources_from_neighbors(env, player, resources_to_buy: Dict[str, int]):
    """
    Logic for purchasing resources from the left and right neighbors.
    """
    neighbors_ids = env._get_neighbors(player.player_id)
    
    for resource, quantity_needed in resources_to_buy.items():
        for neighbor_id in neighbors_ids:
            if quantity_needed <= 0:
                break
            
            neighbor = env.players[neighbor_id]
            sellable = get_sellable_resources(env, neighbor)
            
            if resource in sellable:
                cost_per_resource = 1 if has_trading_bonus(player) else 2
                can_buy = min(quantity_needed, player.coins // cost_per_resource)
                
                if can_buy > 0:
                    player.coins -= can_buy * cost_per_resource
                    neighbor.coins += can_buy * cost_per_resource
                    quantity_needed -= can_buy

def get_sellable_resources(env, player) -> Set[str]:
    """
    Returns resources that can be sold (Wonder, Brown, and Grey cards).
    """
    sellable = set()
    
    # Add wonder starting resource
    for wonder in env.wonder_data:
        if wonder["name"] == player.wonder_name:
            start_resource = wonder.get("start_resource")
            if start_resource:
                sellable.add(start_resource)
            break
            
    # Add resources from Raw Materials (Brown) and Manufactured Goods (Grey)
    for card in player.built_cards:
        if card.color in [CardColor.BROWN, CardColor.GREY]:
            if "production" in card.effect:
                sellable.update(card.effect["production"].keys())
            if "production_choice" in card.effect:
                sellable.update(card.effect["production_choice"]["options"])
                
    return sellable

def has_trading_bonus(player) -> bool:
    """Checks for yellow cards that reduce trade costs."""
    return any("trading" in card.effect for card in player.built_cards if card.color == CardColor.YELLOW)

def has_chain_prerequisite(player, card) -> bool:
    """Checks if a card can be built for free via a chain."""
    if not card.chain_from:
        return False
    return any(prereq in player.built_card_names for prereq in card.chain_from)