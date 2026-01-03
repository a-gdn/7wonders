from typing import Dict, List
from .constants import CardColor

def calculate_scores(env) -> Dict[int, int]:
    """
    Calculate final scores for all players based on official rules.
    
    Returns:
        Dictionary mapping player_id to total VP
    """
    scores = {}
    
    for player in env.players:
        score = 0
        
        # 1. Military conflicts (tokens)
        score += sum(player.military_tokens)
        
        # 2. Treasury (3 coins = 1 VP)
        score += player.coins // 3
        
        # 3. Wonder VP
        for stage in player.wonder_stages:
            if stage.built:
                effects = stage.effect if isinstance(stage.effect, list) else [stage.effect]
                for effect in effects:
                    if "vp" in effect:
                        score += effect["vp"]
        
        # 4. Civilian structures (blue)
        for card in player.built_cards:
            if card.color == CardColor.BLUE:
                effects = card.effect if isinstance(card.effect, list) else [card.effect]
                for effect in effects:
                    score += effect.get("vp", 0)
        
        # 5. Scientific structures (green)
        score += _calculate_science_score(player)
        
        # 6. Commercial structures (yellow)
        score += _calculate_commercial_score(env, player)
        
        # 7. Guilds (purple)
        score += _calculate_guild_score(env, player)
        
        scores[player.player_id] = score
    
    return scores

def get_winner(env) -> int:
    """
    Determine the winner. Tiebreaker: most coins in treasury.
    """
    scores = calculate_scores(env)
    max_score = max(scores.values())
    tied_players = [pid for pid, score in scores.items() if score == max_score]
    
    if len(tied_players) == 1:
        return tied_players[0]
    
    # Tiebreaker logic
    tied_player_coins = {pid: env.players[pid].coins for pid in tied_players}
    max_coins = max(tied_player_coins.values())
    coin_winners = [pid for pid, coins in tied_player_coins.items() if coins == max_coins]
    
    return min(coin_winners)

def calculate_variable_vp(env, player, effect) -> int:
    """
    Calculate VP from variable scoring effects (vp_per_card, vp_per_wonder_stage).
    Used for Commercial (Yellow) and Guild (Purple) cards.
    """
    score = 0
    
    # 1. VP per Card
    vp_data = None
    if "vp_per_card" in effect:
        vp_data = effect["vp_per_card"]
    elif effect.get("type") == "vp_per_card":
        vp_data = effect
        
    if vp_data:
        target_colors = vp_data.get("colors", [])
        if "color" in vp_data:
            target_colors.append(vp_data["color"])
        
        target_scope = vp_data.get("target", "own")
        multiplier = vp_data.get("multiplier", 1)
        
        players_to_check = []
        if "self" in target_scope or target_scope == "own" or target_scope == "neighbors_and_self":
            players_to_check.append(player)
        if "neighbors" in target_scope:
            players_to_check.extend([env.players[pid] for pid in env._get_neighbors(player.player_id)])
            
        count = 0
        for p in players_to_check:
            for c in p.built_cards:
                if c.color.value in target_colors:
                    count += 1
        score += count * multiplier

    # 2. VP per Wonder Stage
    vp_wonder_data = None
    if "vp_per_wonder_stage" in effect:
        vp_wonder_data = effect["vp_per_wonder_stage"]
    elif effect.get("type") == "vp_per_wonder_stage":
        vp_wonder_data = effect
        
    if vp_wonder_data:
        multiplier = vp_wonder_data.get("multiplier", 1)
        target_scope = vp_wonder_data.get("target", "own")
        
        players_to_check = []
        if "self" in target_scope or target_scope == "own" or target_scope == "neighbors_and_self":
            players_to_check.append(player)
        if "neighbors" in target_scope:
            players_to_check.extend([env.players[pid] for pid in env._get_neighbors(player.player_id)])
            
        count = sum(p.current_wonder_stage for p in players_to_check)
        score += count * multiplier
        
    return score

def _calculate_science_score(player) -> int:
    """Calculates science VP: (sets of 3 = 7pts) + (identical symbols^2)."""
    counts = player.science
    score = sum(c * c for c in counts.values()) # Identical symbols
    
    min_count = min(counts.values()) # Complete sets
    if min_count >= 1:
        score += min_count * 7
        
    return score

def _calculate_commercial_score(env, player) -> int:
    """Yellow card scoring logic."""
    score = 0
    for card in player.built_cards:
        if card.color == CardColor.YELLOW:
            effects = card.effect if isinstance(card.effect, list) else [card.effect]
            for effect in effects:
                # Static VP
                if "vp" in effect:
                    score += effect["vp"]
                # Variable VP (e.g. Chamber of Commerce, Arena)
                score += calculate_variable_vp(env, player, effect)
    return score

def _calculate_guild_score(env, player) -> int:
    """Purple card scoring logic based on neighbors."""
    score = 0
    
    for card in player.built_cards:
        if card.color == CardColor.PURPLE:
            effects = card.effect if isinstance(card.effect, list) else [card.effect]
            for effect in effects:
                # Static VP
                if "vp" in effect:
                    score += effect["vp"]
                # Variable VP (e.g. Spies Guild, Builders Guild)
                score += calculate_variable_vp(env, player, effect)
    return score