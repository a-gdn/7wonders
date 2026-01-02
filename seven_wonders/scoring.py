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
                if "vp" in stage.effect:
                    score += stage.effect["vp"]
        
        # 4. Civilian structures (blue)
        score += sum(card.effect.get("vp", 0) 
                    for card in player.built_cards 
                    if card.color == CardColor.BLUE)
        
        # 5. Scientific structures (green)
        score += _calculate_science_score(player)
        
        # 6. Commercial structures (yellow)
        score += _calculate_commercial_score(player)
        
        # 7. Guilds (purple)
        score += _calculate_guild_score(env, player)
        
        # 8. Dynamic Card Scoring (e.g., Age III scoring cards)
        # score += calculate_card_scoring(player, None)
        
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

def calculate_card_scoring(player, effect) -> int:
    """
    Calculate VP from complex scoring cards (typically Age III yellow/purple).
    """
    # TO DO
    
    return 0

def _calculate_science_score(player) -> int:
    """Calculates science VP: (sets of 3 = 7pts) + (identical symbols^2)."""
    counts = player.science
    score = sum(c * c for c in counts.values()) # Identical symbols
    
    min_count = min(counts.values()) # Complete sets
    if min_count >= 1:
        score += min_count * 7
        
    return score

def _calculate_commercial_score(player) -> int:
    """Yellow card scoring logic."""
    score = 0
    for card in player.built_cards:
        if card.color == CardColor.YELLOW:
            effect = card.effect
            if "vp" in effect:
                score += effect["vp"]
            # Wonder-based scoring (e.g. Arena)
            if "vp_per_wonder_stage" in effect:
                mult = effect["vp_per_wonder_stage"].get("multiplier", 1)
                score += player.current_wonder_stage * mult
    return score

def _calculate_guild_score(env, player) -> int:
    """Purple card scoring logic based on neighbors."""
    score = 0
    neighbors = env._get_neighbors(player.player_id)
    
    for card in player.built_cards:
        if card.color == CardColor.PURPLE:
            effect = card.effect
            # Check neighbor collections for specific colors
            if "vp_per_card" in effect and effect.get("target") == "neighbors":
                color = effect["vp_per_card"].get("color")
                multiplier = effect["vp_per_card"].get("multiplier", 1)
                for n_id in neighbors:
                    count = sum(1 for c in env.players[n_id].built_cards if c.color.value == color)
                    score += count * multiplier
    return score