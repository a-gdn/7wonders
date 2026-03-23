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
        
        # 8. Black cards (Cities expansion)
        score += _calculate_black_card_score(player)
        
        # 9. End-of-game card effects
        score += _calculate_end_game_effects(env, player)
        
        # 10. Debt tokens (Cities and Edifice expansions)
        # Cities debt tokens (-1 each)
        if isinstance(player.cities_debt_tokens, list):
            # Flatten if it contains nested lists (defensive)
            cities_debt = []
            for item in player.cities_debt_tokens:
                if isinstance(item, list):
                    cities_debt.extend(item)
                else:
                    cities_debt.append(item)
            score += sum(cities_debt) if cities_debt else 0
        else:
            score += 0
        
        # Edifice debt tokens (-2, -3, or -5)
        score += sum(player.edifice_debt_tokens)
        
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
    Calculate VP from variable scoring effects (vp_per_card, vp_per_wonder_stage, vp_per_shield).
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
    
    # 3. VP per Shield
    vp_shield_data = None
    if "vp_per_shield" in effect:
        vp_shield_data = effect["vp_per_shield"]
    elif effect.get("type") == "vp_per_shield":
        vp_shield_data = effect
    
    if vp_shield_data:
        multiplier = vp_shield_data.get("multiplier", 1) if isinstance(vp_shield_data, dict) else vp_shield_data
        target_scope = vp_shield_data.get("target", "own") if isinstance(vp_shield_data, dict) else "own"
        
        players_to_check = []
        if "self" in target_scope or target_scope == "own" or target_scope == "neighbors_and_self":
            players_to_check.append(player)
        if "neighbors" in target_scope:
            players_to_check.extend([env.players[pid] for pid in env._get_neighbors(player.player_id)])
            
        count = sum(p.shields for p in players_to_check)
        score += count * multiplier
        
    return score

def _calculate_science_score(player) -> int:
    """Calculates science VP: (sets of 3 = 7pts) + (identical symbols^2)."""
    counts = dict(player.science)  # Make a copy to avoid modifying original
    
    # Apply science wildcard if available
    if player.science_has_wildcard:
        # Apply wildcard to the symbol with minimum count (for optimal set completion)
        min_symbol = min(counts, key=lambda k: counts[k])
        counts[min_symbol] += 1
    
    score = sum(c * c for c in counts.values())  # Identical symbols
    
    min_count = min(counts.values())  # Complete sets
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

def _calculate_black_card_score(player) -> int:
    """Black card (Cities expansion) scoring logic."""
    score = 0
    
    for card in player.built_cards:
        if card.color == CardColor.BLACK:
            effects = card.effect if isinstance(card.effect, list) else [card.effect]
            for effect in effects:
                # Static VP
                if "vp" in effect:
                    score += effect["vp"]
                # Note: Variable VP effects like vp_per_black_card are calculated
                # during the game in _apply_card_effects and stored in total_vp_from_cards
    
    return score

def _calculate_end_game_effects(env, player) -> int:
    """
    Calculate scoring from end-of-game card effects.
    This includes effects that depend on game state at the end of the game.
    """
    score = 0
    
    # Coins per military token
    if player.coins_per_military_token > 0:
        # Count VP from positive military token values
        positive_tokens = sum(t for t in player.military_tokens if t > 0)
        player.coins += positive_tokens * player.coins_per_military_token
    
    # Coins per military loss token
    if player.coins_per_military_loss_token > 0:
        # Count -1 tokens
        loss_tokens = sum(1 for t in player.military_tokens if t == -1)
        player.coins += loss_tokens * player.coins_per_military_loss_token
    
    # Coins per black card
    if player.coins_per_black_card > 0:
        black_card_count = sum(1 for card in player.built_cards if card.color == CardColor.BLACK)
        player.coins += black_card_count * player.coins_per_black_card
    
    # Others lose coins per this player's military tokens
    if player.others_lose_coins_per_military_token > 0:
        military_total = sum(player.military_tokens)
        if military_total > 0:
            for other_player in env.players:
                if other_player.player_id != player.player_id:
                    coin_loss = military_total * player.others_lose_coins_per_military_token
                    if other_player.coins >= coin_loss:
                        other_player.coins -= coin_loss
                    else:
                        other_player.coins = 0
                        unpaid = coin_loss - other_player.coins
                        for _ in range(unpaid):
                            other_player.cities_debt_tokens.append(-1)
    
    # Others lose coins per this player's wonder stages
    if player.others_lose_coins_per_wonder_stage > 0:
        wonder_stages_built = player.current_wonder_stage
        coin_loss_per_other = wonder_stages_built * player.others_lose_coins_per_wonder_stage
        if coin_loss_per_other > 0:
            for other_player in env.players:
                if other_player.player_id != player.player_id:
                    if other_player.coins >= coin_loss_per_other:
                        other_player.coins -= coin_loss_per_other
                    else:
                        other_player.coins = 0
                        unpaid = coin_loss_per_other - other_player.coins
                        for _ in range(unpaid):
                            other_player.cities_debt_tokens.append(-1)
    
    # Science: Copy neighbor's science symbols
    if player.science_copy_neighbor_count > 0:
        neighbors = [env.players[(player.player_id - 1) % len(env.players)],
                    env.players[(player.player_id + 1) % len(env.players)]]
        for neighbor in neighbors:
            for symbol in ["compass", "gear", "tablet"]:
                if neighbor.science.get(symbol, 0) > 0 and player.science_copy_neighbor_count > 0:
                    player.science[symbol] += 1
                    player.science_copy_neighbor_count -= 1
    
    # Shield-based VP (vp_per_shield) - rare but check if it exists
    # This should be handled by cards with vp_per_shield in their effects
    
    return score