"""Handle edge cases: new players, transfers, missing data."""
from db.data_loader import players, squads, form_idx, pvt_idx, matches
import math

ROLE_DEFAULTS = {
    "Batter":      {"batting_avg": 28.0, "batting_sr": 145.0, "economy": 0.0,  "form_score": 3},
    "Bowler":      {"batting_avg": 8.0,  "batting_sr": 90.0,  "economy": 9.0,  "form_score": 3},
    "Allrounder":  {"batting_avg": 24.0, "batting_sr": 140.0, "economy": 8.5,  "form_score": 3},
    "Wicketkeeper":{"batting_avg": 26.0, "batting_sr": 142.0, "economy": 0.0,  "form_score": 3},
}

def get_player_role(player_id):
    row = players[players["player_id"]==player_id]
    if row.empty:
        return "Batter"
    return str(row.iloc[0].get("role", "Batter"))

def is_new_player(player_id):
    """Player with no historical data."""
    try:
        form_idx.loc[player_id]
        return False
    except KeyError:
        return True

def was_transferred(player_id, current_team_id):
    """Check if player recently transferred teams."""
    history = squads[squads["player_id"]==player_id].sort_values("season", ascending=False)
    if len(history) < 2:
        return False
    last_two = history.head(2)["team_id"].tolist()
    return last_two[0] != last_two[1]

def get_fallback_stats(player_id, stat_type="batting"):
    """Return role-based defaults for players with missing data."""
    role = get_player_role(player_id)
    defaults = ROLE_DEFAULTS.get(role, ROLE_DEFAULTS["Batter"])
    if stat_type == "batting":
        return defaults["batting_avg"], defaults["batting_sr"]
    elif stat_type == "bowling":
        return defaults["economy"], 0
    return defaults["form_score"]

def validate_playing11(player_ids):
    """Validate playing 11 selection."""
    errors = []
    if len(player_ids) != 11:
        errors.append(f"Must select exactly 11 players, got {len(player_ids)}")
    if len(set(player_ids)) != len(player_ids):
        errors.append("Duplicate players detected")
    return errors

def enrich_player_for_prediction(player_id, opp_team_id):
    """
    Returns enriched stats for a player, handling all edge cases:
    - New/uncapped player: use role-based averages
    - Transferred player: ignore team-specific bias
    - Missing matchup data: fallback to overall stats
    """
    is_new    = is_new_player(player_id)
    transferred = was_transferred(player_id, "")
    role      = get_player_role(player_id)
    defaults  = ROLE_DEFAULTS.get(role, ROLE_DEFAULTS["Batter"])

    result = {
        "player_id":   player_id,
        "is_new":      is_new,
        "transferred": transferred,
        "role":        role,
    }

    # Form stats
    try:
        f = form_idx.loc[player_id]
        result["batting_avg"] = float(f.get("batting_avg", defaults["batting_avg"]))
        result["batting_sr"]  = float(f.get("batting_sr",  defaults["batting_sr"]))
        result["economy"]     = float(f.get("economy",     defaults["economy"]))
        result["form_score"]  = {"Red-hot":6,"Superb":5,"Excellent":4,
                                  "Average":3,"Below Average":2,"Poor":1}.get(
                                  str(f.get("form_level","Average")), 3)
    except KeyError:
        result["batting_avg"] = defaults["batting_avg"]
        result["batting_sr"]  = defaults["batting_sr"]
        result["economy"]     = defaults["economy"]
        result["form_score"]  = defaults["form_score"]

    # vs team stats (skip if transferred - use overall)
    if not transferred:
        try:
            pvt_row = pvt_idx.loc[(player_id, opp_team_id)]
            result["vs_team_sr"]  = float(pvt_row.get("batting_strike_rate", result["batting_sr"]))
            result["vs_team_avg"] = float(pvt_row.get("batting_average",     result["batting_avg"]))
        except KeyError:
            result["vs_team_sr"]  = result["batting_sr"]
            result["vs_team_avg"] = result["batting_avg"]
    else:
        result["vs_team_sr"]  = result["batting_sr"]
        result["vs_team_avg"] = result["batting_avg"]

    return result
