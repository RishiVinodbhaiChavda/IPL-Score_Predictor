"""Loads all CSV data into memory for fast API responses."""
import pandas as pd
import os

BASE = os.path.join(os.path.dirname(__file__), "../../../Dataset")

# Cache for loaded data
_data_cache = {}

def _load(name, **kwargs):
    """Load CSV with caching to avoid reloading."""
    if name not in _data_cache:
        path = os.path.join(BASE, name)
        _data_cache[name] = pd.read_csv(path, encoding="latin-1", **kwargs)
    return _data_cache[name]

# Load only essential data on startup
players        = _load("players.csv")
teams          = _load("teams.csv")
venues         = _load("venues.csv")
squads         = _load("squads.csv")
matches        = _load("matches.csv")
recent_form    = _load("player_recent_form.csv")

# Lazy load other data only when needed
def _get_pvt():
    if "pvt" not in _data_cache:
        _data_cache["pvt"] = _load("player_vs_team.csv")
    return _data_cache["pvt"]

def _get_pvp():
    if "pvp" not in _data_cache:
        _data_cache["pvp"] = _load("player_vs_player.csv")
    return _data_cache["pvp"]

def _get_pvv():
    if "pvv" not in _data_cache:
        _data_cache["pvv"] = _load("player_venue_stats.csv")
    return _data_cache["pvv"]

def _get_phase():
    if "phase" not in _data_cache:
        _data_cache["phase"] = _load("player_phase_stats.csv")
    return _data_cache["phase"]

def _get_bat_type():
    if "bat_type" not in _data_cache:
        _data_cache["bat_type"] = _load("player_batting_vs_type.csv")
    return _data_cache["bat_type"]

def _get_bowl_overall():
    if "bowl_overall" not in _data_cache:
        _data_cache["bowl_overall"] = _load("player_bowling_overall.csv")
    return _data_cache["bowl_overall"]

# Indexed lookups
players_idx    = players.set_index("player_id")
form_idx       = recent_form.set_index("player_id")

# Lazy-loaded indexes
_pvt_idx = None
_pvp_bat_idx = None
_pvv_idx = None
_phase_idx = None
_bat_type_idx = None
_bowl_idx = None

def get_pvt_idx():
    global _pvt_idx
    if _pvt_idx is None:
        pvt = _get_pvt()
        _pvt_idx = pvt.set_index(["player_id", "opponent_team_id"])
    return _pvt_idx

def get_pvp_bat_idx():
    global _pvp_bat_idx
    if _pvp_bat_idx is None:
        pvp = _get_pvp()
        _pvp_bat_idx = pvp.set_index(["batter_id", "bowler_id"])
    return _pvp_bat_idx

def get_pvv_idx():
    global _pvv_idx
    if _pvv_idx is None:
        pvv = _get_pvv()
        _pvv_idx = pvv.set_index(["player_id", "venue_id"])
    return _pvv_idx

def get_phase_idx():
    global _phase_idx
    if _phase_idx is None:
        phase = _get_phase()
        _phase_idx = phase.set_index(["player_id", "season", "phase"])
    return _phase_idx

def get_bat_type_idx():
    global _bat_type_idx
    if _bat_type_idx is None:
        bat_type = _get_bat_type()
        _bat_type_idx = bat_type.set_index(["player_id", "season", "bowling_type"])
    return _bat_type_idx

def get_bowl_idx():
    global _bowl_idx
    if _bowl_idx is None:
        bowl_overall = _get_bowl_overall()
        _bowl_idx = bowl_overall.set_index(["player_id", "season"])
    return _bowl_idx

# Backward compatibility - keep old names but use lazy loading
pvt_idx = None
pvp_bat_idx = None
pvv_idx = None
phase_idx = None
bat_type_idx = None
bowl_idx = None

def _ensure_indexes():
    """Ensure all indexes are loaded when needed."""
    global pvt_idx, pvp_bat_idx, pvv_idx, phase_idx, bat_type_idx, bowl_idx
    if pvt_idx is None:
        pvt_idx = get_pvt_idx()
    if pvp_bat_idx is None:
        pvp_bat_idx = get_pvp_bat_idx()
    if pvv_idx is None:
        pvv_idx = get_pvv_idx()
    if phase_idx is None:
        phase_idx = get_phase_idx()
    if bat_type_idx is None:
        bat_type_idx = get_bat_type_idx()
    if bowl_idx is None:
        bowl_idx = get_bowl_idx()

# Venue avg scores
venue_avg = matches.groupby("venue_id")["first_innings_score"].mean().round(2)
