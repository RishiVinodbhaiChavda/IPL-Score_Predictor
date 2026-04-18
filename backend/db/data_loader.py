"""Loads all CSV data into memory for fast API responses."""
import pandas as pd
import os

BASE = os.path.join(os.path.dirname(__file__), "../../../Dataset")

def _load(name, **kwargs):
    return pd.read_csv(os.path.join(BASE, name), encoding="latin-1", **kwargs)

players        = _load("players.csv")
teams          = _load("teams.csv")
venues         = _load("venues.csv")
squads         = _load("squads.csv")
matches        = _load("matches.csv")
recent_form    = _load("player_recent_form.csv")
pvt            = _load("player_vs_team.csv")
pvp            = _load("player_vs_player.csv")
pvv            = _load("player_venue_stats.csv")
phase          = _load("player_phase_stats.csv")
bat_type       = _load("player_batting_vs_type.csv")
bowl_overall   = _load("player_bowling_overall.csv")

# Indexed lookups
players_idx    = players.set_index("player_id")
pvt_idx        = pvt.set_index(["player_id", "opponent_team_id"])
pvp_bat_idx    = pvp.set_index(["batter_id", "bowler_id"])
pvv_idx        = pvv.set_index(["player_id", "venue_id"])
phase_idx      = phase.set_index(["player_id", "season", "phase"])
bat_type_idx   = bat_type.set_index(["player_id", "season", "bowling_type"])
form_idx       = recent_form.set_index("player_id")
bowl_idx       = bowl_overall.set_index(["player_id", "season"])

# Venue avg scores
venue_avg = matches.groupby("venue_id")["first_innings_score"].mean().round(2)
