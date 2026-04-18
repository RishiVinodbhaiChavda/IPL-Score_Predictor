from fastapi import APIRouter, Query
from db.data_loader import players, squads, form_idx, pvt_idx, bowl_idx
import pandas as pd, math

router = APIRouter()

def safe(val, default=0.0):
    try:
        v = float(val)
        return default if math.isnan(v) or math.isinf(v) else round(v, 2)
    except:
        return default

@router.get("/players")
def get_players(team_id: str = Query(...)):
    # Get 2026 squad (most recent)
    sq = squads[(squads["team_id"]==team_id) & (squads["season"]==2026)]
    if sq.empty:
        sq = squads[(squads["team_id"]==team_id)].sort_values("season", ascending=False)
        sq = sq.drop_duplicates("player_id")

    result = []
    for _, row in sq.iterrows():
        pid  = row["player_id"]
        prow = players[players["player_id"]==pid]
        if prow.empty:
            continue
        p = prow.iloc[0]

        # Recent form
        bat_avg, bat_sr, wkts, econ, form_level = 0.0, 0.0, 0, 0.0, "Average"
        try:
            f = form_idx.loc[pid]
            bat_avg    = safe(f.get("batting_avg", 0))
            bat_sr     = safe(f.get("batting_sr", 0))
            wkts       = int(safe(f.get("wickets_taken", 0)))
            econ       = safe(f.get("economy", 0))
            form_level = str(f.get("form_level", "Average"))
        except KeyError:
            pass

        result.append({
            "player_id":     pid,
            "player_name":   p["player_name"],
            "role":          row["role_in_squad"],
            "batting_style": str(p.get("batting_style", "")),
            "bowling_style": str(p.get("bowling_style", "")),
            "nationality":   str(p.get("nationality", "")),
            "is_overseas":   int(p.get("is_overseas", 0)),
            "photo_url":     str(p.get("photo_url", "")),
            "batting_avg":   bat_avg,
            "batting_sr":    bat_sr,
            "wickets":       wkts,
            "economy":       econ,
            "form_level":    form_level,
        })
    return result
