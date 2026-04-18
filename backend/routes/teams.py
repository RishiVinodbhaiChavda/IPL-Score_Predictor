from fastapi import APIRouter
from db.data_loader import teams, venues, matches

router = APIRouter()

ACTIVE_TEAMS = ["IPL_T20_CSK","IPL_T20_DC","IPL_T20_GT","IPL_T20_KKR",
                "IPL_T20_LSG","IPL_T20_MI","IPL_T20_PBKS","IPL_T20_RR",
                "IPL_T20_RCB","IPL_T20_SRH"]

TEAM_LOGOS = {
    "IPL_T20_CSK":  "https://scores.iplt20.com/ipl/teamlogos/CSK.png",
    "IPL_T20_DC":   "https://scores.iplt20.com/ipl/teamlogos/DC.png",
    "IPL_T20_GT":   "https://scores.iplt20.com/ipl/teamlogos/GT.png",
    "IPL_T20_KKR":  "https://scores.iplt20.com/ipl/teamlogos/KKR.png",
    "IPL_T20_LSG":  "https://scores.iplt20.com/ipl/teamlogos/LSG.png",
    "IPL_T20_MI":   "https://scores.iplt20.com/ipl/teamlogos/MI.png",
    "IPL_T20_PBKS": "https://scores.iplt20.com/ipl/teamlogos/PBKS.png",
    "IPL_T20_RR":   "https://scores.iplt20.com/ipl/teamlogos/RR.png",
    "IPL_T20_RCB":  "https://scores.iplt20.com/ipl/teamlogos/RCB.png",
    "IPL_T20_SRH":  "https://scores.iplt20.com/ipl/teamlogos/SRH.png",
}

@router.get("/teams")
def get_teams():
    result = []
    for _, row in teams[teams["team_id"].isin(ACTIVE_TEAMS)].iterrows():
        result.append({
            "team_id":   row["team_id"],
            "team_name": row["team_name"],
            "short_name":row["short_name"],
            "logo_url":  TEAM_LOGOS.get(row["team_id"], ""),
        })
    return result

@router.get("/venues")
def get_venues():
    result = []
    for _, row in venues.iterrows():
        avg = matches[matches["venue_id"]==row["venue_id"]]["first_innings_score"].mean()
        result.append({
            "venue_id":   row["venue_id"],
            "venue_name": row["venue_name"],
            "city":       row["city"],
            "pitch_type": row["pitch_type"],
            "avg_score":  round(float(avg), 1) if not __import__("math").isnan(avg) else 170.0,
        })
    return result
