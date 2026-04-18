"""
IPL Score Prediction - Feature Engineering
==========================================
Feature priority (as per domain knowledge):
1. Player vs Player matchups (pvp) - most specific signal
2. Player recent form - current momentum
3. Player at venue - ground-specific performance
4. Team vs Team history - head-to-head
5. Player vs Team - player-specific team record
6. Player phase stats - powerplay/death specialists
7. Individual batting/bowling records
8. Pitch + Weather conditions
9. Season trend (scores increasing year on year)
"""
import numpy as np
import math
from db.data_loader import (
    pvt_idx, pvv_idx, phase_idx, bat_type_idx,
    form_idx, bowl_idx, venue_avg, matches, venues, pvp
)

FORM_SCORE = {"Red-hot": 6, "Superb": 5, "Excellent": 4,
              "Average": 3, "Below Average": 2, "Poor": 1}

# Venue pitch type from venues.csv
VENUE_PITCH = dict(zip(venues["venue_id"], venues["pitch_type"])) \
    if "pitch_type" in venues.columns else {}

# Pitch score delta (runs above/below average)
PITCH_DELTA = {"Batting": 10.0, "Balanced": 0.0, "Bowling": -10.0}

# Season trend: IPL avg first innings score per season (computed from data)
# Model uses this to understand the era of cricket being played
SEASON_AVG = {
    2015: 159.0, 2016: 161.0, 2017: 163.0, 2018: 164.0,
    2019: 163.0, 2020: 162.0, 2021: 159.0, 2022: 171.0,
    2023: 183.0, 2024: 190.0, 2025: 189.0, 2026: 200.0,
}

# Pre-index pvp for fast lookup: (batter_id, bowler_id) -> row
_pvp_bat_idx = pvp.set_index(["batter_id", "bowler_id"])
_pvp_bowl_idx = pvp.set_index(["bowler_id", "batter_id"])


def safe(val, default=0.0):
    try:
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


def smean(lst, default=0.0):
    v = [x for x in lst if x is not None and not math.isnan(float(x)) and float(x) != 0]
    return round(sum(v) / len(v), 3) if v else default


def ssum(lst):
    v = [x for x in lst if x is not None and not math.isnan(float(x))]
    return round(sum(v), 2) if v else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 1. PLAYER VS PLAYER MATCHUPS (highest priority)
# ══════════════════════════════════════════════════════════════════════════════
def pvp_batting_features(bat_pids, bowl_pids):
    """
    Aggregate how the batting team's batters perform vs the bowling team's bowlers.
    Returns: avg SR, avg dismissal rate, total runs, matchup coverage ratio
    """
    sr_list, dis_rate_list, runs_list = [], [], []
    matched = 0
    total_pairs = len(bat_pids) * len(bowl_pids)

    for batter in bat_pids:
        for bowler in bowl_pids:
            try:
                row = _pvp_bat_idx.loc[(batter, bowler)]
                balls = safe(row["balls_faced"])
                if balls >= 6:  # min 6 balls for reliability
                    runs = safe(row["runs_scored"])
                    dis  = safe(row["dismissals"])
                    sr_list.append(runs / balls * 100)
                    dis_rate_list.append(dis / balls * 6)  # dismissals per over
                    runs_list.append(runs)
                    matched += 1
            except KeyError:
                pass

    coverage = matched / max(total_pairs, 1)
    return (
        smean(sr_list, default=130.0),      # avg SR in matchups
        smean(dis_rate_list, default=1.0),  # avg dismissal rate
        ssum(runs_list),                    # total runs in matchups
        round(coverage, 3),                 # how much matchup data we have
    )


def pvp_bowling_features(bowl_pids, bat_pids):
    """
    Aggregate how the bowling team's bowlers perform vs the batting team's batters.
    Returns: avg economy, avg wicket rate, total wickets
    """
    econ_list, wkt_rate_list, wkts_list = [], [], []

    for bowler in bowl_pids:
        for batter in bat_pids:
            try:
                row = _pvp_bowl_idx.loc[(bowler, batter)]
                balls = safe(row["balls_faced"])
                if balls >= 6:
                    runs = safe(row["runs_scored"])
                    wkts = safe(row["dismissals"])
                    econ_list.append(runs / (balls / 6))
                    wkt_rate_list.append(wkts / (balls / 6))
                    wkts_list.append(wkts)
            except KeyError:
                pass

    return (
        smean(econ_list, default=8.5),      # avg economy in matchups
        smean(wkt_rate_list, default=1.0),  # avg wickets per over
        ssum(wkts_list),                    # total wickets in matchups
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. PLAYER RECENT FORM (current momentum)
# ══════════════════════════════════════════════════════════════════════════════
def bat_form_features(bat_pids):
    """Recent form of batting lineup."""
    form_scores, sr_list, avg_list, runs_list = [], [], [], []
    for pid in bat_pids:
        try:
            r = form_idx.loc[pid]
            role = str(r.get("player_role", "Batter")).lower()
            if "bowler" in role and "allrounder" not in role:
                continue  # skip pure bowlers
            form_scores.append(FORM_SCORE.get(str(r.get("form_level", "Average")), 3))
            if safe(r.get("batting_sr", 0)) > 0:
                sr_list.append(safe(r["batting_sr"]))
            if safe(r.get("batting_avg", 0)) > 0:
                avg_list.append(safe(r["batting_avg"]))
            if safe(r.get("runs_scored", 0)) > 0:
                runs_list.append(safe(r["runs_scored"]))
        except KeyError:
            pass
    return (
        smean(form_scores, default=3.0),
        smean(sr_list, default=130.0),
        smean(avg_list, default=25.0),
        ssum(runs_list),
    )


def bowl_form_features(bowl_pids):
    """Recent form of bowling lineup."""
    form_scores, econ_list, wkt_list = [], [], []
    for pid in bowl_pids:
        try:
            r = form_idx.loc[pid]
            role = str(r.get("player_role", "Bowler")).lower()
            if "batter" in role and "allrounder" not in role:
                continue
            form_scores.append(FORM_SCORE.get(str(r.get("form_level", "Average")), 3))
            if safe(r.get("economy", 0)) > 0:
                econ_list.append(safe(r["economy"]))
            if safe(r.get("wickets_taken", 0)) > 0:
                wkt_list.append(safe(r["wickets_taken"]))
        except KeyError:
            pass
    return (
        smean(form_scores, default=3.0),
        smean(econ_list, default=8.5),
        ssum(wkt_list),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. PLAYER AT VENUE
# ══════════════════════════════════════════════════════════════════════════════
def bat_venue_features(bat_pids, venue_id):
    sr_list, avg_list, runs_list = [], [], []
    for pid in bat_pids:
        try:
            r = pvv_idx.loc[(pid, venue_id)]
            if safe(r["balls_faced"]) >= 20:
                sr_list.append(safe(r["batting_sr"]))
                avg_list.append(safe(r["batting_avg"]))
                runs_list.append(safe(r["runs_scored"]))
        except KeyError:
            pass
    return smean(sr_list, 130.0), smean(avg_list, 25.0), ssum(runs_list)


def bowl_venue_features(bowl_pids, venue_id):
    econ_list, wkt_list = [], []
    for pid in bowl_pids:
        try:
            r = pvv_idx.loc[(pid, venue_id)]
            if safe(r.get("overs_bowled", 0)) >= 4:
                econ_list.append(safe(r["bowling_economy"]))
                wkt_list.append(safe(r["wickets"]))
        except KeyError:
            pass
    return smean(econ_list, 8.5), ssum(wkt_list)


# ══════════════════════════════════════════════════════════════════════════════
# 4. TEAM VS TEAM (head-to-head at this venue)
# ══════════════════════════════════════════════════════════════════════════════
def team_vs_team_features(bat_team, bowl_team, venue_id):
    """Historical scores when bat_team batted vs bowl_team at this venue."""
    h2h = matches[
        (matches["team1_id"] == bat_team) &
        (matches["team2_id"] == bowl_team)
    ]["first_innings_score"]
    h2h_venue = matches[
        (matches["team1_id"] == bat_team) &
        (matches["team2_id"] == bowl_team) &
        (matches["venue_id"] == venue_id)
    ]["first_innings_score"]

    overall_avg = safe(h2h.mean(), 173.0)
    venue_avg_h2h = safe(h2h_venue.mean(), overall_avg)
    h2h_count = len(h2h)
    return overall_avg, venue_avg_h2h, h2h_count


# ══════════════════════════════════════════════════════════════════════════════
# 5. PLAYER VS TEAM
# ══════════════════════════════════════════════════════════════════════════════
def bat_vs_team_features(bat_pids, opp_team):
    sr_list, avg_list, runs_list = [], [], []
    for pid in bat_pids:
        try:
            r = pvt_idx.loc[(pid, opp_team)]
            if safe(r.get("runs_scored", 0)) > 0:
                sr_list.append(safe(r["batting_strike_rate"]))
                avg_list.append(safe(r["batting_average"]))
                runs_list.append(safe(r["runs_scored"]))
        except KeyError:
            pass
    return smean(sr_list, 130.0), smean(avg_list, 25.0), ssum(runs_list)


def bowl_vs_team_features(bowl_pids, opp_team):
    econ_list, avg_list, wkt_list = [], [], []
    for pid in bowl_pids:
        try:
            r = pvt_idx.loc[(pid, opp_team)]
            if safe(r.get("wickets_taken", 0)) > 0:
                econ_list.append(safe(r["bowling_economy"]))
                avg_list.append(safe(r.get("bowling_average", 0)))
                wkt_list.append(safe(r["wickets_taken"]))
        except KeyError:
            pass
    return smean(econ_list, 8.5), smean(avg_list, 25.0), ssum(wkt_list)


# ══════════════════════════════════════════════════════════════════════════════
# 6. PHASE STATS (powerplay/death specialists)
# ══════════════════════════════════════════════════════════════════════════════
def bat_phase_features(bat_pids, season, phase_name):
    sr_list, runs_list = [], []
    for pid in bat_pids:
        try:
            r = phase_idx.loc[(pid, season, phase_name)]
            if safe(r.get("balls_faced", 0)) >= 6:
                sr_list.append(safe(r["batting_strike_rate"]))
                runs_list.append(safe(r["runs_scored"]))
        except KeyError:
            pass
    return smean(sr_list, 130.0), ssum(runs_list)


def bowl_phase_features(bowl_pids, season, phase_name):
    econ_list, wkt_list = [], []
    for pid in bowl_pids:
        try:
            r = phase_idx.loc[(pid, season, phase_name)]
            if safe(r.get("overs_bowled", 0)) >= 1:
                econ_list.append(safe(r["bowling_economy"]))
                wkt_list.append(safe(r["wickets"]))
        except KeyError:
            pass
    return smean(econ_list, 8.5), ssum(wkt_list)


# ══════════════════════════════════════════════════════════════════════════════
# 7. INDIVIDUAL RECORDS (batting vs pace/spin)
# ══════════════════════════════════════════════════════════════════════════════
def bat_vs_type_features(bat_pids, season, btype):
    sr_list, runs_list = [], []
    for pid in bat_pids:
        try:
            r = bat_type_idx.loc[(pid, season, btype)]
            sr_list.append(safe(r["strike_rate"]))
            runs_list.append(safe(r["runs"]))
        except KeyError:
            pass
    return smean(sr_list, 130.0), ssum(runs_list)


# ══════════════════════════════════════════════════════════════════════════════
# 8. WEATHER & PITCH
# ══════════════════════════════════════════════════════════════════════════════
def weather_features(temperature, humidity, dew_factor):
    temp  = safe(temperature, 30.0)
    humid = safe(humidity, 60.0)
    dew   = safe(dew_factor, 3.0)

    temp_norm  = (temp - 15.0) / 30.0
    humid_norm = (humid - 20.0) / 75.0
    dew_norm   = dew / 10.0

    # Dew > 5: pitch stays true, batting easier
    dew_impact   = (dew - 5.0) * 1.5 if dew > 5 else (dew - 5.0) * 0.5
    temp_impact  = -(temp - 38.0) * 0.3 if temp > 38 else 0.0
    humid_impact = -(humid - 75.0) * 0.15 if humid > 75 else 0.0
    total_impact = round(dew_impact + temp_impact + humid_impact, 2)

    return temp_norm, humid_norm, dew_norm, total_impact


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_features(bat_pids, bowl_pids, venue_id, bat_team, bowl_team,
                   pitch_type, temperature, humidity, dew_factor, season=2025):

    # ── Venue & season context ─────────────────────────────────────────────────
    v_avg = safe(venue_avg.get(venue_id, 173.0))
    season_avg = SEASON_AVG.get(int(season), 173.0)
    season_trend = season_avg - 173.0  # deviation from all-time avg

    # Pitch
    actual_pitch = pitch_type
    if actual_pitch == "Balanced" and venue_id in VENUE_PITCH:
        actual_pitch = VENUE_PITCH[venue_id]
    pitch_bat   = 1.0 if actual_pitch == "Batting" else 0.0
    pitch_bowl  = 1.0 if actual_pitch == "Bowling" else 0.0
    pitch_delta = PITCH_DELTA.get(actual_pitch, 0.0)

    # Weather
    temp_n, humid_n, dew_n, w_impact = weather_features(temperature, humidity, dew_factor)

    # ── 1. PvP matchup features ────────────────────────────────────────────────
    pvp_bat_sr, pvp_bat_dis, pvp_bat_runs, pvp_coverage = pvp_batting_features(bat_pids, bowl_pids)
    pvp_bowl_econ, pvp_bowl_wkt_rate, pvp_bowl_wkts     = pvp_bowling_features(bowl_pids, bat_pids)

    # ── 2. Recent form ─────────────────────────────────────────────────────────
    bf_score, bf_sr, bf_avg, bf_runs = bat_form_features(bat_pids)
    bwf_score, bwf_econ, bwf_wkts   = bowl_form_features(bowl_pids)

    # ── 3. Player at venue ─────────────────────────────────────────────────────
    bv_sr, bv_avg, bv_runs   = bat_venue_features(bat_pids, venue_id)
    bwv_econ, bwv_wkts       = bowl_venue_features(bowl_pids, venue_id)

    # ── 4. Team vs team ────────────────────────────────────────────────────────
    h2h_avg, h2h_venue_avg, h2h_count = team_vs_team_features(bat_team, bowl_team, venue_id)

    # ── 5. Player vs team ──────────────────────────────────────────────────────
    bvt_sr, bvt_avg, bvt_runs     = bat_vs_team_features(bat_pids, bowl_team)
    bwvt_econ, bwvt_avg, bwvt_wkts = bowl_vs_team_features(bowl_pids, bat_team)

    # ── 6. Phase stats ─────────────────────────────────────────────────────────
    bpp_sr, bpp_runs   = bat_phase_features(bat_pids, season, "Powerplay")
    bmd_sr, bmd_runs   = bat_phase_features(bat_pids, season, "Middle")
    bdt_sr, bdt_runs   = bat_phase_features(bat_pids, season, "Death")
    bwpp_econ, bwpp_wkts = bowl_phase_features(bowl_pids, season, "Powerplay")
    bwmd_econ, bwmd_wkts = bowl_phase_features(bowl_pids, season, "Middle")
    bwdt_econ, bwdt_wkts = bowl_phase_features(bowl_pids, season, "Death")

    # ── 7. Batting vs pace/spin ────────────────────────────────────────────────
    bpace_sr, bpace_runs = bat_vs_type_features(bat_pids, season, "Pace")
    bspin_sr, bspin_runs = bat_vs_type_features(bat_pids, season, "Spin")

    features = np.array([
        # Venue & season (2)
        v_avg, season_trend,
        # Pitch (3)
        pitch_bat, pitch_bowl, pitch_delta,
        # Weather (4)
        temp_n, humid_n, dew_n, w_impact,
        # PvP matchups (7) — highest priority
        pvp_bat_sr, pvp_bat_dis, pvp_bat_runs,
        pvp_bowl_econ, pvp_bowl_wkt_rate, pvp_bowl_wkts,
        pvp_coverage,
        # Recent form batting (4)
        bf_score, bf_sr, bf_avg, bf_runs,
        # Recent form bowling (3)
        bwf_score, bwf_econ, bwf_wkts,
        # Player at venue (5)
        bv_sr, bv_avg, bv_runs, bwv_econ, bwv_wkts,
        # Team vs team (3)
        h2h_avg, h2h_venue_avg, h2h_count,
        # Player vs team (6)
        bvt_sr, bvt_avg, bvt_runs,
        bwvt_econ, bwvt_avg, bwvt_wkts,
        # Phase stats batting (6)
        bpp_sr, bpp_runs, bmd_sr, bmd_runs, bdt_sr, bdt_runs,
        # Phase stats bowling (6)
        bwpp_econ, bwpp_wkts, bwmd_econ, bwmd_wkts, bwdt_econ, bwdt_wkts,
        # Batting vs type (4)
        bpace_sr, bpace_runs, bspin_sr, bspin_runs,
    ], dtype=np.float32)

    return features


FEATURE_NAMES = [
    # Venue & season
    "venue_avg_score", "season_trend",
    # Pitch
    "pitch_batting", "pitch_bowling", "pitch_score_delta",
    # Weather
    "temp_normalized", "humidity_normalized", "dew_normalized", "weather_impact",
    # PvP matchups
    "pvp_bat_sr", "pvp_bat_dismissal_rate", "pvp_bat_runs",
    "pvp_bowl_economy", "pvp_bowl_wicket_rate", "pvp_bowl_wickets",
    "pvp_coverage",
    # Recent form batting
    "bat_form_score", "bat_form_sr", "bat_form_avg", "bat_form_runs",
    # Recent form bowling
    "bowl_form_score", "bowl_form_econ", "bowl_form_wkts",
    # Player at venue
    "bat_venue_sr", "bat_venue_avg", "bat_venue_runs",
    "bowl_venue_econ", "bowl_venue_wkts",
    # Team vs team
    "h2h_avg_score", "h2h_venue_avg_score", "h2h_match_count",
    # Player vs team
    "bat_vs_team_sr", "bat_vs_team_avg", "bat_vs_team_runs",
    "bowl_vs_team_econ", "bowl_vs_team_avg", "bowl_vs_team_wkts",
    # Phase batting
    "bat_powerplay_sr", "bat_powerplay_runs",
    "bat_middle_sr", "bat_middle_runs",
    "bat_death_sr", "bat_death_runs",
    # Phase bowling
    "bowl_powerplay_econ", "bowl_powerplay_wkts",
    "bowl_middle_econ", "bowl_middle_wkts",
    "bowl_death_econ", "bowl_death_wkts",
    # Batting vs type
    "bat_vs_pace_sr", "bat_vs_pace_runs",
    "bat_vs_spin_sr", "bat_vs_spin_runs",
]
