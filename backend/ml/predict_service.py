"""Orchestrates prediction: feature engineering + edge cases + model + explainability."""
import numpy as np
from ml.feature_engineering import build_features, FEATURE_NAMES
from ml.edge_case_handler import validate_playing11, enrich_player_for_prediction
from ml.model_loader import predict_score

def get_top_factors(features, feature_names):
    """Simple explainability: return top 3 contributing features."""
    # Normalize features to 0-1 range for comparison
    f = np.array(features, dtype=float)
    f_norm = (f - f.min()) / (f.max() - f.min() + 1e-9)

    # Map feature names to human-readable labels
    LABELS = {
        "venue_avg_score":          "Historical venue average score",
        "batting_team_avg_at_venue":"Batting team's record at this venue",
        "bat_venue_avg_sr":         "Batting team's strike rate at venue",
        "bat_vs_opp_sr":            "Batting team's SR vs this bowling attack",
        "bat_vs_opp_avg":           "Batting team's average vs this bowling attack",
        "bat_powerplay_sr":         "Batting team's powerplay strike rate",
        "bat_death_sr":             "Batting team's death overs strike rate",
        "bat_form_score":           "Batting team's current form",
        "bat_form_sr":              "Batting team's recent strike rate",
        "bowl_venue_econ":          "Bowling team's economy at this venue",
        "bowl_vs_opp_econ":         "Bowling team's economy vs this batting side",
        "bowl_powerplay_econ":      "Bowling team's powerplay economy",
        "bowl_death_econ":          "Bowling team's death overs economy",
        "bowl_form_score":          "Bowling team's current form",
        "pitch_batting":            "Batting-friendly pitch",
        "pitch_bowling":            "Bowling-friendly pitch",
        "weather_score":            "Weather conditions (temp/humidity/dew)",
        "bat_vs_pace_sr":           "Batting team's SR vs pace bowling",
        "bat_vs_spin_sr":           "Batting team's SR vs spin bowling",
    }

    top_idx = np.argsort(f_norm)[::-1][:5]
    factors = []
    for idx in top_idx:
        if idx < len(feature_names):
            name  = feature_names[idx]
            label = LABELS.get(name, name.replace("_", " ").title())
            val   = float(features[idx])
            factors.append({"feature": name, "label": label, "value": round(val, 2)})
    return factors[:3]

def run_prediction(payload: dict):
    """
    payload = {
        batting_team_id, bowling_team_id,
        batting_xi: [player_ids],
        bowling_xi: [player_ids],
        venue_id, pitch_type,
        temperature, humidity, dew_factor
    }
    """
    bat_xi   = payload["batting_xi"]
    bowl_xi  = payload["bowling_xi"]
    venue_id = payload["venue_id"]
    bat_team = payload["batting_team_id"]
    bowl_team= payload["bowling_team_id"]

    # Validate
    errors = validate_playing11(bat_xi) + validate_playing11(bowl_xi)
    if errors:
        return {"error": errors}

    # Build features
    features = build_features(
        bat_pids    = bat_xi,
        bowl_pids   = bowl_xi,
        venue_id    = venue_id,
        bat_team    = bat_team,
        bowl_team   = bowl_team,
        pitch_type  = payload.get("pitch_type", "Balanced"),
        temperature = payload.get("temperature", 30),
        humidity    = payload.get("humidity", 60),
        dew_factor  = payload.get("dew_factor", 3),
        season      = payload.get("season", 2025),
    )

    # Predict
    score, xgb_pred, mlp_pred = predict_score(features)
    score = max(80, min(300, score))  # clamp to realistic range

    # Score range (±15 based on std dev of historical data)
    std = 33.0
    low  = max(60,  int(score - std * 0.45))
    high = min(300, int(score + std * 0.45))

    # Confidence: based on how much data we have
    data_richness = sum(1 for v in features if v > 0) / len(features)
    confidence    = int(55 + data_richness * 35)  # 55-90%

    # Top factors
    factors = get_top_factors(features, FEATURE_NAMES)

    # Simulation: run 5 slight variations
    simulations = []
    for noise in [-0.03, -0.015, 0, 0.015, 0.03]:
        noisy = features * (1 + noise + np.random.normal(0, 0.01, len(features)))
        s, _, _ = predict_score(noisy)
        simulations.append(max(80, min(300, round(s, 1))))

    xgb_score, mlp_score = round(xgb_pred, 1), round(mlp_pred, 1)

    return {
        "score":       round(score, 1),
        "range":       [low, high],
        "confidence":  confidence,
        "xgb_score":   xgb_score,
        "nn_score":    mlp_score,
        "factors":     factors,
        "simulations": simulations,
    }
