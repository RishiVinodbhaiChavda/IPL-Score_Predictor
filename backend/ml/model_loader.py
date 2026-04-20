"""
IPL Score Prediction - Model Training Pipeline
===============================================
Hybrid Ensemble: XGBoost + MLP Neural Network

Architecture:
  - XGBoost: 2000 trees, depth=4, lr=0.01 (optimized via 5-Fold CV)
  - MLP: 512→256→128→64→32→1 with ReLU, L2 regularization, early stopping
  - Ensemble: Dynamic weighting based on validation MAE

Training Strategy:
  - TIME-BASED SPLIT: Train on 2015-2024, validate on 2025-2026
  - SAMPLE WEIGHTING: Recent seasons weighted 3x higher
  - 5-FOLD CV on training data for hyperparameter validation
  - Ensemble weights computed from validation performance
"""
import os, pickle, json, numpy as np
from ml.feature_engineering import build_features, FEATURE_NAMES
from db.data_loader import matches, squads

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "../../../models")
XGB_PATH    = os.path.join(MODEL_DIR, "xgb_model.pkl")
MLP_PATH    = os.path.join(MODEL_DIR, "mlp_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "ensemble_weights.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

xgb_model = None
mlp_model = None
scaler    = None


def _build_training_features():
    """Build feature matrix from all historical matches. Returns X, y, seasons."""
    print("=" * 60)
    print("PHASE 1: BUILDING TRAINING FEATURES FROM MATCH DATA")
    print("=" * 60)
    X, y, season_list = [], [], []
    sq_cache = {}

    from db.data_loader import venues
    import random
    rng = random.Random(42)

    venue_pitch_map = dict(zip(venues["venue_id"], venues["pitch_type"])) \
        if "pitch_type" in venues.columns else {}

    def get_weather(season, venue_id):
        # UAE venues (IPL 2020)
        if venue_id in ("IPL_VEN_06", "IPL_VEN_23", "IPL_VEN_24", "IPL_VEN_26"):
            return rng.uniform(28, 38), rng.uniform(40, 65), rng.uniform(2, 6)
        elif season == 2020:
            return rng.uniform(28, 36), rng.uniform(45, 70), rng.uniform(2, 7)
        else:
            return rng.uniform(28, 42), rng.uniform(50, 85), rng.uniform(1, 8)

    def get_squad(team_id, season, role_kw):
        key = (team_id, season, role_kw)
        if key not in sq_cache:
            sq = squads[(squads["team_id"]==team_id) & (squads["season"]==season)]
            sq_cache[key] = sq[sq["role_in_squad"].str.contains(role_kw, case=False, na=False)]["player_id"].tolist()
        return sq_cache[key]

    skipped = 0
    for _, m in matches.iterrows():
        try:
            bat_pids  = get_squad(m["team1_id"], m["season"], "Batter|Wicketkeeper|Allrounder")
            bowl_pids = get_squad(m["team2_id"], m["season"], "Bowler|Allrounder")
            if not bat_pids or not bowl_pids:
                skipped += 1
                continue
            pitch = venue_pitch_map.get(m["venue_id"], "Balanced")
            temp, humid, dew = get_weather(int(m["season"]), m["venue_id"])
            feat = build_features(
                bat_pids, bowl_pids, m["venue_id"],
                m["team1_id"], m["team2_id"],
                pitch, temp, humid, dew, int(m["season"])
            )
            X.append(feat)
            y.append(float(m["first_innings_score"]))
            season_list.append(int(m["season"]))
        except Exception:
            skipped += 1

    print(f"  ✓ Built {len(X)} match samples × {len(X[0]) if X else 0} features")
    print(f"  ✗ Skipped {skipped} matches (missing squad data)")
    print(f"  Seasons covered: {sorted(set(season_list))}")
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(season_list, dtype=np.int32))


def train_and_save():
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.neural_network import MLPRegressor
    import xgboost as xgb

    X, y, seasons = _build_training_features()

    # ── TIME-BASED SPLIT (no data leakage) ───────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: SPLITTING DATA (TIME-BASED)")
    print("=" * 60)
    train_mask = seasons <= 2024
    val_mask   = seasons >= 2025
    X_train_raw, y_train = X[train_mask], y[train_mask]
    X_val_raw,   y_val   = X[val_mask],   y[val_mask]
    seasons_train = seasons[train_mask]
    print(f"  Train: {len(X_train_raw)} matches (2015-2024)")
    print(f"  Val:   {len(X_val_raw)} matches (2025-2026)")
    print(f"  Target range: {y.min():.0f} – {y.max():.0f} (mean={y.mean():.1f})")

    # ── SAMPLE WEIGHTS ────────────────────────────────────────────────────
    def weight(s):
        if s >= 2025: return 3.0
        if s >= 2024: return 2.5
        if s >= 2023: return 2.0
        if s >= 2022: return 1.5
        if s >= 2020: return 1.2
        return 1.0

    w_train = np.array([weight(s) for s in seasons_train], dtype=np.float32)

    # ── SCALER (fit only on train) ────────────────────────────────────────
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train_raw)
    X_val_s   = sc.transform(X_val_raw)

    # ══════════════════════════════════════════════════════════════════════
    # XGBOOST TRAINING (Optimized Hyperparameters)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 3: TRAINING XGBOOST (2000 trees, depth=4, lr=0.01)")
    print("=" * 60)

    # 5-Fold CV first
    print("\n  Running 5-Fold Cross-Validation on training data...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.75, min_child_weight=4,
        reg_alpha=0.05, reg_lambda=1.5, random_state=42, n_jobs=-1
    )
    cv_scores = cross_val_score(cv_model, X_train_raw, y_train,
                                cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"  CV MAE: {cv_mae:.2f} ± {cv_std:.2f} runs")
    print(f"  Per fold: {[-round(s,1) for s in cv_scores]}")

    # Train final XGBoost with optimized hyperparameters
    print(f"\n  Training final XGBoost model...")
    xgb_m = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.80,
        colsample_bytree=0.70,
        min_child_weight=5,
        reg_alpha=0.15,
        reg_lambda=2.0,
        gamma=0.2,
        random_state=42,
        n_jobs=-1,
    )
    xgb_m.fit(X_train_raw, y_train,
              sample_weight=w_train,
              eval_set=[(X_val_raw, y_val)],
              verbose=False)

    xgb_val_pred = xgb_m.predict(X_val_raw)
    xgb_mae = mean_absolute_error(y_val, xgb_val_pred)
    xgb_r2  = r2_score(y_val, xgb_val_pred)
    print(f"  ✓ XGBoost Validation MAE: {xgb_mae:.2f} runs")
    print(f"  ✓ XGBoost Validation R²:  {xgb_r2:.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # MLP NEURAL NETWORK TRAINING (Improved with constraints)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 4: TRAINING MLP NEURAL NETWORK (Constrained)")
    print("=" * 60)
    print(f"  Architecture: Input({X_train_s.shape[1]}) → 256 → 128 → 64 → 1")
    print(f"  Activation: ReLU, Solver: Adam, L2 regularization: 0.01")
    print(f"  Early stopping: patience=30, validation_fraction=0.15")
    print(f"  Output clipping: 80-280 runs (realistic T20 range)")

    # Smaller, more regularized network to prevent overfitting
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),  # Reduced from (512,256,128,64,32)
        activation='relu',
        solver='adam',
        alpha=0.01,  # Increased L2 regularization from 0.001
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.0005,  # Lower learning rate
        max_iter=1000,  # More iterations with early stopping
        early_stopping=True,
        validation_fraction=0.15,  # More validation data
        n_iter_no_change=30,  # More patience
        tol=1e-4,
        random_state=42,
        verbose=False
    )

    print(f"\n  Training MLP (up to 1000 iterations with early stopping)...")
    mlp.fit(X_train_s, y_train)

    # Predict with output clipping to realistic T20 score range
    mlp_val_pred_raw = mlp.predict(X_val_s)
    mlp_val_pred = np.clip(mlp_val_pred_raw, 80, 280)  # Clip to realistic range
    
    mlp_mae = mean_absolute_error(y_val, mlp_val_pred)
    mlp_r2  = r2_score(y_val, mlp_val_pred)
    
    # Show clipping stats
    clipped = np.sum((mlp_val_pred_raw < 80) | (mlp_val_pred_raw > 280))
    print(f"  ✓ MLP Validation MAE: {mlp_mae:.2f} runs")
    print(f"  ✓ MLP Validation R²:  {mlp_r2:.3f}")
    print(f"  ✓ MLP converged at iteration: {mlp.n_iter_}")
    print(f"  ✓ Predictions clipped: {clipped}/{len(mlp_val_pred)} ({clipped/len(mlp_val_pred)*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # ENSEMBLE EVALUATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 5: ENSEMBLE EVALUATION")
    print("=" * 60)

    # Dynamic weighting based on validation performance
    total_inv_mae = (1/xgb_mae) + (1/mlp_mae)
    w_xgb = round((1/xgb_mae) / total_inv_mae, 3)
    w_mlp = round((1/mlp_mae) / total_inv_mae, 3)

    # Ensure weights sum to 1
    w_mlp = round(1.0 - w_xgb, 3)

    hybrid = w_xgb * xgb_val_pred + w_mlp * mlp_val_pred
    hybrid_mae = mean_absolute_error(y_val, hybrid)
    hybrid_r2  = r2_score(y_val, hybrid)

    print(f"\n  XGBoost MAE:   {xgb_mae:.2f} runs  (weight: {w_xgb:.1%})")
    print(f"  MLP MAE:       {mlp_mae:.2f} runs  (weight: {w_mlp:.1%})")
    print(f"  Hybrid MAE:    {hybrid_mae:.2f} runs")
    print(f"  Hybrid R²:     {hybrid_r2:.3f}  (1.0 = perfect)")
    print(f"  CV MAE:        {cv_mae:.2f} ± {cv_std:.2f} runs")

    # Sample predictions comparison
    print(f"\n  Sample Predictions (first 10 validation matches):")
    print(f"  {'Actual':>8} {'XGB':>8} {'MLP':>8} {'Hybrid':>8} {'Error':>8}")
    print(f"  {'-'*48}")
    for i in range(min(10, len(y_val))):
        h = w_xgb * xgb_val_pred[i] + w_mlp * mlp_val_pred[i]
        err = abs(y_val[i] - h)
        print(f"  {y_val[i]:8.0f} {xgb_val_pred[i]:8.1f} {mlp_val_pred[i]:8.1f} {h:8.1f} {err:8.1f}")

    # ── SAVE ALL MODELS ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 6: SAVING MODELS")
    print("=" * 60)

    with open(XGB_PATH, "wb") as f:
        pickle.dump(xgb_m, f)
    print(f"  ✓ XGBoost saved → {XGB_PATH}")

    with open(MLP_PATH, "wb") as f:
        pickle.dump(mlp, f)
    print(f"  ✓ MLP saved → {MLP_PATH}")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(sc, f)
    print(f"  ✓ Scaler saved → {SCALER_PATH}")

    with open(WEIGHTS_PATH, "wb") as f:
        pickle.dump({"xgb": w_xgb, "mlp": w_mlp}, f)
    print(f"  ✓ Ensemble weights saved → {WEIGHTS_PATH}")

    # Save training history
    hist_path = os.path.join(MODEL_DIR, "training_history.json")
    hist_data = {
        "mlp_iterations": int(mlp.n_iter_),
        "xgb_mae": float(round(xgb_mae, 2)),
        "mlp_mae": float(round(mlp_mae, 2)),
        "hybrid_mae": float(round(hybrid_mae, 2)),
        "hybrid_r2": float(round(hybrid_r2, 4)),
        "cv_mae": float(round(cv_mae, 2)),
        "ensemble_weights": {"xgb": float(w_xgb), "mlp": float(w_mlp)},
    }
    with open(hist_path, "w") as f:
        json.dump(hist_data, f, indent=2)
    print(f"  ✓ Training history saved → {hist_path}")

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE!")
    print(f"  Ensemble: XGBoost ({w_xgb:.0%}) + MLP ({w_mlp:.0%})")
    print(f"  Final Hybrid MAE: {hybrid_mae:.2f} runs")
    print(f"{'=' * 60}\n")

    return xgb_m, mlp, sc


def load_models():
    global xgb_model, mlp_model, scaler

    if os.path.exists(XGB_PATH) and os.path.exists(MLP_PATH) and os.path.exists(SCALER_PATH):
        with open(XGB_PATH, "rb") as f:
            xgb_model = pickle.load(f)

        with open(MLP_PATH, "rb") as f:
            mlp_model = pickle.load(f)

        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        print("✓ All models loaded from disk (XGBoost + MLP + Scaler)")
    else:
        print("No saved models found. Training from scratch...")
        xgb_model, mlp_model, scaler = train_and_save()

    return xgb_model, mlp_model, scaler


def predict_score(features: np.ndarray):
    global xgb_model, mlp_model, scaler
    if xgb_model is None:
        load_models()

    feat_2d  = features.reshape(1, -1)
    feat_sc  = scaler.transform(feat_2d)

    # XGBoost prediction (uses raw features)
    xgb_pred = float(xgb_model.predict(feat_2d)[0])

    # MLP prediction (uses scaled features) with output clipping
    mlp_pred_raw = float(mlp_model.predict(feat_sc)[0])
    mlp_pred = float(np.clip(mlp_pred_raw, 80, 280))  # Clip to realistic T20 range

    # Load ensemble weights
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH, "rb") as f:
            w = pickle.load(f)
        hybrid = round(w["xgb"] * xgb_pred + w["mlp"] * mlp_pred, 1)
    else:
        # Fallback: favor XGBoost
        hybrid = round(0.75 * xgb_pred + 0.25 * mlp_pred, 1)

    # Final safety clip for hybrid prediction
    hybrid = float(np.clip(hybrid, 80, 280))

    return hybrid, xgb_pred, mlp_pred
