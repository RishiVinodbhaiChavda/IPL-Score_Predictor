"""
IPL Score Prediction - Model Training Pipeline
===============================================
REAL Deep Learning Implementation using TensorFlow/Keras

Architecture (Deep Neural Network):
  Input (53 features)
    → Dense(512) → BatchNorm → LeakyReLU → Dropout(0.3)
    → Dense(256) → BatchNorm → LeakyReLU → Dropout(0.25)
    → Dense(128) → BatchNorm → LeakyReLU → Dropout(0.2)
    → Dense(64)  → BatchNorm → LeakyReLU → Dropout(0.15)
    → Dense(32)  → BatchNorm → LeakyReLU
    → Dense(1)   [Linear output — predicted score]

Training Strategy:
  - TIME-BASED SPLIT: Train on 2015-2024, validate on 2025-2026
  - DATA AUGMENTATION: 8x Gaussian noise injection to expand 734 → ~5800 samples
  - SAMPLE WEIGHTING: Recent seasons weighted 3x higher
  - LEARNING RATE: Adam with ReduceLROnPlateau (patience=15, factor=0.5)
  - EARLY STOPPING: patience=40 epochs, restore best weights
  - BATCH SIZE: 32 for stable gradient updates
  - EPOCHS: up to 500 (early stopping will kick in)
  - ENSEMBLE: XGBoost (dynamic%) + DNN (dynamic%) weighted by validation MAE

XGBoost Config:
  - 1500 trees, depth=6, lr=0.02 (slow learning = better generalization)
  - L1 (alpha=0.05) + L2 (lambda=2.0) regularization
  - Subsampling (80% rows, 70% cols per tree)
"""
import os, pickle, json, numpy as np
from ml.feature_engineering import build_features, FEATURE_NAMES
from db.data_loader import matches, squads

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "../../../models")
XGB_PATH    = os.path.join(MODEL_DIR, "xgb_model.pkl")
DNN_PATH    = os.path.join(MODEL_DIR, "dnn_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "ensemble_weights.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

xgb_model = None
dnn_model = None
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


def _augment_data(X, y, seasons, n_augments=8, noise_std=0.02):
    """
    Data Augmentation via Gaussian noise injection.
    Expands dataset by n_augments times to help the DNN generalize.
    Each augmented sample adds small random noise to features.
    """
    print(f"\n  Augmenting data: {len(X)} → {len(X) * (1 + n_augments)} samples "
          f"(noise_std={noise_std})")
    X_aug = [X.copy()]
    y_aug = [y.copy()]
    s_aug = [seasons.copy()]
    rng = np.random.RandomState(42)

    for i in range(n_augments):
        noise = rng.normal(0, noise_std, X.shape).astype(np.float32)
        X_noisy = X + X * noise  # multiplicative noise
        X_aug.append(X_noisy)
        y_aug.append(y.copy())
        s_aug.append(seasons.copy())

    return np.concatenate(X_aug), np.concatenate(y_aug), np.concatenate(s_aug)


def _build_dnn(input_dim):
    """
    Build a real Deep Neural Network using TensorFlow/Keras.

    Architecture:
      Dense(512) → BatchNorm → LeakyReLU → Dropout(0.3)
      Dense(256) → BatchNorm → LeakyReLU → Dropout(0.25)
      Dense(128) → BatchNorm → LeakyReLU → Dropout(0.2)
      Dense(64)  → BatchNorm → LeakyReLU → Dropout(0.15)
      Dense(32)  → BatchNorm → LeakyReLU
      Dense(1)   → Linear (regression output)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    model = models.Sequential([
        # Input normalization
        layers.Input(shape=(input_dim,)),

        # Block 1: 512 neurons
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.001),
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Dropout(0.3),

        # Block 2: 256 neurons
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.001),
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Dropout(0.25),

        # Block 3: 128 neurons
        layers.Dense(128, kernel_regularizer=regularizers.l2(0.0008),
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Dropout(0.2),

        # Block 4: 64 neurons
        layers.Dense(64, kernel_regularizer=regularizers.l2(0.0005),
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Dropout(0.15),

        # Block 5: 32 neurons (no dropout — near output)
        layers.Dense(32, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),

        # Output: single neuron for regression
        layers.Dense(1, activation='linear')
    ])

    return model


def train_and_save():
    import tensorflow as tf
    from tensorflow.keras import callbacks, optimizers
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score
    import xgboost as xgb

    # Suppress TF info messages but keep warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

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
    # XGBOOST TRAINING
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 3: TRAINING XGBOOST (1500 trees, depth=6, lr=0.02)")
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

    # Train final XGBoost
    print(f"\n  Training final XGBoost model...")
    xgb_m = xgb.XGBRegressor(
        n_estimators=1500,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.80,
        colsample_bytree=0.70,
        min_child_weight=5,
        reg_alpha=0.05,
        reg_lambda=2.0,
        gamma=0.15,
        random_state=42,
        n_jobs=-1,
    )
    xgb_m.fit(X_train_raw, y_train,
              sample_weight=w_train,
              eval_set=[(X_val_raw, y_val)],
              verbose=False)

    xgb_val_pred = xgb_m.predict(X_val_raw)
    xgb_mae = mean_absolute_error(y_val, xgb_val_pred)
    print(f"  ✓ XGBoost Validation MAE: {xgb_mae:.2f} runs")

    # ══════════════════════════════════════════════════════════════════════
    # DEEP NEURAL NETWORK TRAINING (TensorFlow/Keras)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 4: TRAINING DEEP NEURAL NETWORK (TensorFlow/Keras)")
    print("=" * 60)
    print(f"  Architecture: Input({X_train_s.shape[1]}) → 512 → 256 → 128 → 64 → 32 → 1")
    print(f"  Optimizer: Adam (lr=0.001)")
    print(f"  Regularization: L2 + BatchNorm + Dropout")
    print(f"  Callbacks: EarlyStopping(patience=40), ReduceLROnPlateau(patience=15)")

    # Data Augmentation for DNN
    X_train_aug, y_train_aug, s_train_aug = _augment_data(
        X_train_s, y_train, seasons_train,
        n_augments=8, noise_std=0.02
    )
    w_train_aug = np.array([weight(s) for s in s_train_aug], dtype=np.float32)

    # Build model
    dnn = _build_dnn(X_train_s.shape[1])

    dnn.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='huber',  # Huber loss is more robust to outliers than MSE
        metrics=['mae']
    )

    # Print model summary
    print("\n  Model Summary:")
    dnn.summary(print_fn=lambda x: print(f"    {x}"))

    # Callbacks
    cb = [
        callbacks.EarlyStopping(
            monitor='val_mae',
            patience=40,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=15,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(MODEL_DIR, "dnn_logs"),
            histogram_freq=0,
            write_graph=False
        ),
    ]

    print(f"\n  Starting DNN training (up to 500 epochs, batch=32)...")
    print(f"  Training on {len(X_train_aug)} augmented samples")
    print(f"  Validating on {len(X_val_s)} real match samples\n")

    history = dnn.fit(
        X_train_aug, y_train_aug,
        sample_weight=w_train_aug,
        validation_data=(X_val_s, y_val),
        epochs=500,
        batch_size=32,
        callbacks=cb,
        verbose=1,  # Show epoch-by-epoch progress
    )

    # Evaluate DNN
    dnn_val_pred = dnn.predict(X_val_s, verbose=0).flatten()
    dnn_mae = mean_absolute_error(y_val, dnn_val_pred)
    best_epoch = int(np.argmin(history.history['val_mae']) + 1)
    print(f"\n  ✓ DNN Validation MAE: {dnn_mae:.2f} runs (best at epoch {best_epoch})")
    print(f"  ✓ Final LR: {float(dnn.optimizer.learning_rate):.2e}")

    # ══════════════════════════════════════════════════════════════════════
    # ENSEMBLE EVALUATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 5: ENSEMBLE EVALUATION")
    print("=" * 60)

    # Dynamic weighting based on validation performance
    total_inv_mae = (1/xgb_mae) + (1/dnn_mae)
    w_xgb = round((1/xgb_mae) / total_inv_mae, 3)
    w_dnn = round((1/dnn_mae) / total_inv_mae, 3)

    # Ensure weights sum to 1
    w_dnn = round(1.0 - w_xgb, 3)

    hybrid = w_xgb * xgb_val_pred + w_dnn * dnn_val_pred
    hybrid_mae = mean_absolute_error(y_val, hybrid)
    hybrid_r2  = r2_score(y_val, hybrid)

    print(f"\n  XGBoost MAE:   {xgb_mae:.2f} runs  (weight: {w_xgb:.1%})")
    print(f"  DNN MAE:       {dnn_mae:.2f} runs  (weight: {w_dnn:.1%})")
    print(f"  Hybrid MAE:    {hybrid_mae:.2f} runs")
    print(f"  Hybrid R²:     {hybrid_r2:.3f}  (1.0 = perfect)")
    print(f"  CV MAE:        {cv_mae:.2f} ± {cv_std:.2f} runs")

    # Sample predictions comparison
    print(f"\n  Sample Predictions (first 10 validation matches):")
    print(f"  {'Actual':>8} {'XGB':>8} {'DNN':>8} {'Hybrid':>8} {'Error':>8}")
    print(f"  {'-'*48}")
    for i in range(min(10, len(y_val))):
        h = w_xgb * xgb_val_pred[i] + w_dnn * dnn_val_pred[i]
        err = abs(y_val[i] - h)
        print(f"  {y_val[i]:8.0f} {xgb_val_pred[i]:8.1f} {dnn_val_pred[i]:8.1f} {h:8.1f} {err:8.1f}")

    # ── SAVE ALL MODELS ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 6: SAVING MODELS")
    print("=" * 60)

    with open(XGB_PATH, "wb") as f:
        pickle.dump(xgb_m, f)
    print(f"  ✓ XGBoost saved → {XGB_PATH}")

    dnn.save(DNN_PATH)
    print(f"  ✓ DNN (Keras) saved → {DNN_PATH}")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(sc, f)
    print(f"  ✓ Scaler saved → {SCALER_PATH}")

    with open(WEIGHTS_PATH, "wb") as f:
        pickle.dump({"xgb": w_xgb, "dnn": w_dnn}, f)
    print(f"  ✓ Ensemble weights saved → {WEIGHTS_PATH}")

    # Save training history
    hist_path = os.path.join(MODEL_DIR, "training_history.json")
    hist_data = {
        "epochs_trained": int(len(history.history['loss'])),
        "best_epoch": int(best_epoch),
        "xgb_mae": float(round(xgb_mae, 2)),
        "dnn_mae": float(round(dnn_mae, 2)),
        "hybrid_mae": float(round(hybrid_mae, 2)),
        "hybrid_r2": float(round(hybrid_r2, 4)),
        "cv_mae": float(round(cv_mae_py, 2)),
        "ensemble_weights": {"xgb": float(w_xgb), "dnn": float(w_dnn)},
        "val_mae_history": [float(round(float(v), 3)) for v in history.history['val_mae']],
    }
    with open(hist_path, "w") as f:
        json.dump(hist_data, f, indent=2)
    print(f"  ✓ Training history saved → {hist_path}")

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE!")
    print(f"  Ensemble: XGBoost ({w_xgb:.0%}) + DNN ({w_dnn:.0%})")
    print(f"  Final Hybrid MAE: {hybrid_mae:.2f} runs")
    print(f"{'=' * 60}\n")

    return xgb_m, dnn, sc


def load_models():
    global xgb_model, dnn_model, scaler

    if os.path.exists(XGB_PATH) and os.path.exists(DNN_PATH) and os.path.exists(SCALER_PATH):
        with open(XGB_PATH, "rb") as f:
            xgb_model = pickle.load(f)

        import tensorflow as tf
        dnn_model = tf.keras.models.load_model(DNN_PATH)

        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        print("✓ All models loaded from disk (XGBoost + DNN + Scaler)")
    else:
        print("No saved models found. Training from scratch...")
        xgb_model, dnn_model, scaler = train_and_save()

    return xgb_model, dnn_model, scaler


def predict_score(features: np.ndarray):
    global xgb_model, dnn_model, scaler
    if xgb_model is None:
        load_models()

    feat_2d  = features.reshape(1, -1)
    feat_sc  = scaler.transform(feat_2d)

    # XGBoost prediction (uses raw features)
    xgb_pred = float(xgb_model.predict(feat_2d)[0])

    # DNN prediction (uses scaled features)
    dnn_pred = float(dnn_model.predict(feat_sc, verbose=0)[0][0])

    # Load ensemble weights
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH, "rb") as f:
            w = pickle.load(f)
        hybrid = round(w["xgb"] * xgb_pred + w["dnn"] * dnn_pred, 1)
    else:
        # Fallback: favor XGBoost
        hybrid = round(0.75 * xgb_pred + 0.25 * dnn_pred, 1)

    return hybrid, xgb_pred, dnn_pred
