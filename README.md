# IPL Score Predictor 🏏

AI-powered first innings score prediction using **Hybrid XGBoost + MLP Neural Network Ensemble**.

## 🎯 Features

- **Hybrid ML Model**: XGBoost (70%) + MLP Neural Network (30%) ensemble
- **Validation MAE**: 22.63 runs (Hybrid), 15.68 runs (5-Fold CV)
- **53 Advanced Features**: Player vs Player matchups, recent form, venue stats, phase stats, weather
- **Real-time Predictions**: Fast API responses (~200ms)
- **Interactive UI**: Team selection → Squad selection → Venue → Weather → Prediction
- **Comprehensive Dataset**: 711 matches (2015-2025), 244 players, 13 CSV files

## 📊 Model Performance

| Model | Validation MAE | Weight |
|-------|---------------|--------|
| XGBoost | 19.75 runs | 70% |
| MLP Neural Network | 45.33 runs | 30% |
| **Hybrid Ensemble** | **22.63 runs** | 100% |
| 5-Fold CV | 15.68 ± 0.80 runs | - |

**Training Graph**: See `models/training_validation_loss.png` for detailed training curves.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### 1. Clone the repository
```bash
git clone https://github.com/RishiVinodbhaiChavda/IPL-Score_Predictor.git
cd IPL-Score_Predictor
```

### 2. Setup Dataset folder
Ensure the `Dataset/` folder is in the parent directory with these CSV files:
- `players.csv`, `teams.csv`, `venues.csv`, `squads.csv`, `matches.csv`
- `player_vs_player.csv`, `player_batting_vs_type.csv`, `player_vs_team.csv`
- `player_venue_stats.csv`, `player_phase_stats.csv`, `player_recent_form.csv`
- `player_bowling_overall.csv`, `player_transfers.csv`, `match_training_data.csv`

### 3. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Train the model (run once)
```bash
python train_model.py
```
This will create the `models/` folder with trained models (~5-10 minutes).

### 5. Start the backend
```bash
uvicorn main:app --reload --port 8001
```
Backend will run at `http://localhost:8001`

### 6. Start the frontend
Open `frontend/index.html` in your browser or use Live Server:
```bash
# Using Python's built-in server
cd frontend
python -m http.server 8000
```
Frontend will run at `http://localhost:8000`

## 📁 Project Structure
```
IPL-Score-Predictor/
├── backend/
│   ├── main.py                      # FastAPI app (port 8001)
│   ├── train_model.py               # Train & save models
│   ├── requirements.txt             # Python dependencies
│   ├── weather_service.py           # Weather API integration
│   ├── WEATHER_API_SETUP.md         # Weather API setup guide
│   ├── routes/
│   │   ├── teams.py                 # GET /api/teams
│   │   ├── players.py               # GET /api/players?team_id=
│   │   ├── predict.py               # POST /api/predict
│   │   └── weather.py               # GET /api/weather
│   ├── ml/
│   │   ├── feature_engineering.py  # Build 53 feature vectors
│   │   ├── edge_case_handler.py    # Handle new/transferred players
│   │   ├── model_loader.py         # XGBoost + MLP hybrid training
│   │   └── predict_service.py      # Orchestrate prediction
│   └── db/
│       └── data_loader.py           # Load CSVs with lazy loading
├── frontend/
│   ├── index.html                   # Team selection page
│   ├── squad.html                   # Playing XI selection
│   ├── venue.html                   # Venue selection
│   ├── summary.html                 # Match summary & weather input
│   ├── result.html                  # Prediction result display
│   ├── conditions.html              # Match conditions (legacy)
│   ├── debug.html                   # Debug page
│   ├── css/style.css                # Unified styles
│   ├── js/
│   │   ├── api.js                   # API calls
│   │   └── state.js                 # localStorage state management
│   └── assets/                      # Venue images (25 stadiums)
├── models/                          # Auto-created after training
│   ├── xgb_model.pkl                # XGBoost model
│   ├── mlp_model.pkl                # MLP Neural Network
│   ├── scaler.pkl                   # Feature scaler
│   ├── ensemble_weights.pkl         # Ensemble weights
│   ├── training_history.json        # Training metrics
│   ├── training_validation_loss.png # Training graph (PNG)
│   └── training_validation_loss.pdf # Training graph (PDF)
├── Dataset/                         # CSV data files (parent dir)
│   ├── players.csv                  # 244 players
│   ├── teams.csv                    # 10 IPL teams
│   ├── venues.csv                   # 25 venues
│   ├── squads.csv                   # 2015-2026 squads
│   ├── matches.csv                  # 711 matches
│   ├── player_vs_player.csv         # 5439 matchups
│   ├── player_batting_vs_type.csv   # Pace/Spin stats
│   ├── player_vs_team.csv           # Player vs team records
│   ├── player_venue_stats.csv       # Venue-specific stats
│   ├── player_phase_stats.csv       # Powerplay/Middle/Death
│   ├── player_recent_form.csv       # Form classification
│   ├── player_bowling_overall.csv   # Bowling stats
│   ├── player_transfers.csv         # Transfer history
│   └── match_training_data.csv      # Playing 11s
├── ARCHITECTURE.md                  # System architecture (Mermaid)
├── ARCHITECTURE_VISUAL.txt          # ASCII architecture diagram
├── generate_architecture_diagram.py # Python diagram generator
├── generate_training_graph.py       # Training graph generator
└── README.md                        # This file
```

## 🔌 API Endpoints

| Method | Endpoint | Description | Response Time |
|--------|----------|-------------|---------------|
| GET | `/api/teams` | All active IPL teams with logos | ~29ms |
| GET | `/api/venues` | All venues with pitch types & avg scores | ~25ms |
| GET | `/api/players?team_id={id}&season={year}` | Squad for a team | ~50ms |
| GET | `/api/weather?venue_id={id}` | Weather data for venue | ~14ms |
| POST | `/api/predict` | Predict first innings score | ~200ms |

### Prediction Request
```json
{
  "batting_team_id": "IPL_TEAM_01",
  "bowling_team_id": "IPL_TEAM_02",
  "venue_id": "IPL_VEN_01",
  "batting_xi": ["IPL_PLAYER_001", "IPL_PLAYER_002", ...],
  "bowling_xi": ["IPL_PLAYER_101", "IPL_PLAYER_102", ...],
  "pitch_type": "Batting",
  "temperature": 32.5,
  "humidity": 65,
  "dew_factor": 5
}
```

## 📈 Prediction Response
```json
{
  "score": 178.5,
  "range": [163, 193],
  "confidence": 78,
  "xgb_score": 181.2,
  "mlp_score": 174.3,
  "factors": [
    {
      "feature": "batting_team_avg_at_venue",
      "label": "Batting team's record at this venue",
      "value": 182.4
    },
    {
      "feature": "bowl_venue_econ",
      "label": "Bowling team's economy at this venue",
      "value": 8.7
    },
    {
      "feature": "bat_form_score",
      "label": "Batting team's current form",
      "value": 4.2
    }
  ],
  "simulations": [175.1, 177.8, 178.5, 180.2, 176.9]
}
```

## 🧠 Machine Learning Pipeline

### Feature Engineering (53 Features)
1. **Player vs Player Matchups** (12 features)
   - Batting avg, strike rate, dismissals vs specific bowlers
   - Bowling avg, economy, wickets vs specific batters

2. **Recent Form** (8 features)
   - Last 5 matches performance
   - Form classification (Excellent/Good/Average/Poor)

3. **Venue Statistics** (10 features)
   - Team averages at venue
   - Player performance at venue
   - Pitch type impact

4. **Phase Statistics** (12 features)
   - Powerplay (1-6 overs)
   - Middle overs (7-15)
   - Death overs (16-20)

5. **Weather Conditions** (3 features)
   - Temperature, Humidity, Dew Factor

6. **Team Composition** (8 features)
   - Batting depth, bowling variety
   - Pace vs Spin balance

### Training Strategy
- **Time-based split**: Train on 2015-2024, validate on 2025-2026 (no data leakage)
- **Sample weighting**: Recent seasons weighted 3x higher
- **5-Fold Cross-Validation** on training data
- **Early stopping**: Patience = 30 epochs for MLP
- **Output clipping**: 80-280 runs (realistic T20 range)

### Model Architecture
**XGBoost**:
- 2000 trees, max_depth=4, learning_rate=0.01
- Subsample=0.80, colsample_bytree=0.70
- L1 regularization=0.15, L2 regularization=2.0

**MLP Neural Network**:
- Architecture: Input(53) → 256 → 128 → 64 → 1
- Activation: ReLU
- Optimizer: Adam (lr=0.0005)
- L2 regularization: 0.01
- Early stopping with 15% validation split

**Ensemble**:
- Dynamic weighting based on validation MAE
- XGBoost: 70%, MLP: 30%

## 📊 Visualizations

### Training Graphs
Run `python generate_training_graph.py` to create:
- MLP training vs validation loss curves
- XGBoost training progression
- Model performance comparison
- Training summary table

Output: `models/training_validation_loss.png` (4800×3000 @ 300 DPI)

### Architecture Diagrams
1. **Mermaid Diagram**: `ARCHITECTURE.md` (export to PNG via https://mermaid.live)
2. **ASCII Diagram**: `ARCHITECTURE_VISUAL.txt` (screenshot-ready)
3. **Python Diagram**: Run `python generate_architecture_diagram.py` (requires `pip install diagrams`)

## 🛠️ Technologies Used

**Backend**:
- FastAPI (REST API)
- XGBoost (Gradient Boosting)
- scikit-learn (MLP, preprocessing)
- pandas (data processing)
- NumPy (numerical operations)

**Frontend**:
- Vanilla JavaScript (no frameworks)
- LocalStorage (state management)
- Fetch API (HTTP requests)
- CSS Grid/Flexbox (responsive layout)

**Data**:
- 13 CSV files with comprehensive IPL statistics
- 711 matches (2015-2025)
- 244 players with detailed stats

## 🎨 UI Flow

1. **Team Selection** (`index.html`)
   - Select batting and bowling teams
   - View team logos and names

2. **Squad Selection** (`squad.html`)
   - Choose 11 players from each team
   - View player photos and roles
   - Automatic role-based filtering

3. **Venue Selection** (`venue.html`)
   - Select stadium from 25 venues
   - View venue images
   - See pitch type and average scores

4. **Match Summary** (`summary.html`)
   - Review selected teams and players
   - Input weather conditions (manual)
   - Ultra-compact player cards (2-column layout)

5. **Prediction Result** (`result.html`)
   - View predicted score with confidence
   - See XGBoost and MLP individual predictions
   - Analyze top contributing factors
   - View score range and simulations

## 🔧 Configuration

### Backend Port
Default: `8001` (configurable in `main.py`)

### Weather API
Optional: Configure in `backend/weather_service.py`
See `backend/WEATHER_API_SETUP.md` for details

### Dataset Location
Expected: `../Dataset/` (parent directory)
Configurable in `backend/db/data_loader.py`

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Rishi Vinodbhai Chavda**
- GitHub: [@RishiVinodbhaiChavda](https://github.com/RishiVinodbhaiChavda)
- Repository: [IPL-Score_Predictor](https://github.com/RishiVinodbhaiChavda/IPL-Score_Predictor)

## 🙏 Acknowledgments

- IPL official website for player images
- Cricket statistics databases
- Open source ML libraries (XGBoost, scikit-learn)

---

**Note**: This project is for educational purposes. Predictions are based on historical data and may not reflect actual match outcomes.
