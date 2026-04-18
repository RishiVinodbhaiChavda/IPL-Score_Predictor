# IPL Score Predictor

AI-powered first innings score prediction using XGBoost + Neural Network.

## Quick Start

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the model (run once)
```bash
cd backend
python train_model.py
```

### 3. Start the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Open the frontend
Open `frontend/index.html` in your browser (or use Live Server in VS Code).

## Project Structure
```
IPL-Score-Predictor/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── train_model.py       # Train & save models
│   ├── requirements.txt
│   ├── routes/
│   │   ├── teams.py         # GET /api/teams, /api/venues
│   │   ├── players.py       # GET /api/players?team_id=
│   │   └── predict.py       # POST /api/predict
│   ├── ml/
│   │   ├── feature_engineering.py  # Build feature vectors
│   │   ├── edge_case_handler.py    # Handle new/transferred players
│   │   ├── model_loader.py         # XGBoost + NN hybrid
│   │   └── predict_service.py      # Orchestrate prediction
│   └── db/
│       └── data_loader.py   # Load all CSVs into memory
├── frontend/
│   ├── index.html           # Team selection
│   ├── squad.html           # Playing XI selection
│   ├── conditions.html      # Match conditions
│   ├── result.html          # Prediction result
│   ├── css/style.css
│   └── js/
│       ├── api.js           # API calls
│       └── state.js         # localStorage state
└── models/                  # Saved model files (auto-created)
```

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/teams | All active IPL teams |
| GET | /api/venues | All venues with avg scores |
| GET | /api/players?team_id= | Squad for a team |
| POST | /api/predict | Predict first innings score |

## Prediction Response
```json
{
  "score": 178.5,
  "range": [163, 193],
  "confidence": 78,
  "xgb_score": 181.2,
  "nn_score": 174.3,
  "factors": [
    {"feature": "batting_team_avg_at_venue", "label": "Batting team's record at this venue", "value": 182.4},
    {"feature": "bowl_venue_econ", "label": "Bowling team's economy at this venue", "value": 8.7},
    {"feature": "bat_form_score", "label": "Batting team's current form", "value": 4.2}
  ],
  "simulations": [175.1, 177.8, 178.5, 180.2, 176.9]
}
```
