from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from ml.predict_service import run_prediction

router = APIRouter()

class PredictRequest(BaseModel):
    batting_team_id:  str
    bowling_team_id:  str
    batting_xi:       List[str]
    bowling_xi:       List[str]
    venue_id:         str
    pitch_type:       Optional[str] = "Balanced"
    temperature:      Optional[float] = 30.0
    humidity:         Optional[float] = 60.0
    dew_factor:       Optional[float] = 3.0
    season:           Optional[int] = 2025

@router.post("/predict")
def predict(req: PredictRequest):
    return run_prediction(req.dict())
