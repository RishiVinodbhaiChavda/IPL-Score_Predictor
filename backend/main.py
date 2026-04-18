from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import teams, players, predict

app = FastAPI(title="IPL Score Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(teams.router,   prefix="/api")
app.include_router(players.router, prefix="/api")
app.include_router(predict.router, prefix="/api")

@app.get("/")
def root():
    return {"status": "IPL Score Predictor API running"}
