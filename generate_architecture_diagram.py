"""
Generate Architecture Diagram Image
Install required package: pip install diagrams
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.client import Users
from diagrams.programming.framework import Fastapi
from diagrams.programming.language import Python, JavaScript
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.inmemory import Redis
from diagrams.custom import Custom

# Create architecture diagram
with Diagram("IPL Score Predictor - System Architecture", 
             filename="architecture_diagram",
             show=False,
             direction="TB",
             graph_attr={"bgcolor": "transparent", "fontsize": "14"}):
    
    user = Users("User Browser")
    
    with Cluster("Frontend Layer (Port 8000)"):
        frontend = JavaScript("Web Interface\nHTML/CSS/JS")
        pages = [
            JavaScript("Team Selection"),
            JavaScript("Squad Selection"),
            JavaScript("Venue Selection"),
            JavaScript("Summary Page"),
            JavaScript("Result Page")
        ]
        storage = Redis("LocalStorage\nState Cache")
    
    with Cluster("Backend Layer (Port 8001)"):
        api = Fastapi("FastAPI Server")
        
        with Cluster("API Routes"):
            teams_route = Python("/api/teams")
            players_route = Python("/api/players")
            venues_route = Python("/api/venues")
            weather_route = Python("/api/weather")
            predict_route = Python("/api/predict")
    
    with Cluster("Data Layer"):
        data_loader = Python("Data Loader\nLazy Loading + Cache")
        
        with Cluster("CSV Datasets (15+ files)"):
            csv1 = PostgreSQL("players.csv\n244 players")
            csv2 = PostgreSQL("matches.csv\n711 matches")
            csv3 = PostgreSQL("player_vs_player.csv\n5439 matchups")
            csv4 = PostgreSQL("venues.csv\n26 stadiums")
    
    with Cluster("ML Layer"):
        feature_eng = Python("Feature Engineering\n53 Features")
        
        with Cluster("Hybrid Ensemble Model"):
            xgboost = Python("XGBoost\n70% Weight\nMAE: 19.75")
            neural_net = Python("Neural Network\n30% Weight\n256→128→64→1")
        
        ensemble = Python("Ensemble Predictor\nMAE: 22.63 runs")
    
    # Connections
    user >> Edge(label="HTTP") >> frontend
    frontend >> Edge(label="Cache") >> storage
    frontend >> Edge(label="REST API") >> api
    
    api >> teams_route
    api >> players_route
    api >> venues_route
    api >> weather_route
    api >> predict_route
    
    teams_route >> data_loader
    players_route >> data_loader
    venues_route >> data_loader
    predict_route >> feature_eng
    
    data_loader >> csv1
    data_loader >> csv2
    data_loader >> csv3
    data_loader >> csv4
    
    feature_eng >> xgboost
    feature_eng >> neural_net
    xgboost >> ensemble
    neural_net >> ensemble
    ensemble >> predict_route

print("✅ Architecture diagram generated: architecture_diagram.png")
print("📁 Location: IPL-Score-Predictor/architecture_diagram.png")
