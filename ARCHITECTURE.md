# IPL Score Predictor - System Architecture

## High-Level Architecture Diagram

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Web Interface<br/>HTML/CSS/JavaScript]
        UI --> |1. Team Selection| TS[Team Selection Page]
        UI --> |2. Squad Selection| SS[Squad Selection Page]
        UI --> |3. Venue Selection| VS[Venue Selection Page]
        UI --> |4. Match Summary| MS[Summary Page]
        UI --> |5. Prediction Result| RS[Result Page]
        
        LS[LocalStorage<br/>State Management]
        TS -.-> LS
        SS -.-> LS
        VS -.-> LS
        MS -.-> LS
    end
    
    subgraph "Backend Layer - FastAPI"
        API[REST API<br/>Port 8001]
        
        subgraph "API Routes"
            TR[Teams Route<br/>/api/teams]
            PR[Players Route<br/>/api/players]
            VR[Venues Route<br/>/api/venues]
            WR[Weather Route<br/>/api/weather]
            PDR[Predict Route<br/>/api/predict]
        end
        
        API --> TR
        API --> PR
        API --> VR
        API --> WR
        API --> PDR
    end
    
    subgraph "Data Layer"
        DL[Data Loader<br/>Lazy Loading + Caching]
        
        subgraph "CSV Datasets"
            D1[(players.csv<br/>244 players)]
            D2[(squads.csv<br/>675 records)]
            D3[(matches.csv<br/>711 matches)]
            D4[(player_vs_player.csv<br/>5439 matchups)]
            D5[(player_recent_form.csv<br/>512 records)]
            D6[(venues.csv<br/>26 stadiums)]
            D7[(player_phase_stats.csv<br/>4820 records)]
            D8[(Other Stats<br/>8+ files)]
        end
        
        DL --> D1
        DL --> D2
        DL --> D3
        DL --> D4
        DL --> D5
        DL --> D6
        DL --> D7
        DL --> D8
    end
    
    subgraph "ML Layer"
        FE[Feature Engineering<br/>53 Features]
        
        subgraph "Hybrid Ensemble Model"
            XGB[XGBoost Model<br/>70% Weight<br/>MAE: 19.75]
            MLP[Neural Network<br/>30% Weight<br/>256→128→64→1]
        end
        
        ENS[Ensemble Predictor<br/>MAE: 22.63 runs]
        
        FE --> XGB
        FE --> MLP
        XGB --> ENS
        MLP --> ENS
    end
    
    subgraph "External Services"
        WS[OpenWeatherMap API<br/>Weather Data]
    end
    
    %% Connections
    UI -->|HTTP Requests| API
    TR --> DL
    PR --> DL
    VR --> DL
    WR --> WS
    PDR --> FE
    FE --> DL
    ENS --> PDR
    
    %% Styling
    classDef frontend fill:#ff6b35,stroke:#fff,stroke-width:2px,color:#fff
    classDef backend fill:#7c3aed,stroke:#fff,stroke-width:2px,color:#fff
    classDef data fill:#00e5ff,stroke:#fff,stroke-width:2px,color:#000
    classDef ml fill:#4ade80,stroke:#fff,stroke-width:2px,color:#000
    classDef external fill:#fbbf24,stroke:#fff,stroke-width:2px,color:#000
    
    class UI,TS,SS,VS,MS,RS,LS frontend
    class API,TR,PR,VR,WR,PDR backend
    class DL,D1,D2,D3,D4,D5,D6,D7,D8 data
    class FE,XGB,MLP,ENS ml
    class WS external
```

## Component Details

### 1. Frontend Layer (Port 8000)
- **Technology**: HTML5, CSS3, Vanilla JavaScript
- **Pages**: 
  - Team Selection → Squad Selection → Venue Selection → Summary → Result
- **State Management**: LocalStorage for caching selected data
- **Features**:
  - Instant page loading with cached data
  - Responsive design with animations
  - Manual weather input

### 2. Backend Layer (Port 8001)
- **Technology**: FastAPI (Python)
- **API Endpoints**:
  - `GET /api/teams` - List all IPL teams
  - `GET /api/players?team_id={id}` - Get squad for a team
  - `GET /api/venues` - List all stadiums
  - `GET /api/weather/{venue_id}` - Get weather data
  - `POST /api/predict` - Generate score prediction

### 3. Data Layer
- **Storage**: CSV files (15+ datasets)
- **Loading Strategy**: Lazy loading with in-memory caching
- **Key Datasets**:
  - Player master data (244 players)
  - Match history (2015-2026)
  - Player vs Player matchups
  - Venue statistics
  - Recent form data

### 4. ML Layer
- **Feature Engineering**: 53 features including:
  - Player vs Player matchups
  - Recent form (last 5 matches)
  - Venue-specific stats
  - Phase stats (Powerplay/Middle/Death)
  - Weather conditions
  - Pitch type

- **Hybrid Ensemble Model**:
  - **XGBoost**: 2000 trees, depth=4, lr=0.01 (70% weight)
  - **Neural Network**: 256→128→64→1 architecture (30% weight)
  - **Validation MAE**: 22.63 runs
  - **Output Range**: 80-280 runs (realistic T20 scores)

### 5. External Services
- **OpenWeatherMap API**: Real-time weather data
- **Fallback**: Manual input with default values

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant LocalStorage
    participant Backend
    participant DataLoader
    participant MLModel
    
    User->>Frontend: Select Teams
    Frontend->>Backend: GET /api/teams
    Backend->>DataLoader: Load teams.csv
    DataLoader-->>Backend: Team data
    Backend-->>Frontend: Team list
    Frontend->>LocalStorage: Save team data
    
    User->>Frontend: Select 11 Players
    Frontend->>Backend: GET /api/players?team_id=X
    Backend->>DataLoader: Load squads.csv + players.csv
    DataLoader-->>Backend: Player data
    Backend-->>Frontend: Player list
    Frontend->>LocalStorage: Save player data
    
    User->>Frontend: Select Venue
    Frontend->>Backend: GET /api/venues
    Backend->>DataLoader: Load venues.csv
    DataLoader-->>Backend: Venue data
    Backend-->>Frontend: Venue list
    Frontend->>LocalStorage: Save venue data
    
    User->>Frontend: Review Summary
    Frontend->>LocalStorage: Load cached data
    LocalStorage-->>Frontend: Teams, Players, Venue
    Frontend-->>User: Display summary (instant)
    
    User->>Frontend: Click Predict
    Frontend->>Backend: POST /api/predict
    Backend->>DataLoader: Load all stats
    DataLoader-->>Backend: Feature data
    Backend->>MLModel: Build 53 features
    MLModel->>MLModel: XGBoost prediction
    MLModel->>MLModel: Neural Network prediction
    MLModel->>MLModel: Ensemble (70% + 30%)
    MLModel-->>Backend: Final score
    Backend-->>Frontend: Prediction result
    Frontend-->>User: Display score + analysis
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | HTML/CSS/JavaScript | User interface |
| **Backend** | FastAPI (Python) | REST API server |
| **ML Framework** | XGBoost + TensorFlow | Prediction models |
| **Data Processing** | Pandas + NumPy | Data manipulation |
| **Storage** | CSV files | Dataset storage |
| **Caching** | LocalStorage (Frontend) | State management |
| **Caching** | In-memory (Backend) | Data loader cache |
| **External API** | OpenWeatherMap | Weather data |

## Performance Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| **Backend Startup** | Time | ~3 seconds |
| **Teams API** | Response Time | ~29ms |
| **Players API** | Response Time | ~50ms |
| **Weather API** | Timeout | 1 second |
| **Prediction API** | Response Time | ~200ms |
| **Frontend Load** | Summary Page | Instant (cached) |
| **Model Accuracy** | MAE | 22.63 runs |
| **Model Accuracy** | 5-Fold CV | 15.68 ± 0.80 runs |

## Deployment Architecture

```
┌─────────────────────────────────────────┐
│         User's Browser                   │
│  http://localhost:8000                   │
└─────────────┬───────────────────────────┘
              │
              │ HTTP Requests
              │
┌─────────────▼───────────────────────────┐
│      Frontend Server (Port 8000)        │
│      Python HTTP Server                 │
└─────────────┬───────────────────────────┘
              │
              │ REST API Calls
              │
┌─────────────▼───────────────────────────┐
│      Backend Server (Port 8001)         │
│      FastAPI + Uvicorn                  │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │   ML Models (Loaded in Memory)     │ │
│  │   - XGBoost Model (xgb_model.pkl)  │ │
│  │   - MLP Model (mlp_model.pkl)      │ │
│  │   - Scaler (scaler.pkl)            │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │   Data Cache (In-Memory)           │ │
│  │   - Essential CSVs loaded          │ │
│  │   - Lazy loading for others        │ │
│  └────────────────────────────────────┘ │
└─────────────┬───────────────────────────┘
              │
              │ File System Access
              │
┌─────────────▼───────────────────────────┐
│         Dataset Folder                   │
│         15+ CSV Files                    │
│         Models Folder                    │
│         4 Model Files                    │
└──────────────────────────────────────────┘
```

## Key Features

1. **Lazy Loading**: Backend loads only essential data on startup, other data loaded on-demand
2. **Caching**: Frontend caches selected data in LocalStorage for instant summary page
3. **Hybrid Model**: Combines XGBoost (accuracy) + Neural Network (generalization)
4. **Feature Engineering**: 53 features from 10+ data sources
5. **Real-time Weather**: Optional API integration with manual fallback
6. **Responsive UI**: Clean, modern interface with animations
7. **Fast Predictions**: ~200ms response time for predictions

---

**Created**: 2026  
**Version**: 1.0.0  
**Model Training Data**: IPL 2015-2026
