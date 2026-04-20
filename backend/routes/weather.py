"""
Weather API Routes
==================
Provides real-time weather data for cricket venues.
"""
from fastapi import APIRouter, HTTPException
from weather_service import get_match_time_weather

router = APIRouter()

@router.get("/weather/{venue_id}")
async def get_venue_weather(venue_id: str, match_hour: int = 19):
    """
    Get current weather conditions for a venue.
    
    Args:
        venue_id: IPL venue ID (e.g., IPL_VEN_13)
        match_hour: Match start time in IST (default: 19 for 7:30 PM)
    
    Returns:
        Weather data including temperature, humidity, dew factor
    """
    weather_data = get_match_time_weather(venue_id, match_hour)
    
    if not weather_data:
        raise HTTPException(
            status_code=404, 
            detail=f"Weather data not available for venue {venue_id}"
        )
    
    return {
        "venue_id": venue_id,
        "weather": weather_data,
        "status": "success"
    }

@router.get("/weather")
async def get_weather_info():
    """Get information about weather service."""
    return {
        "service": "IPL Weather Service",
        "provider": "OpenWeatherMap",
        "coverage": "26 IPL venues",
        "update_frequency": "Real-time",
        "match_times": "Adjusted for evening matches (7:30 PM IST)"
    }