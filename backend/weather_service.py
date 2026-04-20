"""
Weather Service for IPL Score Predictor
=======================================
Fetches real-time weather data for cricket venues using OpenWeatherMap API.
Provides temperature, humidity, and dew point for match conditions.
"""
import requests
import json
from typing import Dict, Optional
from datetime import datetime, timedelta

# Simple in-memory cache
_weather_cache = {}
_cache_ttl = 300  # 5 minutes

# City coordinates for IPL venues
VENUE_COORDINATES = {
    "IPL_VEN_01": {"city": "Visakhapatnam", "lat": 17.7231, "lon": 83.3012},
    "IPL_VEN_02": {"city": "Delhi", "lat": 28.6139, "lon": 77.2090},
    "IPL_VEN_03": {"city": "Guwahati", "lat": 26.1445, "lon": 91.7362},
    "IPL_VEN_04": {"city": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    "IPL_VEN_05": {"city": "Navi Mumbai", "lat": 19.0330, "lon": 73.0297},
    "IPL_VEN_06": {"city": "Dubai", "lat": 25.2048, "lon": 55.2708},
    "IPL_VEN_07": {"city": "Kolkata", "lat": 22.5726, "lon": 88.3639},
    "IPL_VEN_08": {"city": "Lucknow", "lat": 26.8467, "lon": 80.9462},
    "IPL_VEN_09": {"city": "Kanpur", "lat": 26.4499, "lon": 80.3319},
    "IPL_VEN_10": {"city": "Dharamsala", "lat": 32.2190, "lon": 76.3234},
    "IPL_VEN_11": {"city": "Indore", "lat": 22.7196, "lon": 75.8577},
    "IPL_VEN_12": {"city": "Ranchi", "lat": 23.3441, "lon": 85.3096},
    "IPL_VEN_13": {"city": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
    "IPL_VEN_14": {"city": "Chennai", "lat": 13.0827, "lon": 80.2707},
    "IPL_VEN_15": {"city": "Mullanpur", "lat": 30.7046, "lon": 76.7179},
    "IPL_VEN_16": {"city": "Pune", "lat": 18.5204, "lon": 73.8567},
    "IPL_VEN_17": {"city": "Ahmedabad", "lat": 23.0225, "lon": 72.5714},
    "IPL_VEN_18": {"city": "Mohali", "lat": 30.6942, "lon": 76.7344},
    "IPL_VEN_19": {"city": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
    "IPL_VEN_20": {"city": "Rajkot", "lat": 22.3039, "lon": 70.8022},
    "IPL_VEN_21": {"city": "Jaipur", "lat": 26.9124, "lon": 75.7873},
    "IPL_VEN_22": {"city": "Raipur", "lat": 21.2514, "lon": 81.6296},
    "IPL_VEN_23": {"city": "Sharjah", "lat": 25.3463, "lon": 55.4209},
    "IPL_VEN_24": {"city": "Abu Dhabi", "lat": 24.4539, "lon": 54.3773},
    "IPL_VEN_25": {"city": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    "IPL_VEN_26": {"city": "Abu Dhabi", "lat": 24.4539, "lon": 54.3773},
}

# OpenWeatherMap API key (free tier - 1000 calls/day)
API_KEY = "your_openweather_api_key_here"

def calculate_dew_factor(temp_celsius: float, humidity: float) -> float:
    """Calculate dew factor (0-10 scale) based on temperature and humidity."""
    a, b = 17.27, 237.7
    alpha = ((a * temp_celsius) / (b + temp_celsius)) + (humidity / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    
    if dew_point >= 20:
        return min(10.0, 5.0 + (dew_point - 20) * 0.5)
    elif dew_point >= 15:
        return 3.0 + (dew_point - 15) * 0.4
    elif dew_point >= 10:
        return 1.0 + (dew_point - 10) * 0.4
    else:
        return max(0.0, dew_point * 0.1)

def get_weather_data(venue_id: str) -> Optional[Dict]:
    """Fetch current weather data for a venue with caching."""
    
    # Check cache first
    if venue_id in _weather_cache:
        cached_data, timestamp = _weather_cache[venue_id]
        if datetime.now() - timestamp < timedelta(seconds=_cache_ttl):
            return cached_data
    
    if venue_id not in VENUE_COORDINATES:
        return None
    
    if API_KEY == "your_openweather_api_key_here":
        # Return realistic default values if no API key
        coord = VENUE_COORDINATES[venue_id]
        city = coord["city"]
        
        defaults = {
            "Dubai": {"temp": 32, "humidity": 45},
            "Abu Dhabi": {"temp": 33, "humidity": 50},
            "Sharjah": {"temp": 31, "humidity": 48},
            "Mumbai": {"temp": 29, "humidity": 75},
            "Chennai": {"temp": 31, "humidity": 70},
            "Kolkata": {"temp": 28, "humidity": 80},
            "Delhi": {"temp": 25, "humidity": 65},
            "Bengaluru": {"temp": 26, "humidity": 60},
            "Hyderabad": {"temp": 30, "humidity": 55},
            "Pune": {"temp": 27, "humidity": 65},
            "Jaipur": {"temp": 28, "humidity": 50},
            "Ahmedabad": {"temp": 32, "humidity": 45},
            "Lucknow": {"temp": 26, "humidity": 70},
            "Mohali": {"temp": 24, "humidity": 60},
        }
        
        default = defaults.get(city, {"temp": 28, "humidity": 65})
        temp = default["temp"]
        humidity = default["humidity"]
        dew_factor = calculate_dew_factor(temp, humidity)
        
        result = {
            "temperature": temp,
            "humidity": humidity,
            "dew_factor": round(dew_factor, 1),
            "city": city,
            "source": "default",
            "description": "Default conditions"
        }
        
        # Cache the result
        _weather_cache[venue_id] = (result, datetime.now())
        return result
    
    try:
        coord = VENUE_COORDINATES[venue_id]
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": coord["lat"],
            "lon": coord["lon"],
            "appid": API_KEY,
            "units": "metric"
        }
        
        response = requests.get(url, params=params, timeout=2)
        response.raise_for_status()
        data = response.json()
        
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        dew_factor = calculate_dew_factor(temp, humidity)
        
        result = {
            "temperature": round(temp, 1),
            "humidity": humidity,
            "dew_factor": round(dew_factor, 1),
            "city": coord["city"],
            "source": "openweather",
            "description": data["weather"][0]["description"].title(),
        }
        
        # Cache the result
        _weather_cache[venue_id] = (result, datetime.now())
        return result
        
    except Exception as e:
        print(f"Weather API error for {venue_id}: {e}")
        # Return defaults on error
        coord = VENUE_COORDINATES[venue_id]
        city = coord["city"]
        defaults = {
            "Dubai": {"temp": 32, "humidity": 45},
            "Abu Dhabi": {"temp": 33, "humidity": 50},
            "Sharjah": {"temp": 31, "humidity": 48},
            "Mumbai": {"temp": 29, "humidity": 75},
            "Chennai": {"temp": 31, "humidity": 70},
            "Kolkata": {"temp": 28, "humidity": 80},
            "Delhi": {"temp": 25, "humidity": 65},
            "Bengaluru": {"temp": 26, "humidity": 60},
            "Hyderabad": {"temp": 30, "humidity": 55},
            "Pune": {"temp": 27, "humidity": 65},
            "Jaipur": {"temp": 28, "humidity": 50},
            "Ahmedabad": {"temp": 32, "humidity": 45},
            "Lucknow": {"temp": 26, "humidity": 70},
            "Mohali": {"temp": 24, "humidity": 60},
        }
        default = defaults.get(city, {"temp": 28, "humidity": 65})
        temp = default["temp"]
        humidity = default["humidity"]
        dew_factor = calculate_dew_factor(temp, humidity)
        
        result = {
            "temperature": temp,
            "humidity": humidity,
            "dew_factor": round(dew_factor, 1),
            "city": city,
            "source": "default",
            "description": "Default conditions"
        }
        _weather_cache[venue_id] = (result, datetime.now())
        return result

def get_match_time_weather(venue_id: str, match_hour: int = 19) -> Optional[Dict]:
    """Get weather for specific match time."""
    weather = get_weather_data(venue_id)
    if weather:
        if match_hour >= 19:
            weather["temperature"] -= 2
            weather["humidity"] += 5
            weather["dew_factor"] = round(calculate_dew_factor(
                weather["temperature"], weather["humidity"]
            ), 1)
        weather["match_time"] = f"{match_hour}:30 IST"
    return weather