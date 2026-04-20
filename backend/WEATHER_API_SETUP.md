# Weather API Setup

## Get Free OpenWeatherMap API Key

1. Go to [OpenWeatherMap](https://openweathermap.org/api)
2. Click "Sign Up" and create a free account
3. Go to "API Keys" section in your dashboard
4. Copy your API key
5. Open `weather_service.py` and replace:
   ```python
   API_KEY = "your_openweather_api_key_here"
   ```
   with:
   ```python
   API_KEY = "your_actual_api_key"
   ```

## Free Tier Limits
- 1,000 API calls per day
- Current weather data
- 5-day forecast available

## Default Behavior
- Without API key: Uses realistic default weather values for each city
- With API key: Fetches live weather data from OpenWeatherMap

## Features
- ✅ Real-time temperature, humidity, and dew factor
- ✅ Automatic dew factor calculation for cricket conditions
- ✅ Evening match time adjustments (7:30 PM IST)
- ✅ 26 IPL venue coordinates pre-configured
- ✅ Fallback to defaults if API fails

Restart the backend after adding your API key!