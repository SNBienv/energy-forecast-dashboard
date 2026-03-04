# =============================================================================
# Weather & Wind Data Module
# Uses Open-Meteo API (Free, no API key required)
# =============================================================================

import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd

from config import OPEN_METEO_WEATHER_URL, ALERT_THRESHOLDS


def fetch_weather_data(
    lat: float, lon: float, days: int = 7,
    resolution: str = "hourly",
) -> Optional[Dict[str, Any]]:
    """
    Fetch comprehensive weather forecast data from Open-Meteo API.

    Args:
        lat: Latitude of the location
        lon: Longitude of the location
        days: Number of forecast days (1-16)
        resolution: "hourly" | "15min" | "daily"

    Returns:
        Dictionary with weather data or None if request fails
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation_probability",
            "precipitation",
            "rain",
            "weather_code",
            "pressure_msl",
            "cloud_cover",
            "visibility",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "uv_index",
        ],
        "daily": [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "sunrise",
            "sunset",
            "daylight_duration",
            "sunshine_duration",
            "uv_index_max",
            "precipitation_sum",
            "rain_sum",
            "precipitation_hours",
            "precipitation_probability_max",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "wind_direction_10m_dominant",
        ],
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "is_day",
            "precipitation",
            "rain",
            "weather_code",
            "cloud_cover",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
        ],
        "timezone": "GMT",
        "forecast_days": min(days, 16),
    }

    # Sub-hourly: also request 15-minute variables (Open-Meteo minutely_15)
    if resolution == "15min":
        params["minutely_15"] = [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "weather_code",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
            "global_tilted_irradiance",
        ]

    try:
        response = requests.get(OPEN_METEO_WEATHER_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        data["_resolution"] = resolution          # tag for downstream
        return data
    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None


def get_weather_code_description(code: int) -> Dict[str, str]:
    """Convert WMO weather code to description and icon."""
    weather_codes = {
        0: {"description": "Clear sky", "icon": "☀️"},
        1: {"description": "Mainly clear", "icon": "🌤️"},
        2: {"description": "Partly cloudy", "icon": "⛅"},
        3: {"description": "Overcast", "icon": "☁️"},
        45: {"description": "Fog", "icon": "🌫️"},
        48: {"description": "Depositing rime fog", "icon": "🌫️"},
        51: {"description": "Light drizzle", "icon": "🌧️"},
        53: {"description": "Moderate drizzle", "icon": "🌧️"},
        55: {"description": "Dense drizzle", "icon": "🌧️"},
        56: {"description": "Light freezing drizzle", "icon": "🌨️"},
        57: {"description": "Dense freezing drizzle", "icon": "🌨️"},
        61: {"description": "Slight rain", "icon": "🌧️"},
        63: {"description": "Moderate rain", "icon": "🌧️"},
        65: {"description": "Heavy rain", "icon": "🌧️"},
        66: {"description": "Light freezing rain", "icon": "🌨️"},
        67: {"description": "Heavy freezing rain", "icon": "🌨️"},
        71: {"description": "Slight snow fall", "icon": "❄️"},
        73: {"description": "Moderate snow fall", "icon": "❄️"},
        75: {"description": "Heavy snow fall", "icon": "❄️"},
        77: {"description": "Snow grains", "icon": "❄️"},
        80: {"description": "Slight rain showers", "icon": "🌦️"},
        81: {"description": "Moderate rain showers", "icon": "🌦️"},
        82: {"description": "Violent rain showers", "icon": "⛈️"},
        85: {"description": "Slight snow showers", "icon": "🌨️"},
        86: {"description": "Heavy snow showers", "icon": "🌨️"},
        95: {"description": "Thunderstorm", "icon": "⛈️"},
        96: {"description": "Thunderstorm with slight hail", "icon": "⛈️"},
        99: {"description": "Thunderstorm with heavy hail", "icon": "⛈️"},
    }
    return weather_codes.get(code, {"description": "Unknown", "icon": "❓"})


def get_wind_direction_name(degrees: float) -> str:
    """Convert wind direction degrees to cardinal direction."""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(degrees / 22.5) % 16
    return directions[index]


def process_weather_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process raw API data into a more usable format."""
    if not raw_data:
        return {}

    resolution = raw_data.get("_resolution", "hourly")

    processed = {
        "current": {},
        "hourly": {},
        "daily": {},
        "minutely_15": None,   # populated when resolution == "15min"
        "daily_agg": None,     # daily aggregation of hourly data
        "_resolution": resolution,
        "location": {
            "latitude": raw_data.get("latitude"),
            "longitude": raw_data.get("longitude"),
            "timezone": raw_data.get("timezone"),
            "elevation": raw_data.get("elevation"),
        }
    }
    
    # Process current weather
    if "current" in raw_data:
        current = raw_data["current"]
        weather_info = get_weather_code_description(current.get("weather_code", 0))
        processed["current"] = {
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "rain": current.get("rain"),
            "weather_code": current.get("weather_code"),
            "weather_description": weather_info["description"],
            "weather_icon": weather_info["icon"],
            "cloud_cover": current.get("cloud_cover"),
            "pressure": current.get("pressure_msl"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_direction": current.get("wind_direction_10m"),
            "wind_direction_name": get_wind_direction_name(current.get("wind_direction_10m", 0)),
            "wind_gusts": current.get("wind_gusts_10m"),
            "is_day": current.get("is_day"),
            "time": current.get("time"),
        }
    
    # Process hourly data into DataFrame
    if "hourly" in raw_data:
        hourly = raw_data["hourly"]
        processed["hourly"] = pd.DataFrame({
            "time": pd.to_datetime(hourly.get("time", [])),
            "temperature": hourly.get("temperature_2m", []),
            "feels_like": hourly.get("apparent_temperature", []),
            "humidity": hourly.get("relative_humidity_2m", []),
            "dew_point": hourly.get("dew_point_2m", []),
            "precipitation_probability": hourly.get("precipitation_probability", []),
            "precipitation": hourly.get("precipitation", []),
            "rain": hourly.get("rain", []),
            "weather_code": hourly.get("weather_code", []),
            "pressure": hourly.get("pressure_msl", []),
            "cloud_cover": hourly.get("cloud_cover", []),
            "visibility": hourly.get("visibility", []),
            "wind_speed": hourly.get("wind_speed_10m", []),
            "wind_direction": hourly.get("wind_direction_10m", []),
            "wind_gusts": hourly.get("wind_gusts_10m", []),
            "uv_index": hourly.get("uv_index", []),
        })
    
    # Process daily data into DataFrame
    if "daily" in raw_data:
        daily = raw_data["daily"]
        processed["daily"] = pd.DataFrame({
            "date": pd.to_datetime(daily.get("time", [])),
            "weather_code": daily.get("weather_code", []),
            "temp_max": daily.get("temperature_2m_max", []),
            "temp_min": daily.get("temperature_2m_min", []),
            "feels_like_max": daily.get("apparent_temperature_max", []),
            "feels_like_min": daily.get("apparent_temperature_min", []),
            "sunrise": daily.get("sunrise", []),
            "sunset": daily.get("sunset", []),
            "daylight_duration": daily.get("daylight_duration", []),
            "sunshine_duration": daily.get("sunshine_duration", []),
            "uv_index_max": daily.get("uv_index_max", []),
            "precipitation_sum": daily.get("precipitation_sum", []),
            "rain_sum": daily.get("rain_sum", []),
            "precipitation_hours": daily.get("precipitation_hours", []),
            "precipitation_probability_max": daily.get("precipitation_probability_max", []),
            "wind_speed_max": daily.get("wind_speed_10m_max", []),
            "wind_gusts_max": daily.get("wind_gusts_10m_max", []),
            "wind_direction_dominant": daily.get("wind_direction_10m_dominant", []),
        })

    # ----- Sub-hourly (15-minute) DataFrame -----
    if "minutely_15" in raw_data:
        m15 = raw_data["minutely_15"]
        n = len(m15.get("time", []))
        processed["minutely_15"] = pd.DataFrame({
            "time": pd.to_datetime(m15.get("time", [])),
            "temperature": m15.get("temperature_2m", [None]*n),
            "feels_like": m15.get("apparent_temperature", [None]*n),
            "humidity": m15.get("relative_humidity_2m", [None]*n),
            "dew_point": m15.get("dew_point_2m", [None]*n),
            "precipitation_probability": [None]*n,  # not in 15-min API
            "precipitation": m15.get("precipitation", [None]*n),
            "rain": m15.get("rain", [None]*n),
            "weather_code": m15.get("weather_code", [None]*n),
            "pressure": [None]*n,          # not in 15-min API
            "cloud_cover": [None]*n,       # not in 15-min API
            "visibility": [None]*n,        # not in 15-min API
            "wind_speed": m15.get("wind_speed_10m", [None]*n),
            "wind_direction": m15.get("wind_direction_10m", [None]*n),
            "wind_gusts": m15.get("wind_gusts_10m", [None]*n),
            "uv_index": [None]*n,          # not in 15-min API
        })
        # Forward-fill NaN-only columns from hourly if hourly is available
        if isinstance(processed.get("hourly"), pd.DataFrame) and not processed["hourly"].empty:
            for col in ["pressure", "cloud_cover", "visibility", "uv_index", "precipitation_probability"]:
                if col in processed["hourly"].columns:
                    # merge-asof to broadcast hourly into 15-min slots
                    h = processed["hourly"][["time", col]].dropna(subset=[col])
                    if not h.empty:
                        merged = pd.merge_asof(
                            processed["minutely_15"][["time"]],
                            h, on="time", direction="backward",
                        )
                        processed["minutely_15"][col] = merged[col]

    # ----- Daily aggregation of hourly data -----
    if isinstance(processed.get("hourly"), pd.DataFrame) and not processed["hourly"].empty:
        h = processed["hourly"].copy()
        h["_date"] = h["time"].dt.date
        agg = h.groupby("_date").agg(
            temperature=("temperature", "mean"),
            humidity=("humidity", "mean"),
            pressure=("pressure", "mean"),
            cloud_cover=("cloud_cover", "mean"),
            wind_speed=("wind_speed", "mean"),
            wind_gusts=("wind_gusts", "max"),
            wind_direction=("wind_direction", lambda s: s.mode().iloc[0] if not s.mode().empty else s.mean()),
            precipitation=("precipitation", "sum"),
            rain=("rain", "sum"),
            uv_index=("uv_index", "max"),
        ).reset_index()
        agg.rename(columns={"_date": "time"}, inplace=True)
        agg["time"] = pd.to_datetime(agg["time"])
        processed["daily_agg"] = agg

    return processed


def calculate_wind_power_density(wind_speed: float, air_density: float = 1.225) -> float:
    """
    Calculate wind power density (W/m²).
    
    Args:
        wind_speed: Wind speed in m/s
        air_density: Air density in kg/m³ (default 1.225 at sea level)
    
    Returns:
        Wind power density in W/m²
    """
    # Convert km/h to m/s if necessary (assuming input is in km/h)
    wind_speed_ms = wind_speed / 3.6
    return 0.5 * air_density * (wind_speed_ms ** 3)


def estimate_wind_energy_potential(
    wind_speed: float,
    turbine_capacity: float = 2000,  # kW
    rotor_diameter: float = 90,  # meters
    efficiency: float = 0.35,
) -> Dict[str, float]:
    """
    Estimate wind energy generation potential.
    
    Args:
        wind_speed: Wind speed in km/h
        turbine_capacity: Turbine rated capacity in kW
        rotor_diameter: Rotor diameter in meters
        efficiency: Overall system efficiency (0-1)
    
    Returns:
        Dictionary with energy estimates
    """
    wind_speed_ms = wind_speed / 3.6
    swept_area = 3.14159 * (rotor_diameter / 2) ** 2
    power_density = calculate_wind_power_density(wind_speed)
    
    # Betz limit (theoretical maximum ~59%)
    betz_limit = 0.593
    
    # Theoretical power output
    theoretical_power = power_density * swept_area * betz_limit * efficiency
    
    # Capacity factor estimation (simplified)
    if wind_speed_ms < 3:
        capacity_factor = 0
    elif wind_speed_ms < 12:
        capacity_factor = (wind_speed_ms - 3) / 9 * 0.35
    elif wind_speed_ms < 25:
        capacity_factor = 0.35
    else:
        capacity_factor = 0
    
    return {
        "power_density_wm2": power_density,
        "theoretical_power_kw": theoretical_power / 1000,
        "capacity_factor": capacity_factor,
        "estimated_daily_kwh": min(theoretical_power / 1000, turbine_capacity) * 24 * capacity_factor,
        "wind_class": classify_wind_resource(wind_speed_ms),
    }


def classify_wind_resource(wind_speed_ms: float) -> str:
    """Classify wind resource based on average wind speed."""
    if wind_speed_ms < 4.4:
        return "Class 1 - Poor"
    elif wind_speed_ms < 5.6:
        return "Class 2 - Marginal"
    elif wind_speed_ms < 6.4:
        return "Class 3 - Fair"
    elif wind_speed_ms < 7.0:
        return "Class 4 - Good"
    elif wind_speed_ms < 7.5:
        return "Class 5 - Excellent"
    elif wind_speed_ms < 8.0:
        return "Class 6 - Outstanding"
    else:
        return "Class 7 - Superb"
