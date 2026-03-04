# =============================================================================
# Air Quality Module
# Uses Open-Meteo Air Quality API (Free, no API key required)
# =============================================================================

import requests
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd

from config import OPEN_METEO_AIR_QUALITY_URL, ALERT_THRESHOLDS, COLORS


def fetch_air_quality_data(lat: float, lon: float, days: int = 5) -> Optional[Dict[str, Any]]:
    """
    Fetch air quality data from Open-Meteo Air Quality API.
    
    Args:
        lat: Latitude
        lon: Longitude
        days: Forecast days (1-5)
    
    Returns:
        Dictionary with air quality data
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "aerosol_optical_depth",
            "dust",
            "uv_index",
            "uv_index_clear_sky",
            "european_aqi",
            "us_aqi",
        ],
        "current": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "european_aqi",
            "us_aqi",
        ],
        "timezone": "GMT",
        "forecast_days": min(days, 5),
    }
    
    try:
        response = requests.get(OPEN_METEO_AIR_QUALITY_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching air quality data: {e}")
        return None


def process_air_quality_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process raw air quality data into a usable format."""
    if not raw_data:
        return {}
    
    processed = {
        "current": {},
        "hourly": None,
        "location": {
            "latitude": raw_data.get("latitude"),
            "longitude": raw_data.get("longitude"),
            "timezone": raw_data.get("timezone"),
        }
    }
    
    # Process current air quality
    if "current" in raw_data:
        current = raw_data["current"]
        aqi = current.get("us_aqi", current.get("european_aqi", 0))
        processed["current"] = {
            "pm10": current.get("pm10"),
            "pm2_5": current.get("pm2_5"),
            "co": current.get("carbon_monoxide"),
            "no2": current.get("nitrogen_dioxide"),
            "so2": current.get("sulphur_dioxide"),
            "ozone": current.get("ozone"),
            "european_aqi": current.get("european_aqi"),
            "us_aqi": current.get("us_aqi"),
            "aqi_category": get_aqi_category(aqi),
            "aqi_color": get_aqi_color(aqi),
            "health_message": get_health_message(aqi),
            "time": current.get("time"),
        }
    
    # Process hourly data
    if "hourly" in raw_data:
        hourly = raw_data["hourly"]
        processed["hourly"] = pd.DataFrame({
            "time": pd.to_datetime(hourly.get("time", [])),
            "pm10": hourly.get("pm10", []),
            "pm2_5": hourly.get("pm2_5", []),
            "co": hourly.get("carbon_monoxide", []),
            "no2": hourly.get("nitrogen_dioxide", []),
            "so2": hourly.get("sulphur_dioxide", []),
            "ozone": hourly.get("ozone", []),
            "dust": hourly.get("dust", []),
            "aod": hourly.get("aerosol_optical_depth", []),
            "uv_index": hourly.get("uv_index", []),
            "european_aqi": hourly.get("european_aqi", []),
            "us_aqi": hourly.get("us_aqi", []),
        })
    
    return processed


def get_aqi_category(aqi: float) -> str:
    """Get AQI category based on US AQI scale."""
    if aqi is None:
        return "Unknown"
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def get_aqi_color(aqi: float) -> str:
    """Get color code for AQI value."""
    if aqi is None:
        return "#808080"
    if aqi <= 50:
        return COLORS["air_quality_good"]
    elif aqi <= 100:
        return COLORS["air_quality_moderate"]
    elif aqi <= 200:
        return COLORS["air_quality_unhealthy"]
    elif aqi <= 300:
        return COLORS["air_quality_very_unhealthy"]
    else:
        return COLORS["air_quality_hazardous"]


def get_health_message(aqi: float) -> str:
    """Get health recommendation based on AQI."""
    if aqi is None:
        return "Air quality data unavailable."
    if aqi <= 50:
        return "Air quality is satisfactory. Outdoor activities are safe for all groups."
    elif aqi <= 100:
        return "Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion."
    elif aqi <= 150:
        return "Sensitive groups (children, elderly, those with respiratory conditions) should limit prolonged outdoor exertion."
    elif aqi <= 200:
        return "Everyone may begin to experience health effects. Sensitive groups should avoid prolonged outdoor exertion."
    elif aqi <= 300:
        return "Health alert: everyone may experience more serious health effects. Avoid prolonged outdoor activities."
    else:
        return "Health emergency! Everyone should avoid all outdoor activities."


def analyze_air_quality_for_solar(air_quality_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze air quality impact on solar energy production.
    
    High aerosol content and dust reduce solar irradiance reaching panels.
    """
    if not air_quality_data or "hourly" not in air_quality_data:
        return {"impact_available": False}
    
    hourly = air_quality_data["hourly"]
    
    analysis = {
        "impact_available": True,
        "average_dust": hourly["dust"].mean() if "dust" in hourly.columns else None,
        "max_dust": hourly["dust"].max() if "dust" in hourly.columns else None,
        "average_aod": hourly["aod"].mean() if "aod" in hourly.columns else None,
        "pm10_average": hourly["pm10"].mean(),
        "solar_impact_factors": [],
    }
    
    # Aerosol Optical Depth impact
    if analysis["average_aod"] is not None:
        aod = analysis["average_aod"]
        if aod > 0.5:
            transmission_loss = min(aod * 15, 30)  # Simplified model
            analysis["solar_impact_factors"].append({
                "factor": "Aerosol Optical Depth",
                "value": aod,
                "impact_percent": transmission_loss,
                "message": f"High aerosol content reducing solar irradiance by ~{transmission_loss:.0f}%",
            })
    
    # Dust impact
    if analysis["average_dust"] is not None:
        dust = analysis["average_dust"]
        if dust > 50:  # µg/m³
            dust_impact = min(dust * 0.1, 20)  # Simplified model
            analysis["solar_impact_factors"].append({
                "factor": "Dust Concentration",
                "value": dust,
                "impact_percent": dust_impact,
                "message": f"Dust concentration may increase panel soiling rate by ~{dust_impact:.0f}%",
            })
    
    # PM10 impact
    pm10_avg = analysis["pm10_average"]
    if pm10_avg is not None and pm10_avg > 100:
        pm_impact = min((pm10_avg - 100) * 0.05, 15)
        analysis["solar_impact_factors"].append({
            "factor": "PM10",
            "value": pm10_avg,
            "impact_percent": pm_impact,
            "message": f"High particulate matter may reduce solar output by ~{pm_impact:.0f}%",
        })
    
    # Calculate total estimated impact
    total_impact = sum(f["impact_percent"] for f in analysis["solar_impact_factors"])
    analysis["total_estimated_impact_percent"] = min(total_impact, 40)
    analysis["transmission_efficiency"] = 1 - (analysis["total_estimated_impact_percent"] / 100)
    
    return analysis


def generate_air_quality_alerts(air_quality_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate alerts for poor air quality conditions."""
    alerts = []
    
    if not air_quality_data or "current" not in air_quality_data:
        return alerts
    
    current = air_quality_data["current"]
    aqi = current.get("us_aqi") or current.get("european_aqi")
    
    if aqi is None:
        return alerts
    
    if aqi >= ALERT_THRESHOLDS["aqi_hazardous"]:
        alerts.append({
            "level": "danger",
            "title": "Hazardous Air Quality",
            "message": f"AQI: {aqi:.0f}. Health emergency. Avoid all outdoor activities.",
            "icon": "☠️",
            "value": aqi,
        })
    elif aqi >= ALERT_THRESHOLDS["aqi_very_unhealthy"]:
        alerts.append({
            "level": "warning",
            "title": "Very Unhealthy Air Quality",
            "message": f"AQI: {aqi:.0f}. Everyone may experience health effects.",
            "icon": "😷",
            "value": aqi,
        })
    elif aqi >= ALERT_THRESHOLDS["aqi_unhealthy"]:
        alerts.append({
            "level": "advisory",
            "title": "Unhealthy Air Quality",
            "message": f"AQI: {aqi:.0f}. Sensitive groups should limit outdoor activities.",
            "icon": "⚠️",
            "value": aqi,
        })
    elif aqi >= ALERT_THRESHOLDS["aqi_moderate"]:
        alerts.append({
            "level": "info",
            "title": "Moderate Air Quality",
            "message": f"AQI: {aqi:.0f}. Unusually sensitive people should consider limiting outdoor exertion.",
            "icon": "ℹ️",
            "value": aqi,
        })
    
    # Specific pollutant alerts
    pm25 = current.get("pm2_5")
    if pm25 is not None and pm25 > 35:  # WHO guideline
        alerts.append({
            "level": "advisory",
            "title": "Elevated PM2.5",
            "message": f"Fine particulate matter at {pm25:.0f} µg/m³ exceeds WHO guidelines.",
            "icon": "🌫️",
            "value": pm25,
        })
    
    ozone = current.get("ozone")
    if ozone is not None and ozone > 100:  # µg/m³
        alerts.append({
            "level": "advisory",
            "title": "High Ozone Levels",
            "message": f"Ground-level ozone at {ozone:.0f} µg/m³. May cause respiratory irritation.",
            "icon": "🌞",
            "value": ozone,
        })
    
    return alerts


def get_pollutant_info() -> Dict[str, Dict[str, Any]]:
    """Get information about measured pollutants."""
    return {
        "pm2_5": {
            "name": "PM2.5",
            "full_name": "Fine Particulate Matter",
            "unit": "µg/m³",
            "who_guideline": 15,  # Annual mean
            "description": "Tiny particles that can penetrate deep into lungs and bloodstream.",
            "sources": "Combustion, traffic, industrial processes, dust",
        },
        "pm10": {
            "name": "PM10",
            "full_name": "Coarse Particulate Matter",
            "unit": "µg/m³",
            "who_guideline": 45,  # Annual mean
            "description": "Particles that can irritate airways and affect the respiratory system.",
            "sources": "Road dust, construction, sand storms, pollen",
        },
        "ozone": {
            "name": "O₃",
            "full_name": "Ozone",
            "unit": "µg/m³",
            "who_guideline": 100,  # 8-hour mean
            "description": "A reactive gas that can cause breathing difficulties and lung damage.",
            "sources": "Formed by reactions between NOx and VOCs in sunlight",
        },
        "no2": {
            "name": "NO₂",
            "full_name": "Nitrogen Dioxide",
            "unit": "µg/m³",
            "who_guideline": 25,  # Annual mean
            "description": "A gas that irritates airways and can aggravate respiratory diseases.",
            "sources": "Vehicle exhaust, power plants, industrial processes",
        },
        "so2": {
            "name": "SO₂",
            "full_name": "Sulphur Dioxide",
            "unit": "µg/m³",
            "who_guideline": 40,  # 24-hour mean
            "description": "A gas that can cause breathing difficulties and worsen existing conditions.",
            "sources": "Burning fossil fuels, industrial processes, volcanoes",
        },
        "co": {
            "name": "CO",
            "full_name": "Carbon Monoxide",
            "unit": "µg/m³",
            "who_guideline": 4000,  # 24-hour mean
            "description": "A colorless, odorless gas that reduces oxygen delivery in the body.",
            "sources": "Incomplete combustion, vehicle exhaust, fires",
        },
    }
