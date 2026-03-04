# =============================================================================
# Extreme Weather Events Forecasting Module
# Analyzes weather data to identify and predict severe weather conditions
# =============================================================================

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd

from config import ALERT_THRESHOLDS


class AlertLevel:
    """Alert severity levels."""
    INFO = "info"
    ADVISORY = "advisory"
    WARNING = "warning"
    DANGER = "danger"
    EMERGENCY = "emergency"


class AlertType:
    """Types of weather alerts."""
    HIGH_WIND = "high_wind"
    EXTREME_HEAT = "extreme_heat"
    EXTREME_COLD = "extreme_cold"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"
    DUST_STORM = "dust_storm"
    FLOODING = "flooding"
    UV_RADIATION = "uv_radiation"
    POOR_VISIBILITY = "poor_visibility"
    TROPICAL_STORM = "tropical_storm"


def analyze_weather_alerts(weather_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Analyze weather data and generate alerts for extreme conditions.
    
    Args:
        weather_data: Processed weather data dictionary
    
    Returns:
        List of alert dictionaries
    """
    alerts = []
    
    # Analyze current conditions
    if "current" in weather_data and weather_data["current"]:
        current = weather_data["current"]
        alerts.extend(analyze_current_conditions(current))
    
    # Analyze forecast data
    if "hourly" in weather_data and weather_data["hourly"] is not None:
        alerts.extend(analyze_hourly_forecast(weather_data["hourly"]))
    
    if "daily" in weather_data and weather_data["daily"] is not None:
        alerts.extend(analyze_daily_forecast(weather_data["daily"]))
    
    # Sort by severity and time
    severity_order = {
        AlertLevel.EMERGENCY: 0,
        AlertLevel.DANGER: 1,
        AlertLevel.WARNING: 2,
        AlertLevel.ADVISORY: 3,
        AlertLevel.INFO: 4,
    }
    alerts.sort(key=lambda x: (severity_order.get(x["level"], 5), x.get("time", "")))
    
    return alerts


def analyze_current_conditions(current: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze current weather conditions for alerts."""
    alerts = []
    
    # High wind alert
    wind_speed = current.get("wind_speed", 0)
    wind_gusts = current.get("wind_gusts", 0)
    
    if wind_gusts >= ALERT_THRESHOLDS["wind_speed_danger"]:
        alerts.append({
            "type": AlertType.HIGH_WIND,
            "level": AlertLevel.DANGER,
            "title": "Dangerous Wind Conditions",
            "message": f"Wind gusts of {wind_gusts:.0f} km/h detected. Secure loose objects and avoid outdoor activities.",
            "time": current.get("time"),
            "value": wind_gusts,
            "icon": "💨",
        })
    elif wind_speed >= ALERT_THRESHOLDS["wind_speed_warning"]:
        alerts.append({
            "type": AlertType.HIGH_WIND,
            "level": AlertLevel.WARNING,
            "title": "High Wind Warning",
            "message": f"Wind speeds of {wind_speed:.0f} km/h. Exercise caution outdoors.",
            "time": current.get("time"),
            "value": wind_speed,
            "icon": "💨",
        })
    
    # Extreme heat alert
    temperature = current.get("temperature", 0)
    feels_like = current.get("feels_like", temperature)
    
    if feels_like >= ALERT_THRESHOLDS["temp_high_warning"]:
        alerts.append({
            "type": AlertType.EXTREME_HEAT,
            "level": AlertLevel.WARNING,
            "title": "Extreme Heat Warning",
            "message": f"Temperature feels like {feels_like:.0f}°C. Stay hydrated and avoid prolonged sun exposure.",
            "time": current.get("time"),
            "value": feels_like,
            "icon": "🌡️",
        })
    
    # Extreme cold alert
    if temperature <= ALERT_THRESHOLDS["temp_low_warning"]:
        alerts.append({
            "type": AlertType.EXTREME_COLD,
            "level": AlertLevel.ADVISORY,
            "title": "Cold Weather Advisory",
            "message": f"Temperature of {temperature:.0f}°C. Dress warmly and protect sensitive equipment.",
            "time": current.get("time"),
            "value": temperature,
            "icon": "❄️",
        })
    
    # Thunderstorm alert
    weather_code = current.get("weather_code", 0)
    if weather_code in [95, 96, 99]:
        alerts.append({
            "type": AlertType.THUNDERSTORM,
            "level": AlertLevel.WARNING,
            "title": "Thunderstorm Alert",
            "message": "Active thunderstorm detected. Seek shelter and avoid open areas.",
            "time": current.get("time"),
            "value": weather_code,
            "icon": "⛈️",
        })
    
    return alerts


def analyze_hourly_forecast(hourly_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyze hourly forecast for upcoming extreme conditions."""
    alerts = []
    
    # Look at next 24 hours
    next_24h = hourly_df.head(24) if len(hourly_df) >= 24 else hourly_df
    
    # Check for upcoming high winds
    high_wind_hours = next_24h[next_24h["wind_speed"] >= ALERT_THRESHOLDS["wind_speed_warning"]]
    if not high_wind_hours.empty:
        max_wind = high_wind_hours["wind_speed"].max()
        first_hour = high_wind_hours.iloc[0]["time"]
        alerts.append({
            "type": AlertType.HIGH_WIND,
            "level": AlertLevel.ADVISORY,
            "title": "High Winds Forecast",
            "message": f"Winds up to {max_wind:.0f} km/h expected starting {format_time(first_hour)}.",
            "time": str(first_hour),
            "value": max_wind,
            "icon": "💨",
        })
    
    # Check for high UV index
    high_uv_hours = next_24h[next_24h["uv_index"] >= ALERT_THRESHOLDS["uv_warning"]]
    if not high_uv_hours.empty:
        max_uv = high_uv_hours["uv_index"].max()
        level = AlertLevel.WARNING if max_uv >= ALERT_THRESHOLDS["uv_danger"] else AlertLevel.ADVISORY
        alerts.append({
            "type": AlertType.UV_RADIATION,
            "level": level,
            "title": "High UV Radiation",
            "message": f"UV index reaching {max_uv:.0f}. Use sun protection and limit exposure.",
            "time": str(high_uv_hours.iloc[0]["time"]),
            "value": max_uv,
            "icon": "☀️",
        })
    
    # Check for heavy precipitation
    heavy_rain_hours = next_24h[next_24h["precipitation"] >= ALERT_THRESHOLDS["precipitation_heavy"]]
    if not heavy_rain_hours.empty:
        max_precip = heavy_rain_hours["precipitation"].max()
        alerts.append({
            "type": AlertType.HEAVY_RAIN,
            "level": AlertLevel.WARNING,
            "title": "Heavy Rain Expected",
            "message": f"Precipitation up to {max_precip:.1f} mm/h expected. Possible flooding in low areas.",
            "time": str(heavy_rain_hours.iloc[0]["time"]),
            "value": max_precip,
            "icon": "🌧️",
        })
    
    # Check for poor visibility
    if "visibility" in next_24h.columns:
        low_vis_hours = next_24h[next_24h["visibility"] < 1000]  # Less than 1km
        if not low_vis_hours.empty:
            min_vis = low_vis_hours["visibility"].min()
            alerts.append({
                "type": AlertType.POOR_VISIBILITY,
                "level": AlertLevel.ADVISORY,
                "title": "Reduced Visibility",
                "message": f"Visibility dropping to {min_vis:.0f}m. Exercise caution if traveling.",
                "time": str(low_vis_hours.iloc[0]["time"]),
                "value": min_vis,
                "icon": "🌫️",
            })
    
    return alerts


def analyze_daily_forecast(daily_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyze daily forecast for multi-day weather events."""
    alerts = []
    
    # Check for extended periods of extreme heat
    hot_days = daily_df[daily_df["temp_max"] >= ALERT_THRESHOLDS["temp_high_warning"]]
    if len(hot_days) >= 3:
        alerts.append({
            "type": AlertType.EXTREME_HEAT,
            "level": AlertLevel.WARNING,
            "title": "Heat Wave Expected",
            "message": f"{len(hot_days)} consecutive days above {ALERT_THRESHOLDS['temp_high_warning']}°C. Take precautions.",
            "time": str(hot_days.iloc[0]["date"]),
            "value": len(hot_days),
            "icon": "🔥",
        })
    
    # Check for high cumulative precipitation (flooding risk)
    total_precip = daily_df.head(3)["precipitation_sum"].sum()
    if total_precip >= 100:
        alerts.append({
            "type": AlertType.FLOODING,
            "level": AlertLevel.WARNING,
            "title": "Flooding Risk",
            "message": f"{total_precip:.0f}mm of rain expected over 3 days. Flash flooding possible.",
            "time": str(daily_df.iloc[0]["date"]),
            "value": total_precip,
            "icon": "🌊",
        })
    
    # Check for consistently high winds
    windy_days = daily_df[daily_df["wind_speed_max"] >= ALERT_THRESHOLDS["wind_speed_warning"]]
    if len(windy_days) >= 2:
        max_wind = daily_df["wind_gusts_max"].max()
        alerts.append({
            "type": AlertType.HIGH_WIND,
            "level": AlertLevel.ADVISORY,
            "title": "Prolonged Windy Period",
            "message": f"Strong winds expected for {len(windy_days)} days. Gusts up to {max_wind:.0f} km/h.",
            "time": str(windy_days.iloc[0]["date"]),
            "value": max_wind,
            "icon": "🌬️",
        })
    
    return alerts


def analyze_renewable_energy_impact(
    weather_data: Dict[str, Any],
    alerts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze how weather conditions impact renewable energy systems.
    
    Returns:
        Dictionary with impact assessments for solar and wind systems
    """
    impacts = {
        "solar": [],
        "wind": [],
        "general": [],
    }
    
    # Analyze current conditions
    current = weather_data.get("current", {})
    
    # Solar impacts
    cloud_cover = current.get("cloud_cover", 0)
    if cloud_cover > 80:
        impacts["solar"].append({
            "severity": "moderate",
            "message": f"High cloud cover ({cloud_cover}%) reducing solar output by ~{cloud_cover * 0.7:.0f}%",
            "icon": "☁️",
        })
    
    # Dust storm assessment for solar (based on visibility and wind)
    if current.get("visibility", 10000) < 5000 and current.get("wind_speed", 0) > 30:
        impacts["solar"].append({
            "severity": "significant",
            "message": "Possible dust conditions - increased soiling expected",
            "icon": "🏜️",
        })
    
    # Wind energy impacts
    wind_speed = current.get("wind_speed", 0)
    wind_speed_ms = wind_speed / 3.6
    
    if wind_speed_ms < 3:
        impacts["wind"].append({
            "severity": "significant",
            "message": f"Wind speed ({wind_speed:.0f} km/h) below cut-in threshold. Turbines idle.",
            "icon": "🔴",
        })
    elif wind_speed_ms > 25:
        impacts["wind"].append({
            "severity": "critical",
            "message": f"Wind speed ({wind_speed:.0f} km/h) above cut-out. Turbines shut down for protection.",
            "icon": "⚠️",
        })
    elif wind_speed_ms >= 12:
        impacts["wind"].append({
            "severity": "positive",
            "message": f"Optimal wind conditions ({wind_speed:.0f} km/h). Turbines at rated capacity.",
            "icon": "✅",
        })
    
    # Check alerts for energy-relevant conditions
    for alert in alerts:
        if alert["type"] == AlertType.THUNDERSTORM:
            impacts["general"].append({
                "severity": "critical",
                "message": "Thunderstorm: Lightning risk for wind turbines and electrical systems",
                "icon": "⚡",
            })
        elif alert["type"] == AlertType.EXTREME_HEAT:
            impacts["solar"].append({
                "severity": "moderate",
                "message": "High temperatures reducing solar panel efficiency by ~5-10%",
                "icon": "🌡️",
            })
    
    return impacts


def format_time(dt) -> str:
    """Format datetime for display."""
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except:
            return dt
    
    if isinstance(dt, datetime):
        return dt.strftime("%a %H:%M")
    return str(dt)


def get_alert_color(level: str) -> str:
    """Get display color for alert level."""
    colors = {
        AlertLevel.INFO: "#3498db",
        AlertLevel.ADVISORY: "#f1c40f",
        AlertLevel.WARNING: "#e67e22",
        AlertLevel.DANGER: "#e74c3c",
        AlertLevel.EMERGENCY: "#8e44ad",
    }
    return colors.get(level, "#95a5a6")


def generate_forecast_summary(
    daily_df: pd.DataFrame,
    location_name: str = "Location"
) -> str:
    """Generate a human-readable weather forecast summary."""
    if daily_df is None or daily_df.empty:
        return "No forecast data available."
    
    today = daily_df.iloc[0]
    tomorrow = daily_df.iloc[1] if len(daily_df) > 1 else None
    
    from modules.weather_wind import get_weather_code_description
    
    weather_info = get_weather_code_description(today.get("weather_code", 0))
    
    summary = f"**{location_name} Weather Outlook**\n\n"
    summary += f"**Today**: {weather_info['icon']} {weather_info['description']}\n"
    summary += f"- High: {today['temp_max']:.0f}°C, Low: {today['temp_min']:.0f}°C\n"
    summary += f"- Wind: Up to {today['wind_speed_max']:.0f} km/h\n"
    
    if today.get("precipitation_sum", 0) > 0:
        summary += f"- Rain: {today['precipitation_sum']:.1f}mm\n"
    
    if tomorrow is not None:
        tomorrow_weather = get_weather_code_description(tomorrow.get("weather_code", 0))
        summary += f"\n**Tomorrow**: {tomorrow_weather['icon']} {tomorrow_weather['description']}\n"
        summary += f"- High: {tomorrow['temp_max']:.0f}°C, Low: {tomorrow['temp_min']:.0f}°C\n"
    
    return summary
