# =============================================================================
# Solar Energy & Soiling Effect Module
# Uses Open-Meteo for solar radiation and NASA POWER API for historical data
# =============================================================================

import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
import pandas as pd
import math

from config import (
    OPEN_METEO_BASE_URL,
    NASA_POWER_URL,
    SOILING_FACTORS,
    REGION_CLIMATE,
)


def fetch_solar_radiation_data(lat: float, lon: float, days: int = 7, resolution: str = "hourly") -> Optional[Dict[str, Any]]:
    """
    Fetch solar radiation forecast from Open-Meteo API.
    
    Args:
        lat: Latitude
        lon: Longitude
        days: Forecast days
        resolution: 'hourly', '15min', or 'daily'
    
    Returns:
        Dictionary with solar radiation data
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
            "global_tilted_irradiance",
            "terrestrial_radiation",
            "shortwave_radiation",
            "shortwave_radiation_instant",
        ],
        "daily": [
            "shortwave_radiation_sum",
            "sunrise",
            "sunset",
            "sunshine_duration",
            "daylight_duration",
        ],
        "timezone": "GMT",
        "forecast_days": min(days, 16),
    }

    # Add 15-minute solar radiation parameters when sub-hourly is requested
    if resolution == "15min":
        params["minutely_15"] = [
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
            "global_tilted_irradiance",
            "shortwave_radiation",
            "shortwave_radiation_instant",
            "direct_radiation_instant",
            "diffuse_radiation_instant",
            "direct_normal_irradiance_instant",
        ]
    
    try:
        response = requests.get(f"{OPEN_METEO_BASE_URL}/forecast", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        data["_resolution"] = resolution
        return data
    except requests.RequestException as e:
        print(f"Error fetching solar data: {e}")
        return None


def fetch_nasa_power_data(
    lat: float,
    lon: float,
    start_date: str = None,
    end_date: str = None,
    parameters: List[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Fetch historical solar resource data from NASA POWER API.
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date (YYYYMMDD format)
        end_date: End date (YYYYMMDD format)
        parameters: List of NASA POWER parameters to fetch
    
    Returns:
        Dictionary with NASA POWER data
    """
    if start_date is None:
        start_date = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y%m%d")
    if end_date is None:
        end_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")
    
    if parameters is None:
        parameters = [
            "ALLSKY_SFC_SW_DWN",      # All Sky Surface Shortwave Downward Irradiance
            "CLRSKY_SFC_SW_DWN",      # Clear Sky Surface Shortwave Downward Irradiance
            "ALLSKY_SFC_SW_DNI",      # Direct Normal Irradiance
            "ALLSKY_SFC_SW_DIFF",     # Diffuse Irradiance
            "ALLSKY_KT",              # All Sky Insolation Clearness Index
            "T2M",                     # Temperature at 2 Meters
            "T2M_MAX",                 # Maximum Temperature
            "T2M_MIN",                 # Minimum Temperature
            "RH2M",                    # Relative Humidity
            "PRECTOTCORR",            # Precipitation
            "WS10M",                   # Wind Speed at 10 Meters
        ]
    
    params = {
        "parameters": ",".join(parameters),
        "community": "RE",  # Renewable Energy
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }
    
    try:
        response = requests.get(NASA_POWER_URL, params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching NASA POWER data: {e}")
        return None


def process_solar_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process raw solar radiation data into a usable format."""
    if not raw_data:
        return {}
    
    processed = {
        "hourly": None,
        "daily": None,
        "minutely_15": None,
        "daily_agg": None,
        "_resolution": raw_data.get("_resolution", "hourly"),
        "location": {
            "latitude": raw_data.get("latitude"),
            "longitude": raw_data.get("longitude"),
            "timezone": raw_data.get("timezone"),
        }
    }
    
    if "hourly" in raw_data:
        hourly = raw_data["hourly"]
        processed["hourly"] = pd.DataFrame({
            "time": pd.to_datetime(hourly.get("time", [])),
            "direct_radiation": hourly.get("direct_radiation", []),
            "diffuse_radiation": hourly.get("diffuse_radiation", []),
            "direct_normal_irradiance": hourly.get("direct_normal_irradiance", []),
            "global_tilted_irradiance": hourly.get("global_tilted_irradiance", []),
            "shortwave_radiation": hourly.get("shortwave_radiation", []),
        })
        
        # Calculate Global Horizontal Irradiance (GHI) if not directly available
        processed["hourly"]["ghi"] = (
            processed["hourly"]["direct_radiation"] + 
            processed["hourly"]["diffuse_radiation"]
        )
    
    if "daily" in raw_data:
        daily = raw_data["daily"]
        processed["daily"] = pd.DataFrame({
            "date": pd.to_datetime(daily.get("time", [])),
            "radiation_sum": daily.get("shortwave_radiation_sum", []),
            "sunrise": daily.get("sunrise", []),
            "sunset": daily.get("sunset", []),
            "sunshine_duration": daily.get("sunshine_duration", []),
            "daylight_duration": daily.get("daylight_duration", []),
        })
        
        # Calculate peak sun hours (PSH)
        # radiation_sum from Open-Meteo is in MJ/m²; convert to kWh/m² (÷3.6)
        # PSH = kWh/m² at STC (1 kW/m²)
        processed["daily"]["radiation_kwh"] = (
            processed["daily"]["radiation_sum"] / 3.6
        )
        processed["daily"]["peak_sun_hours"] = (
            processed["daily"]["radiation_kwh"]
        )

    # ----- Sub-hourly (15-minute) solar DataFrame -----
    if "minutely_15" in raw_data:
        m15 = raw_data["minutely_15"]
        processed["minutely_15"] = pd.DataFrame({
            "time": pd.to_datetime(m15.get("time", [])),
            "direct_radiation": m15.get("direct_radiation", []),
            "diffuse_radiation": m15.get("diffuse_radiation", []),
            "direct_normal_irradiance": m15.get("direct_normal_irradiance", []),
            "global_tilted_irradiance": m15.get("global_tilted_irradiance", []),
            "shortwave_radiation": m15.get("shortwave_radiation", []),
        })
        processed["minutely_15"]["ghi"] = (
            processed["minutely_15"]["direct_radiation"]
            + processed["minutely_15"]["diffuse_radiation"]
        )

    # ----- Daily aggregation of hourly solar data -----
    if isinstance(processed.get("hourly"), pd.DataFrame) and not processed["hourly"].empty:
        h = processed["hourly"].copy()
        h["_date"] = h["time"].dt.date
        agg = h.groupby("_date").agg(
            direct_radiation=("direct_radiation", "mean"),
            diffuse_radiation=("diffuse_radiation", "mean"),
            direct_normal_irradiance=("direct_normal_irradiance", "mean"),
            global_tilted_irradiance=("global_tilted_irradiance", "mean"),
            shortwave_radiation=("shortwave_radiation", "mean"),
            ghi=("ghi", "mean"),
        ).reset_index()
        agg.rename(columns={"_date": "time"}, inplace=True)
        agg["time"] = pd.to_datetime(agg["time"])
        processed["daily_agg"] = agg

    return processed


def calculate_soiling_loss(
    region_name: str = None,
    climate_type: str = None,
    days_since_cleaning: int = 30,
    rainfall_mm: float = 0,
) -> Dict[str, float]:
    """
    Calculate soiling loss for solar panels.
    
    Args:
        region_name: Name of the region (from predefined list)
        climate_type: Climate type if region not in predefined list
        days_since_cleaning: Days since last panel cleaning
        rainfall_mm: Recent rainfall in mm (rain can partially clean panels)
    
    Returns:
        Dictionary with soiling loss estimates
    """
    # Determine base soiling rate
    if region_name and region_name in REGION_CLIMATE:
        climate = REGION_CLIMATE[region_name]
    elif climate_type:
        climate = climate_type
    else:
        climate = "semi_arid"  # Default assumption
    
    base_soiling_rate = SOILING_FACTORS.get(climate, 0.08)
    
    # Daily soiling accumulation rate (simplified linear model)
    daily_rate = base_soiling_rate / 30  # Assume base rate is monthly
    
    # Calculate accumulated soiling
    accumulated_soiling = daily_rate * days_since_cleaning
    
    # Rain cleaning effect (partial cleaning based on rainfall)
    if rainfall_mm > 5:
        rain_cleaning = min(0.8, rainfall_mm / 50)  # Up to 80% cleaning with heavy rain
        accumulated_soiling *= (1 - rain_cleaning)
    
    # Cap maximum soiling loss
    accumulated_soiling = min(accumulated_soiling, 0.30)  # Max 30% loss
    
    return {
        "soiling_loss_percent": accumulated_soiling * 100,
        "transmission_factor": 1 - accumulated_soiling,
        "climate_type": climate,
        "base_monthly_soiling": base_soiling_rate * 100,
        "days_since_cleaning": days_since_cleaning,
        "recommended_cleaning_interval": estimate_cleaning_interval(climate),
    }


def estimate_cleaning_interval(climate_type: str) -> int:
    """Estimate recommended cleaning interval in days based on climate."""
    intervals = {
        "desert": 14,
        "semi_arid": 21,
        "tropical": 45,
        "coastal": 30,
        "urban": 30,
    }
    return intervals.get(climate_type, 30)


def calculate_solar_potential(
    ghi_kwh_m2_day: float,
    panel_area_m2: float = 100,
    panel_efficiency: float = 0.20,
    system_losses: float = 0.14,
    soiling_loss: float = 0.05,
    temperature_coefficient: float = -0.004,
    ambient_temp: float = 25,
    noct: float = 45,
) -> Dict[str, float]:
    """
    Calculate solar energy generation potential.
    
    Args:
        ghi_kwh_m2_day: Global Horizontal Irradiance in kWh/m²/day
        panel_area_m2: Total panel area in m²
        panel_efficiency: Panel efficiency (0-1)
        system_losses: System losses (inverter, wiring, etc.)
        soiling_loss: Soiling loss factor
        temperature_coefficient: Power temperature coefficient (%/°C)
        ambient_temp: Ambient temperature in °C
        noct: Nominal Operating Cell Temperature
    
    Returns:
        Dictionary with energy production estimates
    """
    # Calculate cell temperature
    # Simplified model: Tcell = Tambient + (NOCT - 20) * (Irradiance / 800)
    irradiance_wm2 = ghi_kwh_m2_day * 1000 / 5  # Approximate peak hours
    cell_temp = ambient_temp + (noct - 20) * (irradiance_wm2 / 800)
    
    # Temperature derating
    temp_derating = 1 + temperature_coefficient * (cell_temp - 25)
    
    # Calculate energy production
    gross_energy = ghi_kwh_m2_day * panel_area_m2 * panel_efficiency
    
    # Apply losses
    net_energy = gross_energy * (1 - system_losses) * (1 - soiling_loss) * temp_derating
    
    return {
        "gross_energy_kwh_day": gross_energy,
        "net_energy_kwh_day": net_energy,
        "monthly_energy_kwh": net_energy * 30,
        "annual_energy_kwh": net_energy * 365,
        "cell_temperature": cell_temp,
        "temperature_derating": temp_derating,
        "capacity_factor": net_energy / (panel_area_m2 * panel_efficiency * 24),
        "specific_yield_kwh_kwp": net_energy / (panel_area_m2 * panel_efficiency),
    }


def get_solar_resource_class(ghi_kwh_m2_day: float) -> str:
    """Classify solar resource based on daily GHI."""
    if ghi_kwh_m2_day < 3.0:
        return "Poor"
    elif ghi_kwh_m2_day < 4.0:
        return "Moderate"
    elif ghi_kwh_m2_day < 5.0:
        return "Good"
    elif ghi_kwh_m2_day < 6.0:
        return "Very Good"
    elif ghi_kwh_m2_day < 7.0:
        return "Excellent"
    else:
        return "Outstanding"


def calculate_optimal_tilt(latitude: float) -> Dict[str, float]:
    """
    Calculate optimal panel tilt angles for different scenarios.
    
    Args:
        latitude: Location latitude
    
    Returns:
        Dictionary with recommended tilt angles
    """
    abs_lat = abs(latitude)
    
    return {
        "annual_optimal": abs_lat,  # General rule of thumb
        "summer_optimal": abs_lat - 15,  # Lower tilt for summer
        "winter_optimal": abs_lat + 15,  # Higher tilt for winter
        "fixed_tilt_recommendation": abs_lat * 0.9,  # Slightly less than latitude
        "tracking_benefit_percent": estimate_tracking_benefit(abs_lat),
    }


def estimate_tracking_benefit(latitude: float) -> float:
    """Estimate the benefit of solar tracking vs fixed mount."""
    # Single-axis tracking typically adds 20-30%
    # Benefit varies slightly with latitude
    base_benefit = 25
    lat_adjustment = (latitude - 30) * 0.1  # Adjust based on deviation from 30°
    return base_benefit + lat_adjustment
