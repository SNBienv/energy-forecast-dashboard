# =============================================================================
# Renewable Energy & Weather Dashboard - Configuration
# =============================================================================

# Predefined regions with coordinates
REGIONS = {
    "Kinshasa": {"lat": -4.4419, "lon": 15.2663, "country": "DR Congo", "timezone": "Africa/Kinshasa"},
    "Benguerir": {"lat": 32.2333, "lon": -7.9500, "country": "Morocco", "timezone": "Africa/Casablanca"},
    "Casablanca": {"lat": 33.5731, "lon": -7.5898, "country": "Morocco", "timezone": "Africa/Casablanca"},
    "Errachidia": {"lat": 31.9314, "lon": -4.4288, "country": "Morocco", "timezone": "Africa/Casablanca"},
    "Dakhla": {"lat": 23.6848, "lon": -15.9580, "country": "Morocco", "timezone": "Africa/Casablanca"},
    "Porto-Novo": {"lat": 6.4969, "lon": 2.6289, "country": "Benin", "timezone": "Africa/Porto-Novo"},
    "Abidjan": {"lat": 5.3600, "lon": -4.0083, "country": "Ivory Coast", "timezone": "Africa/Abidjan"},
    "Meskala": {"lat": 31.3500, "lon": -9.7667, "country": "Morocco", "timezone": "Africa/Casablanca"},
    "Goma": {"lat": -1.6771, "lon": 29.2386, "country": "DR Congo", "timezone": "Africa/Lubumbashi"},
    "Bukavu": {"lat": -2.5083, "lon": 28.8608, "country": "DR Congo", "timezone": "Africa/Lubumbashi"},
}

# API Configuration (Free APIs - no keys required for basic usage)
# Open-Meteo: Free weather API
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1"
OPEN_METEO_WEATHER_URL = f"{OPEN_METEO_BASE_URL}/forecast"
OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# NASA POWER API for solar data
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Soiling factors by region type (simplified model)
SOILING_FACTORS = {
    "desert": 0.15,      # High soiling in desert regions
    "semi_arid": 0.10,   # Moderate soiling
    "tropical": 0.05,    # Lower soiling due to rain
    "coastal": 0.07,     # Salt spray considerations
    "urban": 0.08,       # Urban pollution
}

# Region climate classifications for soiling estimation
REGION_CLIMATE = {
    "Kinshasa": "tropical",
    "Benguerir": "semi_arid",
    "Casablanca": "coastal",
    "Errachidia": "desert",
    "Dakhla": "desert",
    "Porto-Novo": "tropical",
    "Abidjan": "coastal",
    "Meskala": "semi_arid",
    "Goma": "tropical",
    "Bukavu": "tropical",
}

# Weather alert thresholds
ALERT_THRESHOLDS = {
    "wind_speed_warning": 50,      # km/h
    "wind_speed_danger": 80,       # km/h
    "temp_high_warning": 40,       # °C
    "temp_low_warning": 5,         # °C
    "uv_warning": 8,               # UV Index
    "uv_danger": 11,               # UV Index
    "precipitation_heavy": 20,     # mm/h
    "aqi_moderate": 51,            # Air Quality Index
    "aqi_unhealthy": 101,
    "aqi_very_unhealthy": 151,
    "aqi_hazardous": 201,
}

# Map styling
MAP_TILES = {
    "OpenStreetMap": "OpenStreetMap",
    "Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Terrain": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
}

# Color schemes for visualizations
COLORS = {
    "wind": "#3498db",
    "solar": "#f39c12",
    "temperature": "#e74c3c",
    "humidity": "#2ecc71",
    "rain": "#9b59b6",
    "air_quality_good": "#00e400",
    "air_quality_moderate": "#ffff00",
    "air_quality_unhealthy": "#ff7e00",
    "air_quality_very_unhealthy": "#ff0000",
    "air_quality_hazardous": "#7e0023",
}
