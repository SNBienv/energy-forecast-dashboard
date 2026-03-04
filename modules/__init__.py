# Renewable Energy Dashboard - Modules Package

from modules.weather_wind import (
    fetch_weather_data,
    process_weather_data,
    get_weather_code_description,
    get_wind_direction_name,
    calculate_wind_power_density,
    estimate_wind_energy_potential,
    classify_wind_resource,
)

from modules.solar_energy import (
    fetch_solar_radiation_data,
    fetch_nasa_power_data,
    process_solar_data,
    calculate_soiling_loss,
    calculate_solar_potential,
    get_solar_resource_class,
    calculate_optimal_tilt,
)

from modules.extreme_weather import (
    analyze_weather_alerts,
    analyze_renewable_energy_impact,
    generate_forecast_summary,
    AlertLevel,
    AlertType,
    get_alert_color,
)

from modules.air_quality import (
    fetch_air_quality_data,
    process_air_quality_data,
    get_aqi_category,
    get_aqi_color,
    get_health_message,
    analyze_air_quality_for_solar,
    generate_air_quality_alerts,
    get_pollutant_info,
)

from modules.map_visualization import (
    create_base_map,
    create_comprehensive_map,
    create_regions_overview_map,
    get_map_html,
    add_location_marker,
    add_morocco_boundary,
    create_wind_rose_data,
)

from modules.forecasting import (
    persistence_forecast,
    sarimax_forecast,
    xgboost_forecast,
    lstm_forecast,
    ensemble_forecast,
    compute_accuracy_metrics,
    api_forecast_from_weather,
    available_methods,
    run_forecast_comparison,
)

__all__ = [
    # Weather/Wind
    "fetch_weather_data",
    "process_weather_data",
    "get_weather_code_description",
    "calculate_wind_power_density",
    "estimate_wind_energy_potential",
    # Solar
    "fetch_solar_radiation_data",
    "process_solar_data",
    "calculate_soiling_loss",
    "calculate_solar_potential",
    "calculate_optimal_tilt",
    # Alerts
    "analyze_weather_alerts",
    "analyze_renewable_energy_impact",
    "AlertLevel",
    # Air Quality
    "fetch_air_quality_data",
    "process_air_quality_data",
    "analyze_air_quality_for_solar",
    "generate_air_quality_alerts",
    # Maps
    "create_comprehensive_map",
    "create_regions_overview_map",
    # Forecasting
    "ensemble_forecast",
    "run_forecast_comparison",
    "compute_accuracy_metrics",
]
