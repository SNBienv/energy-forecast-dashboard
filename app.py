# =============================================================================
# Renewable Energy & Weather Dashboard
# Main Streamlit Application
# =============================================================================
# 
# A comprehensive dashboard for:
# - Wind energy resource assessment
# - Solar energy potential with soiling analysis
# - Weather forecasting and extreme event alerts
# - Air quality monitoring
# - Interactive map visualization
#
# Target Regions: Kinshasa, Benguerir, Casablanca, Errachidia, Dakhla,
#                 Porto-Novo, Abidjan, Meskala, Goma, Bukavu
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import traceback
import folium
from streamlit_folium import st_folium

# Import our modules
from config import REGIONS, REGION_CLIMATE, ALERT_THRESHOLDS, COLORS
from modules.weather_wind import (
    fetch_weather_data,
    process_weather_data,
    estimate_wind_energy_potential,
    get_weather_code_description,
)
from modules.solar_energy import (
    fetch_solar_radiation_data,
    process_solar_data,
    calculate_soiling_loss,
    calculate_solar_potential,
    calculate_optimal_tilt,
    get_solar_resource_class,
)
from modules.extreme_weather import (
    analyze_weather_alerts,
    analyze_renewable_energy_impact,
    generate_forecast_summary,
    get_alert_color,
)
from modules.air_quality import (
    fetch_air_quality_data,
    process_air_quality_data,
    analyze_air_quality_for_solar,
    generate_air_quality_alerts,
    get_pollutant_info,
)
from modules.map_visualization import (
    create_comprehensive_map,
    create_regions_overview_map,
    create_town_map,
    create_wind_resource_map,
    create_solar_resource_map,
)
from modules.forecasting import (
    run_forecast_comparison,
    available_methods,
)
from modules.wind_resource import (
    TURBINE_CATALOG,
    MOROCCO_SITES,
    weibull_fit,
    weibull_pdf,
    extrapolate_wind_speed,
    power_curve as wind_power_curve,
    capacity_factor as wind_cf,
    wind_power_density,
    bastankhah_wake_deficit,
    bastankhah_wake_field,
    compute_farm_wake,
    farm_aep,
    generate_grid_layout,
    generate_staggered_layout,
    assess_wind_resource,
    compute_lcoe as wind_lcoe,
    compute_npv as wind_npv,
    estimate_capex_per_kw,
    iec_wind_class,
)
from modules.solar_resource import (
    PV_CATALOG,
    MOROCCO_SOLAR_SITES,
    optimal_tilt_angle as sr_optimal_tilt,
    assess_solar_resource,
    compute_solar_lcoe,
    estimate_monthly_profile,
    morocco_soiling_loss,
    pv_power_output,
    cell_temperature,
)
from modules.hybrid_hydrogen import (
    ELECTROLYSER_CATALOG,
    MOROCCO_H2_ZONES,
    MOROCCO_H2_CONTEXT,
    hybrid_dispatch,
    compute_lcoh,
    quick_hybrid_sizing,
    complementarity_index,
    morocco_roadmap_comparison,
    green_ammonia_production,
)
from modules.interpolation import (
    interpolate_dataframe,
    step_function_df,
)
from modules.copernicus_data import (
    fetch_weather_data_cds,
    fetch_solar_data_cds,
    cds_available,
)

# Default plotly template for dark mode
import plotly.io as pio
pio.templates.default = "plotly_dark"


# --- Resolution helpers -------------------------------------------------------
_RESOLUTION_MAP = {
    "Hourly":              {"resolution": "hourly", "freq": "h",     "label": "Hourly"},
    "Sub-hourly (15 min)": {"resolution": "15min",  "freq": "15min", "label": "15-Minute"},
    "Sub-hourly (5 min)":  {"resolution": "5min",   "freq": "5min",  "label": "5-Minute"},
    "Sub-hourly (1 min)":  {"resolution": "1min",   "freq": "1min",  "label": "1-Minute"},
    "Daily":               {"resolution": "daily",  "freq": "D",     "label": "Daily"},
}

# Cache for interpolated DataFrames so they are computed only once per session
_INTERP_CACHE_KEY = "_interp_cache"


def get_ts(data_dict, resolution, lat: float = 20.0, lon: float = 0.0):
    """Return the time-series DataFrame matching the selected resolution.

    For 5-min and 1-min resolutions the hourly data is up-sampled using
    physics-based spectral interpolation (Von-Kármán for wind, harmonic
    diurnal for temperature, Clausius-Clapeyron for humidity, etc.).
    """
    if not data_dict:
        return None
    if resolution == "15min":
        df = data_dict.get("minutely_15")
        if df is not None and not df.empty:
            return df
    elif resolution == "daily":
        df = data_dict.get("daily_agg")
        if df is not None and not df.empty:
            return df
    elif resolution in ("5min", "1min"):
        # Physics-based upsampling from hourly data
        base = data_dict.get("hourly")
        if base is None or base.empty:
            return None
        cache = st.session_state.setdefault(_INTERP_CACHE_KEY, {})
        cache_key = (id(data_dict), resolution)
        if cache_key in cache:
            return cache[cache_key]
        interp_df = interpolate_dataframe(
            base, target_freq=resolution, lat=lat, lon=lon,
        )
        cache[cache_key] = interp_df
        return interp_df
    return data_dict.get("hourly")


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Renewable Energy & Weather Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling (dark & light theme compatible)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #0e4525 0%, #1a5632 50%, #0e4525 100%);
        color: #ffffff !important;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    /* Metric cards – theme-aware */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.80) !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 600;
    }
    [data-testid="stMetricDelta"] > div {
        color: #4fc978 !important;
    }
    .alert-danger {
        background-color: rgba(231, 76, 60, 0.15);
        border-left: 4px solid #e74c3c;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #f8d7da;
    }
    .alert-warning {
        background-color: rgba(243, 156, 18, 0.15);
        border-left: 4px solid #f39c12;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #fff3cd;
    }
    .alert-info {
        background-color: rgba(52, 152, 219, 0.15);
        border-left: 4px solid #3498db;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #d1ecf1;
    }
    /* Forecast day cards */
    .forecast-card {
        text-align: center;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.10);
        padding: 1rem;
        border-radius: 10px;
    }
    .forecast-card strong {
        color: #ffffff;
    }
    /* Region cards on overview page */
    .region-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.10);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        color: #e0e0e0;
    }
    .region-card strong {
        color: #ffffff;
    }
    /* AQI box */
    .aqi-box {
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
    }
    /* Tables */
    .stTable, .stDataFrame {
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================

if "selected_region" not in st.session_state:
    st.session_state.selected_region = None
if "weather_data" not in st.session_state:
    st.session_state.weather_data = None
if "solar_data" not in st.session_state:
    st.session_state.solar_data = None
if "air_quality_data" not in st.session_state:
    st.session_state.air_quality_data = None
if "custom_coords" not in st.session_state:
    st.session_state.custom_coords = None
if "time_resolution" not in st.session_state:
    st.session_state.time_resolution = "hourly"


# =============================================================================
# Sidebar - Location Selection
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/wind-turbine.png", width=80)
    st.title("🌍 Location Selection")
    
    # Selection method
    selection_method = st.radio(
        "Select Location Method",
        ["Predefined Regions", "Custom Coordinates"],
        help="Choose a predefined region or enter custom coordinates"
    )
    
    if selection_method == "Predefined Regions":
        selected_name = st.selectbox(
            "Select Region",
            list(REGIONS.keys()),
            help="Choose from predefined African regions"
        )
        
        if selected_name:
            region = REGIONS[selected_name]
            lat = region["lat"]
            lon = region["lon"]
            st.info(f"📍 {selected_name}, {region['country']}")
            st.caption(f"Coordinates: {lat:.4f}°, {lon:.4f}°")
    else:
        st.subheader("Enter Coordinates")
        lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0, step=0.01)
        lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0, step=0.01)
        selected_name = f"Custom ({lat:.2f}, {lon:.2f})"
    
    st.divider()
    
    # Forecast settings
    st.subheader("⚙️ Settings")
    forecast_days = st.slider("Forecast Days", 1, 14, 7)

    # ---- Data source selector ----
    _cds_ok = cds_available()
    _source_options = ["Open-Meteo (free, forecast)"]
    if _cds_ok:
        _source_options.append("Copernicus ERA5 (reanalysis)")
    data_source_label = st.selectbox(
        "📡 Data Source",
        _source_options,
        help=(
            "Open-Meteo: real-time forecast, no key needed.\n"
            "Copernicus ERA5: high-quality reanalysis (historical), "
            "requires .cdsapirc credentials."
            + ("" if _cds_ok else "\n\n⚠️ cdsapi not configured — install cdsapi and create ~/.cdsapirc")
        ),
    )
    data_source = "copernicus" if "Copernicus" in data_source_label else "openmeteo"

    time_res_label = st.selectbox(
        "⏱️ Time Resolution",
        list(_RESOLUTION_MAP.keys()),
        help="Choose temporal resolution for data and forecasts",
    )
    res_config = _RESOLUTION_MAP[time_res_label]
    resolution = res_config["resolution"]
    freq = res_config["freq"]
    res_label = res_config["label"]
    
    # Panel configuration for solar calculations
    with st.expander("☀️ Solar Panel Settings"):
        panel_area = st.number_input("Panel Area (m²)", min_value=1, max_value=10000, value=100)
        panel_efficiency = st.slider("Panel Efficiency (%)", 10, 25, 20) / 100
        days_since_cleaning = st.number_input("Days Since Cleaning", min_value=0, max_value=365, value=30)
    
    # Wind turbine configuration
    with st.expander("💨 Wind Turbine Settings"):
        turbine_capacity = st.number_input("Turbine Capacity (kW)", min_value=1, max_value=10000, value=2000)
        rotor_diameter = st.number_input("Rotor Diameter (m)", min_value=10, max_value=200, value=90)
    
    st.divider()
    
    # Fetch data button
    fetch_button = st.button("🔄 Fetch Data", type="primary", width='stretch')
    
    st.divider()
    st.caption("Data Sources:")
    st.caption("• Open-Meteo (Weather & AQ)")
    st.caption("• Copernicus ERA5 (Reanalysis)")
    st.caption("• NASA POWER (Solar)")
    st.caption("© 2024 RE Dashboard")


# =============================================================================
# Data Fetching
# =============================================================================

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_all_data(lat: float, lon: float, days: int, resolution: str = "hourly",
                  source: str = "openmeteo"):
    """Fetch all data for a location from the selected source."""
    if source == "copernicus":
        weather_raw = fetch_weather_data_cds(lat, lon, days, resolution=resolution)
        solar_raw = fetch_solar_data_cds(lat, lon, days, resolution=resolution)
    else:
        weather_raw = fetch_weather_data(lat, lon, days, resolution=resolution)
        solar_raw = fetch_solar_radiation_data(lat, lon, days, resolution=resolution)

    weather = process_weather_data(weather_raw) if weather_raw else {}
    solar = process_solar_data(solar_raw) if solar_raw else {}

    # Air quality is always from Open-Meteo (not in ERA5)
    aq_raw = fetch_air_quality_data(lat, lon, min(days, 5))
    air_quality = process_air_quality_data(aq_raw) if aq_raw else {}
    
    return weather, solar, air_quality


# Fetch data when button is clicked
if fetch_button and lat and lon:
    _spinner_msg = ("🔄 Fetching data from Copernicus ERA5…" if data_source == "copernicus"
                    else "🔄 Fetching data from Open-Meteo…")
    with st.spinner(_spinner_msg):
        weather_data, solar_data, air_quality_data = load_all_data(
            lat, lon, forecast_days, resolution, source=data_source)
        st.session_state.weather_data = weather_data
        st.session_state.solar_data = solar_data
        st.session_state.air_quality_data = air_quality_data
        st.session_state.selected_region = selected_name
        st.session_state.time_resolution = resolution
        st.session_state.data_source = data_source
        _src_label = "Copernicus ERA5" if data_source == "copernicus" else "Open-Meteo"
        st.success(f"✅ Data loaded from {_src_label}!")


# =============================================================================
# Main Content
# =============================================================================

st.markdown('<h1 class="main-header">🌍 Renewable Energy & Weather Dashboard</h1>', unsafe_allow_html=True)
st.caption(f"🕐 All times in GMT — Current: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} GMT")


# Display overview map if no data selected
if st.session_state.weather_data is None:
    st.markdown("### 📍 Coverage Map - Target Regions")
    st.info("👈 Select a region from the sidebar and click 'Fetch Data' to view detailed analysis.")
    
    # Show overview map
    overview_map = create_regions_overview_map()
    st_folium(overview_map, height=500, use_container_width=True)
    
    # Region cards
    st.markdown("### 📋 Available Regions")
    cols = st.columns(5)
    for i, (name, info) in enumerate(REGIONS.items()):
        with cols[i % 5]:
            st.markdown(f"""
            <div class="region-card">
                <strong>{name}</strong><br>
                <small>{info['country']}</small><br>
                <small style="color: #adb5bd;">{info['lat']:.2f}°, {info['lon']:.2f}°</small>
            </div>
            """, unsafe_allow_html=True)

else:
    # ==========================================================================
    # Dashboard with fetched data
    # ==========================================================================
    
    weather_data = st.session_state.weather_data
    solar_data = st.session_state.solar_data
    air_quality_data = st.session_state.air_quality_data
    location_name = st.session_state.selected_region
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "🌤️ Weather", "💨 Wind Energy", "☀️ Solar Energy", 
        "🌬️ Air Quality", "⚠️ Alerts", "🗺️ Map", "📊 Forecast Comparison",
        "🌀 Wind Resource", "🔆 Solar Resource", "⚡ Hybrid & H₂",
    ])
    
    # ==========================================================================
    # TAB 1: Current Weather & Forecast
    # ==========================================================================
    with tab1:
      try:
        st.header(f"🌤️ Weather - {location_name}")
        
        if weather_data and "current" in weather_data:
            current = weather_data["current"]
            
            # Current conditions cards
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "🌡️ Temperature",
                    f"{current.get('temperature', 'N/A')}°C",
                    f"Feels like {current.get('feels_like', 'N/A')}°C"
                )
            with col2:
                st.metric(
                    "💧 Humidity",
                    f"{current.get('humidity', 'N/A')}%"
                )
            with col3:
                st.metric(
                    "💨 Wind",
                    f"{current.get('wind_speed', 'N/A')} km/h",
                    f"Gusts: {current.get('wind_gusts', 'N/A')} km/h"
                )
            with col4:
                st.metric(
                    "☁️ Cloud Cover",
                    f"{current.get('cloud_cover', 'N/A')}%"
                )
            with col5:
                st.metric(
                    "📊 Pressure",
                    f"{current.get('pressure', 'N/A')} hPa"
                )
            
            # Current conditions description
            weather_info = current.get("weather_description", "")
            weather_icon = current.get("weather_icon", "")
            st.markdown(f"### {weather_icon} {weather_info}")
            
            st.divider()
            
            # Daily forecast
            if "daily" in weather_data and weather_data["daily"] is not None:
                st.subheader("📅 7-Day Forecast")
                daily = weather_data["daily"]
                
                # Create forecast cards
                cols = st.columns(min(7, len(daily)))
                for i, (_, row) in enumerate(daily.head(7).iterrows()):
                    with cols[i]:
                        day_name = row["date"].strftime("%a")
                        day_weather = get_weather_code_description(row.get("weather_code", 0))
                        st.markdown(f"""
                        <div class="forecast-card">
                            <strong>{day_name}</strong><br>
                            <span style="font-size: 2rem;">{day_weather['icon']}</span><br>
                            <span style="color: #ff6b6b;">{row['temp_max']:.0f}°</span>
                            <span style="color: #74b9ff;">{row['temp_min']:.0f}°</span><br>
                            <small>💨 {row['wind_speed_max']:.0f} km/h</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.divider()
            
            # Hourly charts
            ts = get_ts(weather_data, resolution, lat=lat, lon=lon)
            if ts is not None:
                st.subheader(f"📈 {res_label} Forecast")
                hourly = ts
                
                # Temperature and precipitation chart
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(
                        x=hourly["time"],
                        y=hourly["temperature"],
                        name="Temperature (°C)",
                        line=dict(color=COLORS["temperature"], width=2)
                    ),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Bar(
                        x=hourly["time"],
                        y=hourly["precipitation"],
                        name="Precipitation (mm)",
                        marker_color=COLORS["rain"],
                        opacity=0.6
                    ),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title="Temperature & Precipitation Forecast",
                    xaxis_title="Time (GMT)",
                    height=400,
                    hovermode="x unified"
                )
                fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
                fig.update_yaxes(title_text="Precipitation (mm)", secondary_y=True)
                
                st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Weather data not available. Please try fetching again.")
      except Exception as _tab1_err:
        st.error(f"Error rendering Weather tab: {_tab1_err}")
    
    # ==========================================================================
    # TAB 2: Wind Energy
    # ==========================================================================
    with tab2:
      try:
        st.header(f"💨 Wind Energy Analysis - {location_name}")
        
        ts2 = get_ts(weather_data, resolution, lat=lat, lon=lon)
        if weather_data and ts2 is not None:
            hourly = ts2
            current = weather_data.get("current", {})
            
            # Calculate wind energy metrics
            avg_wind = hourly["wind_speed"].mean()
            max_wind = hourly["wind_speed"].max()
            current_wind = current.get("wind_speed", avg_wind)
            
            wind_potential = estimate_wind_energy_potential(
                avg_wind,
                turbine_capacity=turbine_capacity,
                rotor_diameter=rotor_diameter
            )
            
            # Wind metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💨 Current Wind", f"{current_wind:.1f} km/h")
            with col2:
                st.metric("📊 Average Wind", f"{avg_wind:.1f} km/h")
            with col3:
                st.metric("⚡ Power Density", f"{wind_potential['power_density_wm2']:.0f} W/m²")
            with col4:
                st.metric("🏭 Est. Daily Energy", f"{wind_potential['estimated_daily_kwh']:.0f} kWh")
            
            # Wind class assessment
            wind_class = wind_potential["wind_class"]
            st.info(f"**Wind Resource Classification:** {wind_class}")
            
            st.divider()
            
            # Wind speed chart
            col1, col2 = st.columns(2)
            
            with col1:
                fig_wind = go.Figure()
                fig_wind.add_trace(go.Scatter(
                    x=hourly["time"],
                    y=hourly["wind_speed"],
                    name="Wind Speed",
                    fill="tozeroy",
                    line=dict(color=COLORS["wind"])
                ))
                fig_wind.add_trace(go.Scatter(
                    x=hourly["time"],
                    y=hourly["wind_gusts"],
                    name="Wind Gusts",
                    line=dict(color="#e74c3c", dash="dot")
                ))
                fig_wind.update_layout(
                    title="Wind Speed Forecast",
                    xaxis_title="Time (GMT)",
                    yaxis_title="Wind Speed (km/h)",
                    height=400
                )
                st.plotly_chart(fig_wind, width="stretch")
            
            with col2:
                # Wind direction rose (simplified)
                fig_dir = px.histogram(
                    hourly,
                    x="wind_direction",
                    nbins=16,
                    title="Wind Direction Distribution",
                    labels={"wind_direction": "Direction (°)"}
                )
                fig_dir.update_layout(height=400)
                st.plotly_chart(fig_dir, width="stretch")
            
            # Wind energy production estimate table
            st.subheader("⚡ Energy Production Estimates")
            
            estimates = {
                "Metric": [
                    "Average Wind Speed",
                    "Power Density",
                    "Capacity Factor",
                    "Daily Production (est.)",
                    "Monthly Production (est.)",
                    "Annual Production (est.)",
                ],
                "Value": [
                    f"{avg_wind:.1f} km/h ({avg_wind/3.6:.1f} m/s)",
                    f"{wind_potential['power_density_wm2']:.1f} W/m²",
                    f"{wind_potential['capacity_factor']*100:.1f}%",
                    f"{wind_potential['estimated_daily_kwh']:.0f} kWh",
                    f"{wind_potential['estimated_daily_kwh']*30:.0f} kWh",
                    f"{wind_potential['estimated_daily_kwh']*365:.0f} kWh",
                ]
            }
            st.table(pd.DataFrame(estimates))
        else:
            st.warning("Wind data not available.")
      except Exception as _tab2_err:
        st.error(f"Error rendering Wind Energy tab: {_tab2_err}")
    
    # ==========================================================================
    # TAB 3: Solar Energy
    # ==========================================================================
    with tab3:
      try:
        st.header(f"☀️ Solar Energy Analysis - {location_name}")
        
        if solar_data and "daily" in solar_data and solar_data["daily"] is not None:
            daily_solar = solar_data["daily"]
            
            # Calculate solar metrics
            # radiation_sum is in MJ/m²; divide by 3.6 to get kWh/m²
            avg_radiation = daily_solar["radiation_sum"].mean() / 3.6
            avg_psh = daily_solar["peak_sun_hours"].mean()
            
            # Soiling calculation
            climate_type = REGION_CLIMATE.get(location_name, "semi_arid")
            soiling = calculate_soiling_loss(
                region_name=location_name,
                days_since_cleaning=days_since_cleaning
            )
            
            # Solar potential
            solar_potential = calculate_solar_potential(
                ghi_kwh_m2_day=avg_radiation,
                panel_area_m2=panel_area,
                panel_efficiency=panel_efficiency,
                soiling_loss=soiling["soiling_loss_percent"] / 100,
                ambient_temp=weather_data.get("current", {}).get("temperature", 25) if weather_data else 25
            )
            
            # Optimal tilt
            optimal_tilt = calculate_optimal_tilt(lat)
            
            # Solar metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("☀️ Avg. Irradiance", f"{avg_radiation:.2f} kWh/m²/day")
            with col2:
                st.metric("🔆 Peak Sun Hours", f"{avg_psh:.1f} hours")
            with col3:
                st.metric("⚡ Daily Output", f"{solar_potential['net_energy_kwh_day']:.0f} kWh")
            with col4:
                resource_class = get_solar_resource_class(avg_radiation)
                st.metric("📊 Resource Class", resource_class)
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Solar radiation chart
                fig_solar = go.Figure()
                fig_solar.add_trace(go.Bar(
                    x=daily_solar["date"],
                    y=daily_solar["radiation_sum"] / 3.6,
                    name="Daily Irradiance",
                    marker_color=COLORS["solar"]
                ))
                fig_solar.update_layout(
                    title="Daily Solar Irradiance",
                    xaxis_title="Date",
                    yaxis_title="Irradiance (kWh/m²)",
                    height=400
                )
                st.plotly_chart(fig_solar, width="stretch")
            
            with col2:
                # Sunshine duration
                fig_sun = go.Figure()
                fig_sun.add_trace(go.Bar(
                    x=daily_solar["date"],
                    y=daily_solar["sunshine_duration"] / 3600,
                    name="Sunshine Duration",
                    marker_color="#f1c40f"
                ))
                fig_sun.add_trace(go.Scatter(
                    x=daily_solar["date"],
                    y=daily_solar["daylight_duration"] / 3600,
                    name="Daylight Duration",
                    line=dict(color="#e67e22", dash="dash")
                ))
                fig_sun.update_layout(
                    title="Sunshine & Daylight Duration",
                    xaxis_title="Date",
                    yaxis_title="Hours",
                    height=400
                )
                st.plotly_chart(fig_sun, width="stretch")
            
            st.divider()
            
            # Soiling Analysis
            st.subheader("🏜️ Soiling Effect Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Climate Type:** {soiling['climate_type'].title()}
                
                **Current Soiling Loss:** {soiling['soiling_loss_percent']:.1f}%
                
                **Panel Transmission:** {soiling['transmission_factor']*100:.1f}%
                
                **Base Monthly Soiling Rate:** {soiling['base_monthly_soiling']:.1f}%
                
                **Recommended Cleaning Interval:** Every {soiling['recommended_cleaning_interval']} days
                """)
            
            with col2:
                # Soiling impact gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=soiling['soiling_loss_percent'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Soiling Loss (%)"},
                    gauge={
                        'axis': {'range': [0, 30]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 5], 'color': "lightgreen"},
                            {'range': [5, 15], 'color': "yellow"},
                            {'range': [15, 30], 'color': "red"}
                        ],
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, width="stretch")
            
            # Production estimates table
            st.subheader("⚡ Energy Production Estimates")
            
            production_data = {
                "Metric": [
                    "Gross Daily Production",
                    "Net Daily Production (after losses)",
                    "Monthly Production",
                    "Annual Production",
                    f"Optimal Panel Tilt",
                    "Capacity Factor",
                    "Specific Yield",
                ],
                "Value": [
                    f"{solar_potential['gross_energy_kwh_day']:.1f} kWh",
                    f"{solar_potential['net_energy_kwh_day']:.1f} kWh",
                    f"{solar_potential['monthly_energy_kwh']:.0f} kWh",
                    f"{solar_potential['annual_energy_kwh']:.0f} kWh",
                    f"{optimal_tilt['annual_optimal']:.0f}°",
                    f"{solar_potential['capacity_factor']*100:.1f}%",
                    f"{solar_potential['specific_yield_kwh_kwp']:.1f} kWh/kWp/day",
                ]
            }
            st.table(pd.DataFrame(production_data))
        else:
            st.warning("Solar data not available.")
      except Exception as _tab3_err:
        st.error(f"Error rendering Solar Energy tab: {_tab3_err}")
    
    # ==========================================================================
    # TAB 4: Air Quality
    # ==========================================================================
    with tab4:
      try:
        st.header(f"🌬️ Air Quality - {location_name}")
        
        if air_quality_data and "current" in air_quality_data:
            current_aq = air_quality_data["current"]
            
            # AQI display
            aqi = current_aq.get("us_aqi") or current_aq.get("european_aqi", 0)
            aqi_color = current_aq.get("aqi_color", "#808080")
            aqi_category = current_aq.get("aqi_category", "Unknown")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Large AQI display
                st.markdown(f"""
                <div class="aqi-box" style="background: {aqi_color}20; border: 3px solid {aqi_color};">
                    <h1 style="font-size: 4rem; color: {aqi_color}; margin: 0;">{aqi:.0f}</h1>
                    <h3 style="color: {aqi_color};">{aqi_category}</h3>
                    <p style="color: #c0c0c0;">US Air Quality Index</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"### Health Advisory")
                st.markdown(current_aq.get("health_message", "No advisory available."))
                
                # Pollutant levels
                st.markdown("### Current Pollutant Levels")
                pollutant_info = get_pollutant_info()
                
                pollutant_data = []
                for key, info in pollutant_info.items():
                    value = current_aq.get(key)
                    if value is not None:
                        status = "✅ Good" if value <= info["who_guideline"] else "⚠️ Elevated"
                        pollutant_data.append({
                            "Pollutant": info["name"],
                            "Value": f"{value:.1f} {info['unit']}",
                            "WHO Guideline": f"{info['who_guideline']} {info['unit']}",
                            "Status": status
                        })
                
                if pollutant_data:
                    st.table(pd.DataFrame(pollutant_data))
            
            st.divider()
            
            # Hourly AQ forecast
            if "hourly" in air_quality_data and air_quality_data["hourly"] is not None:
                hourly_aq = air_quality_data["hourly"]
                
                st.subheader("📈 Air Quality Forecast")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_aqi = go.Figure()
                    fig_aqi.add_trace(go.Scatter(
                        x=hourly_aq["time"],
                        y=hourly_aq["us_aqi"],
                        name="AQI",
                        fill="tozeroy",
                        line=dict(color="#27ae60")
                    ))
                    fig_aqi.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
                    fig_aqi.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Moderate")
                    fig_aqi.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Unhealthy")
                    fig_aqi.update_layout(
                        title="AQI Forecast",
                        xaxis_title="Time (GMT)",
                        yaxis_title="AQI",
                        height=400
                    )
                    st.plotly_chart(fig_aqi, width="stretch")
                
                with col2:
                    fig_pm = go.Figure()
                    fig_pm.add_trace(go.Scatter(
                        x=hourly_aq["time"],
                        y=hourly_aq["pm2_5"],
                        name="PM2.5",
                        line=dict(color="#e74c3c")
                    ))
                    fig_pm.add_trace(go.Scatter(
                        x=hourly_aq["time"],
                        y=hourly_aq["pm10"],
                        name="PM10",
                        line=dict(color="#f39c12")
                    ))
                    fig_pm.update_layout(
                        title="Particulate Matter Forecast",
                        xaxis_title="Time (GMT)",
                        yaxis_title="Concentration (µg/m³)",
                        height=400
                    )
                    st.plotly_chart(fig_pm, width="stretch")
            
            # Air quality impact on solar
            st.subheader("☀️ Impact on Solar Energy")
            aq_impact = analyze_air_quality_for_solar(air_quality_data)
            
            if aq_impact.get("impact_available"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Estimated Solar Impact",
                        f"-{aq_impact['total_estimated_impact_percent']:.1f}%",
                        "Due to atmospheric conditions"
                    )
                with col2:
                    st.metric(
                        "Transmission Efficiency",
                        f"{aq_impact['transmission_efficiency']*100:.1f}%"
                    )
                
                if aq_impact["solar_impact_factors"]:
                    st.markdown("**Impact Factors:**")
                    for factor in aq_impact["solar_impact_factors"]:
                        st.markdown(f"- {factor['message']}")
            else:
                st.info("Detailed air quality impact data not available for this location.")
        else:
            st.warning("Air quality data not available.")
      except Exception as _tab4_err:
        st.error(f"Error rendering Air Quality tab: {_tab4_err}")
    
    # ==========================================================================
    # TAB 5: Weather Alerts
    # ==========================================================================
    with tab5:
      try:
        st.header(f"⚠️ Weather Alerts & Warnings - {location_name}")
        
        # Generate alerts
        weather_alerts = analyze_weather_alerts(weather_data) if weather_data else []
        aq_alerts = generate_air_quality_alerts(air_quality_data) if air_quality_data else []
        
        all_alerts = weather_alerts + aq_alerts
        
        if all_alerts:
            # Group by severity
            danger_alerts = [a for a in all_alerts if a.get("level") in ["danger", "emergency"]]
            warning_alerts = [a for a in all_alerts if a.get("level") == "warning"]
            advisory_alerts = [a for a in all_alerts if a.get("level") in ["advisory", "info"]]
            
            if danger_alerts:
                st.error("### 🚨 Danger Alerts")
                for alert in danger_alerts:
                    st.markdown(f"""
                    <div class="alert-danger">
                        <strong>{alert.get('icon', '⚠️')} {alert.get('title')}</strong><br>
                        {alert.get('message')}
                    </div>
                    """, unsafe_allow_html=True)
            
            if warning_alerts:
                st.warning("### ⚠️ Warnings")
                for alert in warning_alerts:
                    st.markdown(f"""
                    <div class="alert-warning">
                        <strong>{alert.get('icon', '⚠️')} {alert.get('title')}</strong><br>
                        {alert.get('message')}
                    </div>
                    """, unsafe_allow_html=True)
            
            if advisory_alerts:
                st.info("### ℹ️ Advisories")
                for alert in advisory_alerts:
                    st.markdown(f"""
                    <div class="alert-info">
                        <strong>{alert.get('icon', 'ℹ️')} {alert.get('title')}</strong><br>
                        {alert.get('message')}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("✅ No weather alerts at this time. Conditions are favorable.")
        
        st.divider()
        
        # Renewable energy impact assessment
        st.subheader("🔋 Renewable Energy Impact Assessment")
        
        if weather_data:
            impacts = analyze_renewable_energy_impact(weather_data, weather_alerts)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ☀️ Solar System Impacts")
                if impacts["solar"]:
                    for impact in impacts["solar"]:
                        icon = "🟢" if impact["severity"] == "positive" else "🟡" if impact["severity"] == "moderate" else "🔴"
                        st.markdown(f"{icon} {impact['message']}")
                else:
                    st.markdown("✅ No significant impacts on solar generation.")
            
            with col2:
                st.markdown("#### 💨 Wind System Impacts")
                if impacts["wind"]:
                    for impact in impacts["wind"]:
                        icon = "🟢" if impact["severity"] == "positive" else "🟡" if impact["severity"] == "moderate" else "🔴"
                        st.markdown(f"{icon} {impact['message']}")
                else:
                    st.markdown("✅ No significant impacts on wind generation.")
            
            if impacts["general"]:
                st.markdown("#### ⚡ General System Impacts")
                for impact in impacts["general"]:
                    st.markdown(f"⚠️ {impact['message']}")
      except Exception as _tab5_err:
        st.error(f"Error rendering Alerts tab: {_tab5_err}")
    
    # ==========================================================================
    # TAB 6: Interactive Map
    # ==========================================================================
    with tab6:
      try:
        st.header(f"🗺️ Interactive Map - {location_name}")
        
        # Create comprehensive map
        comprehensive_map = create_comprehensive_map(
            lat=lat,
            lon=lon,
            name=location_name,
            weather_data=weather_data,
            air_quality_data=air_quality_data,
            solar_data=solar_data,
            show_all_regions=True
        )
        
        # Display map
        st_folium(comprehensive_map, height=600, use_container_width=True)
        
        st.divider()
        
        # Quick summary
        st.subheader("📋 Location Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📍 Coordinates**")
            st.code(f"Latitude: {lat:.4f}°\nLongitude: {lon:.4f}°")
        
        with col2:
            if weather_data and "location" in weather_data:
                loc = weather_data["location"]
                st.markdown("**🌍 Location Info**")
                st.code(f"Timezone: GMT (standardized)\nElevation: {loc.get('elevation', 'N/A')}m")
        
        with col3:
            if location_name in REGIONS:
                region = REGIONS[location_name]
                st.markdown("**🏳️ Region Info**")
                st.code(f"Country: {region['country']}\nClimate: {REGION_CLIMATE.get(location_name, 'N/A').title()}")

        st.divider()

        # ---- 50 km Town Map ----
        st.subheader("🏘️ Local Area Map (50 km)")
        st.caption("Zoomable map of the 50 km × 50 km area around the selected location. "
                   "Use scroll-wheel or pinch to zoom into streets.")
        town_map = create_town_map(lat, lon, location_name, size_km=50,
                                   weather_data=weather_data)
        st_folium(town_map, height=550, use_container_width=True)

        st.divider()

        # ---- GWA / GSA–style Resource Maps ----
        st.subheader("🗺️ Resource Maps (GWA / GSA style)")
        res_tab_w, res_tab_s = st.tabs(["💨 Wind Resource", "☀️ Solar Resource"])

        with res_tab_w:
            st.caption("Mean wind speed heatmap — colour ramp similar to the "
                       "Global Wind Atlas (GWA).")
            wind_rmap = create_wind_resource_map(
                lat, lon, location_name, weather_data=weather_data, grid_km=50,
            )
            st_folium(wind_rmap, height=550, use_container_width=True)

        with res_tab_s:
            st.caption("Global Horizontal Irradiance (GHI) heatmap — colour ramp "
                       "similar to the Global Solar Atlas (GSA).")
            solar_rmap = create_solar_resource_map(
                lat, lon, location_name, solar_data=solar_data, grid_km=50,
            )
            st_folium(solar_rmap, height=550, use_container_width=True)

      except Exception as _tab6_err:
        st.error(f"Error rendering Map tab: {_tab6_err}")
        st.code(traceback.format_exc())

    # ==========================================================================
    # TAB 7: Forecast Comparison (Internal vs API)
    # ==========================================================================
    with tab7:
      try:
        st.header(f"📊 Forecast Comparison - {location_name}")
        st.markdown("""
        Compare **API-provided forecasts** (Open-Meteo) with our **internal prediction methods**.
        The API hourly data is split into a history window (used by internal models) and a
        future window; the internal models then forecast the future window so we can
        measure how each method stacks up against the professional API forecast.
        """)

        ts7 = get_ts(weather_data, resolution, lat=lat, lon=lon)
        if weather_data and ts7 is not None and isinstance(ts7, pd.DataFrame):
            hourly_df = ts7
            total_hours = len(hourly_df)

            # --- User controls ---
            fc_col1, fc_col2 = st.columns(2)
            with fc_col1:
                variable = st.selectbox(
                    "Variable to compare",
                    ["temperature", "wind_speed", "humidity", "cloud_cover", "pressure"],
                    format_func=lambda v: {
                        "temperature": "🌡️ Temperature (°C)",
                        "wind_speed": "💨 Wind Speed (km/h)",
                        "humidity": "💧 Humidity (%)",
                        "cloud_cover": "☁️ Cloud Cover (%)",
                        "pressure": "📊 Pressure (hPa)",
                    }.get(v, v),
                )
            with fc_col2:
                _step_label = {"h": "hours", "15min": "15-min steps", "5min": "5-min steps", "1min": "1-min steps", "D": "days"}.get(freq, "steps")
                fc_horizon = st.slider(
                    f"Forecast horizon ({_step_label})",
                    min_value=6,
                    max_value=min(total_hours // 2, 72),
                    value=min(24, total_hours // 2),
                    step=6,
                )

            # --- Run comparison ---
            with st.spinner("Running internal forecast models…"):
                comparison = run_forecast_comparison(
                    weather_data,
                    target_col=variable,
                    horizon_hours=fc_horizon,
                    history_hours=total_hours - fc_horizon,
                    freq=freq,
                    ts_df=ts7,
                )

            history = comparison["history"]
            api_future = comparison["api_future"]
            internal = comparison["internal_forecasts"]
            metrics = comparison["metrics"]

            # At high resolution, expand API forecast to a staircase
            _is_highres = resolution in ("5min", "1min")

            if not api_future.empty and not internal.empty:
                # ---- Main chart: API vs Internal methods ----
                var_labels = {
                    "temperature": "Temperature (°C)",
                    "wind_speed": "Wind Speed (km/h)",
                    "humidity": "Humidity (%)",
                    "cloud_cover": "Cloud Cover (%)",
                    "pressure": "Pressure (hPa)",
                }
                y_label = var_labels.get(variable, variable)

                fig = go.Figure()

                # Historical data (grey background)
                if not history.empty:
                    fig.add_trace(go.Scatter(
                        x=history["time"],
                        y=history[variable],
                        mode="lines",
                        name="Historical Observations",
                        line=dict(color="#95a5a6", width=1),
                        opacity=0.6,
                    ))

                # API forecast — staircase at high resolution, smooth otherwise
                if _is_highres:
                    api_step = step_function_df(
                        api_future.rename(columns={"value": variable}),
                        target_freq=resolution,
                    )
                    fig.add_trace(go.Scatter(
                        x=api_step["time"],
                        y=api_step[variable],
                        mode="lines",
                        name="API Forecast (staircase)",
                        line=dict(color="#2980b9", width=2, shape="hv"),
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=api_future["time"],
                        y=api_future["value"],
                        mode="lines",
                        name="API Forecast (Open-Meteo)",
                        line=dict(color="#2980b9", width=3),
                    ))

                # Internal methods
                method_colors = {
                    "Persistence": "#e74c3c",
                    "SARIMAX": "#f39c12",
                    "XGBoost": "#9b59b6",
                    "LSTM": "#1abc9c",
                    "Ensemble Mean": "#2c3e50",
                }

                for method_name, grp in internal.groupby("method"):
                    color = method_colors.get(str(method_name), "#7f8c8d")
                    is_ensemble = method_name == "Ensemble Mean"
                    fig.add_trace(go.Scatter(
                        x=grp["time"],
                        y=grp["value"],
                        mode="lines",
                        name=str(method_name),
                        line=dict(
                            color=color,
                            width=3 if is_ensemble else 1.5,
                            dash="solid" if is_ensemble else "dot",
                        ),
                    ))

                # Vertical line at split point
                if not history.empty:
                    split_time = history["time"].iloc[-1]
                    split_dt = split_time.to_pydatetime() if hasattr(split_time, 'to_pydatetime') else split_time
                    fig.add_shape(
                        type="line",
                        x0=split_dt, x1=split_dt,
                        y0=0, y1=1, yref="paper",
                        line=dict(dash="dash", color="rgba(255,255,255,0.3)", width=1.5),
                    )
                    fig.add_annotation(
                        x=split_dt, y=1, yref="paper",
                        text="Forecast Start",
                        showarrow=False,
                        font=dict(size=11, color="#adb5bd"),
                        yshift=10,
                    )

                fig.update_layout(
                    title=f"{y_label} — API vs Internal Forecasts",
                    xaxis_title="Time (GMT)",
                    yaxis_title=y_label,
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, width="stretch")

                # ---- Accuracy table ----
                st.subheader("📏 Accuracy Metrics (vs API Forecast)")
                st.caption(
                    "Lower MAE / RMSE = closer to API forecast. "
                    "Bias > 0 means the method over-estimates."
                )

                if metrics:
                    metrics_rows = []
                    for method_name, vals in metrics.items():
                        row = {"Method": method_name}
                        row.update(vals)
                        metrics_rows.append(row)
                    metrics_df = pd.DataFrame(metrics_rows).set_index("Method")

                    # Highlight the best method (lowest RMSE)
                    best_method = metrics_df["RMSE"].idxmin()
                    st.dataframe(
                        metrics_df.style.highlight_min(
                            subset=["MAE", "RMSE", "MAPE (%)"],
                            color="#d4edda",
                        ),
                        width="stretch",
                    )
                    st.success(f"🏆 Best internal method (lowest RMSE): **{best_method}**")
                else:
                    st.info("Not enough overlapping data to compute accuracy metrics.")

                # ---- Error distribution chart ----
                st.subheader("📉 Forecast Error Distribution")

                error_fig = go.Figure()
                for method_name, grp in internal.groupby("method"):
                    merged = pd.merge(
                        api_future[["time", "value"]].rename(columns={"value": "api"}),
                        grp[["time", "value"]].rename(columns={"value": "pred"}),
                        on="time", how="inner",
                    )
                    if merged.empty:
                        continue
                    errors = merged["pred"] - merged["api"]
                    color = method_colors.get(str(method_name), "#7f8c8d")
                    error_fig.add_trace(go.Scatter(
                        x=merged["time"],
                        y=errors,
                        mode="lines",
                        name=str(method_name),
                        line=dict(color=color, width=1.5),
                    ))

                error_fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
                error_fig.update_layout(
                    title="Prediction Error Over Time (Internal − API)",
                    xaxis_title="Time",
                    yaxis_title="Error",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0),
                    hovermode="x unified",
                )
                st.plotly_chart(error_fig, width="stretch")

                # ---- Method descriptions ----
                with st.expander("ℹ️ About the Internal Forecasting Methods"):
                    avail = available_methods()
                    avail_md = "  \n".join(
                        f"- **{m}**: {'✅ Available' if v else '❌ Not installed'}"
                        for m, v in avail.items()
                    )
                    st.markdown(f"**Back-end availability**  \n{avail_md}")
                    st.markdown("""
                    | Method | Description |
                    |--------|-------------|
                    | **Persistence** | Assumes the last observed value continues unchanged. Simplest possible baseline. |
                    | **SARIMAX** | Seasonal ARIMA(1,1,1)(1,1,0,24). Captures trend, autocorrelation, and 24-h seasonal patterns. |
                    | **XGBoost** | Gradient-boosted trees with lag features (1-24 h) and hour-of-day cyclical encoding. Recursive multi-step prediction. |
                    | **LSTM** | PyTorch recurrent neural network (48 hidden units, 30 epochs). Learns non-linear temporal dependencies from a sliding window. |
                    | **Ensemble Mean** | Weighted average of all available method outputs. Generally more robust than any single method. |
                    """)
            else:
                st.warning("Not enough hourly data to run forecast comparison. Try fetching more forecast days.")
        else:
            st.warning("Weather data not available. Please fetch data from the sidebar first.")
      except Exception as _tab7_err:
        st.error(f"Error rendering Forecast Comparison tab: {_tab7_err}")
        st.code(traceback.format_exc())

    # ==========================================================================
    # TAB 8: Wind Resource Assessment  (YouWind-style)
    # ==========================================================================
    with tab8:
      try:
        st.header(f"🌀 Wind Resource Assessment — {location_name}")
        st.markdown("**Bastankhah-Porté-Agel wake model** · Weibull fitting · IEC classification · Farm AEP")

        ts8 = get_ts(weather_data, resolution, lat=lat, lon=lon)
        if weather_data and ts8 is not None:
            hourly = ts8
            ws_arr = hourly["wind_speed"].values / 3.6        # km/h → m/s
            wd_arr = hourly["wind_direction"].values

            # --- Sidebar-like controls in expanders ---
            wr_col1, wr_col2 = st.columns(2)
            with wr_col1:
                with st.expander("⚙️ Assessment Parameters", expanded=True):
                    wr_hub = st.number_input("Hub height (m)", 20, 200, 80, key="wr_hub")
                    wr_alpha = st.slider("Shear exponent α", 0.05, 0.30, 0.14, 0.01, key="wr_alpha")
                    wr_rho = st.number_input("Air density ρ (kg/m³)", 1.00, 1.30, 1.225, 0.005, key="wr_rho", format="%.3f")
            with wr_col2:
                with st.expander("🏗️ Turbine Selection", expanded=True):
                    wr_turb_name = st.selectbox("Turbine model", list(TURBINE_CATALOG.keys()), index=8, key="wr_turb")
                    wr_turb = TURBINE_CATALOG[wr_turb_name]
                    st.caption(f"{wr_turb.rated_kw} kW | D={wr_turb.rotor_d}m | Hub={wr_turb.hub_height}m | Ct={wr_turb.ct}")

            # --- Run assessment ---
            resource = assess_wind_resource(ws_arr, wd_arr, 10.0, wr_hub, wr_alpha, wr_rho)

            # Metrics row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Mean WS (hub)", f"{resource['mean_ws_hub']:.1f} m/s")
            m2.metric("Weibull k", f"{resource['weibull_k_hub']:.2f}")
            m3.metric("Weibull c", f"{resource['weibull_c_hub']:.1f} m/s")
            m4.metric("Power Density", f"{resource['power_density_hub_wm2']:.0f} W/m²")
            m5.metric("IEC Class", resource['iec_class']['wind_class'].split('(')[0].strip())

            st.divider()

            # --- Weibull PDF + Wind Rose ---
            wr_c1, wr_c2 = st.columns(2)
            with wr_c1:
                ws_bins = np.linspace(0, max(20, resource['mean_ws_hub'] * 3), 100)
                pdf_vals = weibull_pdf(ws_bins, resource['weibull_k_hub'], resource['weibull_c_hub'])
                fig_wb = go.Figure()
                fig_wb.add_trace(go.Scatter(x=ws_bins, y=pdf_vals, fill="tozeroy",
                                            name="Weibull PDF", line=dict(color="#3498db")))
                # Histogram of actual data
                ws_hub = np.array([extrapolate_wind_speed(u, 10, wr_hub, wr_alpha) for u in ws_arr])
                fig_wb.add_trace(go.Histogram(x=ws_hub, histnorm="probability density",
                                              nbinsx=30, name="Observed", opacity=0.4,
                                              marker_color="#e74c3c"))
                fig_wb.update_layout(title="Wind Speed Distribution (Hub Height)",
                                     xaxis_title="Wind Speed (m/s)", yaxis_title="Probability Density",
                                     height=400)
                st.plotly_chart(fig_wb, width="stretch")

            with wr_c2:
                # Wind rose
                rose = resource['wind_rose']
                rose_df = pd.DataFrame(rose)
                fig_rose = px.bar_polar(rose_df, r="frequency", theta="direction",
                                        color="mean_ws", color_continuous_scale="YlOrRd",
                                        title="Wind Rose")
                fig_rose.update_layout(height=400,
                                       polar=dict(angularaxis=dict(direction="clockwise", rotation=90)))
                st.plotly_chart(fig_rose, width="stretch")

            # --- Power curve ---
            st.subheader("⚡ Turbine Power Curve & Capacity Factor")
            pc_c1, pc_c2 = st.columns(2)
            with pc_c1:
                pc_ws = np.linspace(0, 30, 200)
                pc_p = wind_power_curve(pc_ws, wr_turb)
                fig_pc = go.Figure()
                fig_pc.add_trace(go.Scatter(x=pc_ws, y=pc_p, name=wr_turb.name,
                                            line=dict(color="#2ecc71", width=2)))
                fig_pc.add_vline(x=wr_turb.cut_in, line_dash="dash", annotation_text="Cut-in")
                fig_pc.add_vline(x=wr_turb.rated_ws, line_dash="dash", annotation_text="Rated")
                fig_pc.add_vline(x=wr_turb.cut_out, line_dash="dash", annotation_text="Cut-out")
                fig_pc.update_layout(title=f"Power Curve — {wr_turb.name}",
                                     xaxis_title="Wind Speed (m/s)", yaxis_title="Power (kW)",
                                     height=350)
                st.plotly_chart(fig_pc, width="stretch")
            with pc_c2:
                cf_val = wind_cf(ws_hub, wr_turb)
                aep_single = cf_val * wr_turb.rated_kw * 8760 / 1000  # MWh/yr (scaled)
                capex_kw = estimate_capex_per_kw(wr_turb.rated_kw)
                lcoe_val = wind_lcoe(capex_kw, cf_val, rho_site=wr_rho)
                st.metric("Capacity Factor", f"{cf_val*100:.1f}%")
                st.metric("Est. AEP (single turbine)", f"{aep_single:.1f} MWh/yr")
                st.metric("CAPEX (est.)", f"${capex_kw:.0f}/kW")
                st.metric("LCOE (est.)", f"${lcoe_val:.4f}/kWh")

            st.divider()

            # --- Wake model visualisation ---
            st.subheader("🌀 Bastankhah Wake Model Visualisation")
            wk_c1, wk_c2 = st.columns([1, 2])
            with wk_c1:
                wk_ct = st.slider("Thrust Coeff. Ct", 0.2, 1.0, float(wr_turb.ct), 0.05, key="wk_ct")
                wk_kw = st.slider("Wake expansion k_w", 0.02, 0.15, 0.075, 0.005, key="wk_kw")
                st.caption("k_w ≈ 0.04 offshore · 0.075 onshore")
            with wk_c2:
                X, Y, U = bastankhah_wake_field((-1, 15), (-3, 3), wk_ct, wk_kw, nx=150, ny=80)
                fig_wake = go.Figure(data=go.Heatmap(
                    x=X[0], y=Y[:, 0], z=U,
                    colorscale="RdYlGn", zmin=0.5, zmax=1.0,
                    colorbar=dict(title="U/U∞"),
                ))
                fig_wake.update_layout(title="Single-Turbine Wake Field (Bastankhah)",
                                       xaxis_title="Downstream (x/D)",
                                       yaxis_title="Lateral (y/D)",
                                       height=400, yaxis=dict(scaleanchor="x"))
                st.plotly_chart(fig_wake, width="stretch")

            # --- Wind farm layout & AEP ---
            st.subheader("🏗️ Wind Farm Layout & AEP")
            fm_c1, fm_c2 = st.columns(2)
            with fm_c1:
                fm_rows = st.number_input("Rows", 2, 10, 3, key="fm_r")
                fm_cols = st.number_input("Columns", 2, 15, 4, key="fm_c")
                fm_sx = st.slider("Spacing X (× D)", 3.0, 12.0, 7.0, 0.5, key="fm_sx")
                fm_sy = st.slider("Spacing Y (× D)", 3.0, 12.0, 5.0, 0.5, key="fm_sy")
                fm_layout_type = st.radio("Layout", ["Grid", "Staggered"], key="fm_lt", horizontal=True)

            if fm_layout_type == "Grid":
                layout_xy = generate_grid_layout(fm_rows, fm_cols, fm_sx, fm_sy, wr_turb.rotor_d)
            else:
                layout_xy = generate_staggered_layout(fm_rows, fm_cols, fm_sx, fm_sy, wr_turb.rotor_d)

            with fm_c2:
                # Plot layout
                fig_layout = go.Figure()
                fig_layout.add_trace(go.Scatter(
                    x=layout_xy[:, 0], y=layout_xy[:, 1],
                    mode="markers+text",
                    marker=dict(size=12, color="#3498db", symbol="x"),
                    text=[f"T{i+1}" for i in range(len(layout_xy))],
                    textposition="top center", name="Turbines",
                ))
                fig_layout.update_layout(
                    title=f"Farm Layout — {len(layout_xy)} × {wr_turb.name}",
                    xaxis_title="X (m)", yaxis_title="Y (m)",
                    height=400, yaxis=dict(scaleanchor="x"),
                )
                st.plotly_chart(fig_layout, width="stretch")

            # Compute farm AEP
            with st.spinner("Computing farm AEP with wake effects…"):
                farm_result = farm_aep(layout_xy, ws_hub, wd_arr, wr_turb, k_w=wk_kw)

            fa1, fa2, fa3, fa4 = st.columns(4)
            fa1.metric("N Turbines", farm_result['n_turbines'])
            fa2.metric("Farm AEP", f"{farm_result['aep_mwh']:.0f} MWh")
            fa3.metric("Wake Loss", f"{farm_result['wake_loss_pct']:.1f}%")
            fa4.metric("Farm CF", f"{farm_result['capacity_factor']*100:.1f}%")

            # Production capacity
            total_mw = wr_turb.rated_kw * farm_result['n_turbines'] / 1000
            st.info(f"**Total Installed Capacity:** {total_mw:.1f} MW  |  "
                    f"**AEP (no wake):** {farm_result['aep_no_wake_mwh']:.0f} MWh  |  "
                    f"**Wake Effect Efficiency:** {100 - farm_result['wake_loss_pct']:.1f}%")
        else:
            st.warning("Weather data not available. Please fetch data from the sidebar first.")
      except Exception as _tab8_err:
        st.error(f"Error rendering Wind Resource tab: {_tab8_err}")

    # ==========================================================================
    # TAB 9: Solar Resource Assessment
    # ==========================================================================
    with tab9:
      try:
        st.header(f"🔆 Solar Resource Assessment — {location_name}")
        st.markdown("**PV system sizing** · Irradiance analysis · Temperature derating · LCOE")

        sol_ts = get_ts(solar_data, resolution, lat=lat, lon=lon)
        wea_ts = get_ts(weather_data, resolution, lat=lat, lon=lon)
        if solar_data and sol_ts is not None:
            sol_hourly = sol_ts
            has_temp = wea_ts is not None

            # Controls
            sr_c1, sr_c2 = st.columns(2)
            with sr_c1:
                with st.expander("☀️ PV Module", expanded=True):
                    sr_mod_name = st.selectbox("Module", list(PV_CATALOG.keys()), index=2, key="sr_mod")
                    sr_mod = PV_CATALOG[sr_mod_name]
                    st.caption(f"{sr_mod.rated_wp}Wp | η={sr_mod.efficiency*100:.1f}% | {sr_mod.technology}")
                    sr_n_modules = st.number_input("Number of modules", 10, 100000, 1000, key="sr_nm")
            with sr_c2:
                with st.expander("⚙️ System Parameters", expanded=True):
                    sr_tilt = st.number_input("Panel tilt (°)", 0, 90,
                                              int(sr_optimal_tilt(lat)["annual_optimal"]), key="sr_tilt")
                    sr_capex = st.number_input("CAPEX ($/kWp)", 400, 2000, 700, key="sr_capex")
                    sr_climate = st.selectbox("Climate (soiling)",
                                             ["desert", "semi_arid", "coastal", "tropical"],
                                             index=1, key="sr_clim")

            # Prepare data
            ghi_arr = sol_hourly["shortwave_radiation"].values    # W/m²
            if has_temp:
                temp_arr = wea_ts["temperature"].values
            else:
                temp_arr = np.full(len(ghi_arr), 25.0)

            # Assess
            sr_result = assess_solar_resource(ghi_arr, temp_arr, lat, lon, sr_mod, sr_n_modules, sr_tilt)

            # Metrics
            sr_m1, sr_m2, sr_m3, sr_m4, sr_m5 = st.columns(5)
            sr_m1.metric("System Size", f"{sr_result['system_kwp']:.0f} kWp")
            sr_m2.metric("Capacity Factor", f"{sr_result['capacity_factor']*100:.1f}%")
            sr_m3.metric("Daily Output", f"{sr_result['avg_daily_kwh']:.0f} kWh")
            sr_m4.metric("Avg PSH", f"{sr_result['avg_daily_psh']:.1f} h")
            sr_m5.metric("Perf. Ratio", f"{sr_result['performance_ratio']*100:.0f}%")

            st.divider()

            # Charts
            sr_ch1, sr_ch2 = st.columns(2)
            with sr_ch1:
                # Hourly GHI profile
                fig_ghi = go.Figure()
                fig_ghi.add_trace(go.Scatter(
                    x=sol_hourly["time"], y=ghi_arr,
                    fill="tozeroy", name="GHI",
                    line=dict(color="#f1c40f"),
                ))
                fig_ghi.update_layout(title="Global Horizontal Irradiance",
                                      xaxis_title="Time (GMT)", yaxis_title="GHI (W/m²)",
                                      height=400)
                st.plotly_chart(fig_ghi, width="stretch")

            with sr_ch2:
                # Hourly power output
                energy_h = np.array([
                    pv_power_output(ghi_arr[i], sr_mod,
                                    cell_temperature(temp_arr[i], ghi_arr[i]),
                                    sr_n_modules) / 1000
                    for i in range(len(ghi_arr))
                ])
                fig_pv = go.Figure()
                fig_pv.add_trace(go.Scatter(
                    x=sol_hourly["time"], y=energy_h,
                    fill="tozeroy", name="PV Output",
                    line=dict(color="#e67e22"),
                ))
                fig_pv.update_layout(title="PV Power Output",
                                     xaxis_title="Time (GMT)", yaxis_title="Power (kW)",
                                     height=400)
                st.plotly_chart(fig_pv, width="stretch")

            # Soiling & LCOE
            st.subheader("💰 Economics & Soiling")
            ec1, ec2 = st.columns(2)
            with ec1:
                sr_lcoe = compute_solar_lcoe(sr_result['system_kwp'],
                                             sr_result['annual_kwh_est'],
                                             capex_per_kwp=sr_capex)
                soiling = morocco_soiling_loss(sr_climate)
                st.metric("LCOE", f"${sr_lcoe:.4f}/kWh")
                st.metric("Annual Yield (est.)", f"{sr_result['annual_kwh_est']/1000:.0f} MWh")
                st.metric("Specific Yield", f"{sr_result['specific_yield_kwh_kwp']:.1f} kWh/kWp")
                st.metric("Soiling Loss", f"{soiling['soiling_loss_pct']:.1f}%")
            with ec2:
                # Monthly profile
                monthly = estimate_monthly_profile(lat,
                    sr_result['avg_ghi_wm2'] * 8.76 if sr_result['avg_ghi_wm2'] > 0 else 2000)
                fig_month = go.Figure()
                fig_month.add_trace(go.Bar(
                    x=monthly["month_name"], y=monthly["ghi_kwh_m2"],
                    marker_color="#f39c12", name="Monthly GHI",
                ))
                fig_month.update_layout(title="Estimated Monthly Solar Profile",
                                        xaxis_title="Month",
                                        yaxis_title="GHI (kWh/m²)",
                                        height=350)
                st.plotly_chart(fig_month, width="stretch")

            # Morocco presets
            with st.expander("🇲🇦 Morocco Solar Sites Reference"):
                site_rows = []
                for sname, sinfo in MOROCCO_SOLAR_SITES.items():
                    site_rows.append({
                        "Site": sname,
                        "GHI (kWh/m²/yr)": sinfo["ghi_kwh_m2_yr"],
                        "DNI (kWh/m²/yr)": sinfo["dni_kwh_m2_yr"],
                        "Avg Temp (°C)": sinfo["avg_temp_c"],
                        "Elevation (m)": sinfo["elev"],
                    })
                st.table(pd.DataFrame(site_rows))
        else:
            st.warning("Solar data not available. Please fetch data from the sidebar first.")
      except Exception as _tab9_err:
        st.error(f"Error rendering Solar Resource tab: {_tab9_err}")

    # ==========================================================================
    # TAB 10: Hybrid Wind+Solar & Green Hydrogen (Morocco)
    # ==========================================================================
    with tab10:
      try:
        st.header("⚡ Hybrid Wind+Solar & Green Hydrogen — Morocco")
        st.markdown("""
        Size a **hybrid RE plant** feeding a **green hydrogen electrolyser**.
        Based on Morocco's national hydrogen strategy (MASEN / GGGI targets).
        """)

        # --- System configuration ---
        h2_c1, h2_c2, h2_c3 = st.columns(3)
        with h2_c1:
            h2_zone = st.selectbox("🇲🇦 Morocco H₂ Zone", list(MOROCCO_H2_ZONES.keys()), key="h2z")
            h2_target = st.number_input("Target H₂ production (tonnes/yr)", 10, 100000, 1000, key="h2t")
            h2_elec = st.selectbox("Electrolyser", list(ELECTROLYSER_CATALOG.keys()), index=1, key="h2e")
        with h2_c2:
            h2_wr = st.slider("Wind share (%)", 0, 100, 60, key="h2wr")
            h2_bat = st.slider("Battery storage (hours)", 0, 8, 2, key="h2bat")
            h2_dr = st.slider("Discount rate (%)", 4, 12, 7, key="h2dr")
        with h2_c3:
            h2_wind_capex = st.number_input("Wind CAPEX ($/kW)", 600, 3000, 1200, key="h2wc")
            h2_sol_capex = st.number_input("Solar CAPEX ($/kWp)", 300, 1500, 700, key="h2sc")
            h2_price = st.number_input("H₂ selling price ($/kg)", 1.0, 10.0, 4.0, 0.5, key="h2p")

        # --- Run quick sizing ---
        with st.spinner("Sizing hybrid plant…"):
            sizing = quick_hybrid_sizing(
                target_h2_tonnes_yr=h2_target,
                zone_name=h2_zone,
                electrolyser_name=h2_elec,
                wind_capex_per_kw=h2_wind_capex,
                solar_capex_per_kwp=h2_sol_capex,
                wind_solar_ratio=h2_wr / 100,
                battery_hours=h2_bat,
                discount_rate=h2_dr / 100,
                h2_selling_price=h2_price,
            )

        st.divider()

        # --- Key metrics ---
        hm1, hm2, hm3, hm4, hm5 = st.columns(5)
        hm1.metric("💨 Wind", f"{sizing['wind_capacity_mw']:.1f} MW")
        hm2.metric("☀️ Solar", f"{sizing['solar_capacity_mwp']:.1f} MWp")
        hm3.metric("🔌 Electrolyser", f"{sizing['electrolyser_capacity_mw']:.1f} MW")
        hm4.metric("⚡ H₂ / year", f"{sizing['annual_h2_tonnes']:.0f} t")
        hm5.metric("💰 LCOH", f"${sizing['lcoh_usd_kg']:.2f}/kg")

        st.divider()

        # --- CAPEX breakdown ---
        h2_r1, h2_r2 = st.columns(2)
        with h2_r1:
            st.subheader("💵 CAPEX Breakdown")
            cb = sizing["capex_breakdown"]
            capex_df = pd.DataFrame({
                "Component": ["Wind", "Solar", "Electrolyser", "Battery", "Desalination"],
                "Cost (M$)": [cb["wind_musd"], cb["solar_musd"], cb["electrolyser_musd"],
                              cb["battery_musd"], cb["desalination_musd"]],
            })
            fig_capex = px.pie(capex_df, names="Component", values="Cost (M$)",
                               title=f"Total CAPEX: ${cb['total_musd']:.1f}M",
                               color_discrete_sequence=px.colors.qualitative.Set2)
            fig_capex.update_layout(height=380)
            st.plotly_chart(fig_capex, width="stretch")

        with h2_r2:
            st.subheader("📊 Project Economics")
            econ_data = {
                "Metric": [
                    "Total CAPEX", "Annual OPEX", "Annual Revenue (H₂)",
                    "CO₂ Credits/yr", "NPV (25yr)", "Payback Period",
                    "CO₂ Avoided/yr", "Water Consumption",
                ],
                "Value": [
                    f"${cb['total_musd']:.1f} M",
                    f"${sizing['opex_yr_musd']:.2f} M",
                    f"${sizing['annual_revenue_musd']:.2f} M",
                    f"${sizing['co2_credit_yr_usd']:,.0f}",
                    f"${sizing['npv_musd']:.1f} M",
                    f"{sizing['payback_yr']:.1f} years",
                    f"{sizing['co2_avoided_tonnes_yr']:,.0f} t CO₂",
                    f"{sizing['water_m3_yr']:,.0f} m³/yr",
                ],
            }
            st.table(pd.DataFrame(econ_data))

        st.divider()

        # --- LCOH comparison bar chart ---
        st.subheader("📈 LCOH Benchmarking")
        roadmap = morocco_roadmap_comparison(sizing['annual_h2_tonnes'], sizing['lcoh_usd_kg'])

        lcoh_compare = pd.DataFrame({
            "Type": ["Grey H₂", "Blue H₂", "Green Target", "This Project"],
            "LCOH ($/kg)": [roadmap["benchmark_lcoh_grey"], roadmap["benchmark_lcoh_blue"],
                            roadmap["benchmark_lcoh_green_target"], sizing["lcoh_usd_kg"]],
        })
        fig_lcoh = px.bar(lcoh_compare, x="Type", y="LCOH ($/kg)",
                          color="Type",
                          color_discrete_map={"Grey H₂": "#95a5a6", "Blue H₂": "#3498db",
                                              "Green Target": "#2ecc71", "This Project": "#e74c3c"},
                          title="LCOH Comparison")
        fig_lcoh.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_lcoh, width="stretch")

        if roadmap["is_market_viable"]:
            st.success(f"✅ Project LCOH (${sizing['lcoh_usd_kg']:.2f}/kg) is below market price (${h2_price}/kg) — **viable**")
        else:
            st.warning(f"⚠️ Project LCOH (${sizing['lcoh_usd_kg']:.2f}/kg) exceeds market price (${h2_price}/kg)")

        st.info(f"🇲🇦 This project represents **{roadmap['share_of_2030_target_pct']:.4f}%** "
                f"of Morocco's 2030 green H₂ export target ({MOROCCO_H2_CONTEXT['export_target_Mt_yr_2030']} Mt/yr)")

        # --- Hourly dispatch simulation (if we have weather data) ---
        wea_ts10 = get_ts(weather_data, resolution, lat=lat, lon=lon)
        sol_ts10 = get_ts(solar_data, resolution, lat=lat, lon=lon)
        if wea_ts10 is not None and sol_ts10 is not None:

            with st.expander(f"📊 {res_label} Dispatch Simulation (based on fetched data)", expanded=False):
                sol_h = sol_ts10
                wea_h = wea_ts10

                # Simplified power models
                ws_ms = wea_h["wind_speed"].values / 3.6
                ghi_w = sol_h["shortwave_radiation"].values

                # Wind power (simplified CF scaling)
                wind_norm = np.clip(ws_ms / 12.0, 0, 1) ** 3
                wind_kw_h = wind_norm * sizing["wind_capacity_mw"] * 1000

                # Solar power (simplified)
                solar_norm = np.clip(ghi_w / 1000.0, 0, 1)
                solar_kw_h = solar_norm * sizing["solar_capacity_mwp"] * 1000 * 0.18

                elec_obj = ELECTROLYSER_CATALOG[h2_elec]
                dispatch = hybrid_dispatch(
                    wind_kw_h, solar_kw_h, elec_obj, sizing["n_electrolysers"],
                    battery_kwh=sizing["battery_mwh"] * 1000,
                )

                # Complementarity
                ci = complementarity_index(wind_kw_h, solar_kw_h)

                dc1, dc2 = st.columns(2)
                with dc1:
                    time_ax = sol_h["time"] if "time" in sol_h.columns else list(range(len(wind_kw_h)))
                    fig_disp = go.Figure()
                    fig_disp.add_trace(go.Scatter(x=time_ax, y=wind_kw_h / 1000,
                                                  name="Wind (MW)", fill="tozeroy",
                                                  line=dict(color="#3498db")))
                    fig_disp.add_trace(go.Scatter(x=time_ax, y=solar_kw_h / 1000,
                                                  name="Solar (MW)", fill="tozeroy",
                                                  line=dict(color="#f1c40f")))
                    fig_disp.add_trace(go.Scatter(x=time_ax,
                                                  y=dispatch["hourly"]["elec_input_kw"] / 1000,
                                                  name="Electrolyser (MW)",
                                                  line=dict(color="#e74c3c", width=2)))
                    fig_disp.update_layout(title="Hourly Power Dispatch",
                                           xaxis_title="Time (GMT)", yaxis_title="Power (MW)",
                                           height=400, hovermode="x unified")
                    st.plotly_chart(fig_disp, width="stretch")

                with dc2:
                    # H2 production
                    fig_h2 = go.Figure()
                    fig_h2.add_trace(go.Bar(x=time_ax,
                                            y=dispatch["hourly"]["h2_kg"],
                                            name="H₂ (kg/h)",
                                            marker_color="#2ecc71"))
                    fig_h2.update_layout(title="Hourly H₂ Production",
                                         xaxis_title="Time (GMT)", yaxis_title="H₂ (kg/h)",
                                         height=400)
                    st.plotly_chart(fig_h2, width="stretch")

                di1, di2, di3, di4 = st.columns(4)
                di1.metric("H₂ Produced", f"{dispatch['total_h2_kg']:.0f} kg")
                di2.metric("Elec. CF", f"{dispatch['electrolyser_cf']*100:.1f}%")
                di3.metric("Curtailment", f"{dispatch['curtailment_pct']:.1f}%")
                di4.metric("Complementarity", f"{ci['complementarity_index']:.2f}")

        # --- Green ammonia bonus ---
        with st.expander("🧪 Green Ammonia (NH₃) Potential"):
            nh3 = green_ammonia_production(sizing["annual_h2_kg"])
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | H₂ Input | {nh3['h2_input_tonnes_yr']:.0f} tonnes/yr |
            | NH₃ Output | {nh3['nh3_tonnes_yr']:.0f} tonnes/yr |
            | Conversion Efficiency | {nh3['conversion_efficiency']*100:.0f}% |
            | Additional Energy | {nh3['energy_input_mwh_yr']:.0f} MWh/yr |
            """)

        # --- Morocco H2 zones reference ---
        with st.expander("🇲🇦 Morocco Green H₂ Zones Reference"):
            zone_rows = []
            for zname, zinfo in MOROCCO_H2_ZONES.items():
                zone_rows.append({
                    "Zone": zname,
                    "Wind CF": f"{zinfo['wind_cf']*100:.0f}%",
                    "Solar CF": f"{zinfo['solar_cf']*100:.0f}%",
                    "Mean WS (m/s)": zinfo["mean_ws_ms"],
                    "GHI (kWh/m²/yr)": zinfo["ghi_kwh_m2_yr"],
                    "Water": zinfo["water_source"],
                })
            st.table(pd.DataFrame(zone_rows))

      except Exception as _tab10_err:
        st.error(f"Error rendering Hybrid & H₂ tab: {_tab10_err}")


# =============================================================================
# Footer
# =============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #adb5bd; padding: 1rem;">
    <p>🌍 Renewable Energy & Weather Dashboard</p>
    <p>Data Sources: Open-Meteo API | NASA POWER</p>
    <p>🕐 All timestamps displayed in GMT</p>
    <p>© 2025 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
