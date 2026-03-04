# 🌍 Renewable Energy & Weather Dashboard

A comprehensive Python dashboard for renewable energy assessment and weather forecasting, targeting African regions with high solar and wind potential.

## 📋 Features

### 🌤️ Weather Forecasting
- Current conditions (temperature, humidity, pressure, cloud cover)
- 7-day daily forecast with weather icons
- Hourly temperature and precipitation charts
- Real-time weather code descriptions

### 💨 Wind Energy Analysis
- Wind speed and gust forecasting
- Wind power density calculations
- Wind resource classification (Classes 1-7)
- Energy production estimates for configured turbines
- Wind direction distribution analysis

### ☀️ Solar Energy Analysis
- Solar irradiance forecasting (GHI, DNI, diffuse)
- Peak sun hours calculation
- **Soiling effect modeling** - accounts for dust/dirt accumulation
- Solar resource classification
- Energy production estimates with temperature derating
- Optimal panel tilt angle recommendations

### 🌬️ Air Quality Monitoring
- Real-time AQI (US & European scales)
- Pollutant levels (PM2.5, PM10, O₃, NO₂, SO₂, CO)
- Health advisories based on air quality
- Impact assessment on solar panel efficiency
- Aerosol Optical Depth analysis

### ⚠️ Extreme Weather Alerts
- High wind warnings
- Extreme heat/cold advisories
- Heavy precipitation alerts
- Thunderstorm warnings
- UV radiation warnings
- Flooding risk assessment
- Renewable energy impact analysis

### 🗺️ Interactive Maps
- Folium-based interactive maps
- Multiple tile layers (OpenStreetMap, Satellite, Topographic)
- Weather data overlays
- Air quality indicators
- Wind speed visualization
- All target regions displayed

## 🎯 Target Regions

| Region | Country | Coordinates |
|--------|---------|-------------|
| Kinshasa | DR Congo | -4.44°, 15.27° |
| Benguerir | Morocco | 32.23°, -7.95° |
| Casablanca | Morocco | 33.57°, -7.59° |
| Errachidia | Morocco | 31.93°, -4.43° |
| Dakhla | Morocco | 23.68°, -15.96° |
| Porto-Novo | Benin | 6.50°, 2.63° |
| Abidjan | Ivory Coast | 5.36°, -4.01° |
| Meskala | Morocco | 31.35°, -9.77° |
| Goma | DR Congo | -1.68°, 29.24° |
| Bukavu | DR Congo | -2.51°, 28.86° |

**Custom coordinates** can also be entered manually.

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Setup

1. **Clone or download the project:**
   ```bash
   cd C:\Users\Public\RenewableEnergyDashboard
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser:**
   The app will automatically open at `http://localhost:8501`

## 📖 Usage Guide

### Basic Usage

1. **Select a Location:**
   - Choose from predefined regions in the sidebar dropdown
   - Or switch to "Custom Coordinates" and enter latitude/longitude

2. **Configure Settings:**
   - Adjust forecast days (1-14)
   - Set solar panel parameters (area, efficiency, cleaning interval)
   - Configure wind turbine settings (capacity, rotor diameter)

3. **Fetch Data:**
   - Click the "🔄 Fetch Data" button
   - Wait for data to load from APIs

4. **Explore Tabs:**
   - **Weather**: Current conditions and forecasts
   - **Wind Energy**: Wind resource assessment
   - **Solar Energy**: Solar potential with soiling analysis
   - **Air Quality**: AQI and pollutant monitoring
   - **Alerts**: Extreme weather warnings
   - **Map**: Interactive visualization

### Understanding the Soiling Effect

The soiling module estimates energy loss due to dust/dirt accumulation:

| Climate Type | Monthly Soiling Rate | Cleaning Interval |
|--------------|---------------------|-------------------|
| Desert | 15% | 14 days |
| Semi-arid | 10% | 21 days |
| Tropical | 5% | 45 days |
| Coastal | 7% | 30 days |
| Urban | 8% | 30 days |

## 🔌 Data Sources

All APIs used are **free and require no API keys**:

| Data Type | Source | Documentation |
|-----------|--------|---------------|
| Weather & Forecast | [Open-Meteo](https://open-meteo.com/) | [API Docs](https://open-meteo.com/en/docs) |
| Air Quality | [Open-Meteo Air Quality](https://open-meteo.com/en/docs/air-quality-api) | [API Docs](https://open-meteo.com/en/docs/air-quality-api) |
| Solar Radiation | [Open-Meteo](https://open-meteo.com/) | [API Docs](https://open-meteo.com/en/docs) |
| Historical Solar | [NASA POWER](https://power.larc.nasa.gov/) | [API Docs](https://power.larc.nasa.gov/docs/) |

## 📁 Project Structure

```
RenewableEnergyDashboard/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration and constants
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── modules/
    ├── __init__.py          # Package initialization
    ├── weather_wind.py      # Weather & wind data fetching
    ├── solar_energy.py      # Solar energy calculations
    ├── extreme_weather.py   # Alerts and warnings
    ├── air_quality.py       # Air quality monitoring
    └── map_visualization.py # Interactive maps (Folium)
```

## 🧮 Key Calculations

### Wind Power Density
```
P = 0.5 × ρ × v³
```
Where: P = power density (W/m²), ρ = air density (kg/m³), v = wind speed (m/s)

### Solar Energy with Soiling
```
E_net = E_gross × (1 - system_losses) × (1 - soiling_loss) × temp_derating
```

### Temperature Derating
```
Derating = 1 + temp_coefficient × (T_cell - 25)
```

## 🤝 Contributing

Feel free to extend this dashboard:
- Add more regions to `config.py`
- Integrate additional data sources
- Enhance visualization options
- Add export functionality

## 📄 License

This project is open source and available for educational and commercial use.

## 🙏 Acknowledgments

- [Open-Meteo](https://open-meteo.com/) for free weather APIs
- [NASA POWER](https://power.larc.nasa.gov/) for solar resource data
- [Streamlit](https://streamlit.io/) for the web framework
- [Folium](https://python-visualization.github.io/folium/) for interactive maps

---

**Built with ❤️ for renewable energy development in Africa**
