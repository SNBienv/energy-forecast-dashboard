"""Quick test of all modules with Kinshasa data."""
import sys, traceback

# Test imports
print("=== IMPORT TESTS ===")
try:
    from modules.weather_wind import fetch_weather_data, process_weather_data, estimate_wind_energy_potential
    from modules.solar_energy import fetch_solar_radiation_data, process_solar_data, calculate_soiling_loss, calculate_solar_potential, calculate_optimal_tilt, get_solar_resource_class
    from modules.extreme_weather import analyze_weather_alerts, analyze_renewable_energy_impact
    from modules.air_quality import fetch_air_quality_data, process_air_quality_data, analyze_air_quality_for_solar, generate_air_quality_alerts, get_pollutant_info
    from modules.map_visualization import create_comprehensive_map, create_regions_overview_map
    from modules.forecasting import run_forecast_comparison
    print("All imports OK")
except Exception as e:
    print(f"IMPORT FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test data fetching (Kinshasa)
LAT, LON = -4.4419, 15.2663

print("\n=== DATA FETCH TESTS ===")

# Weather
print("Fetching weather...")
try:
    weather_raw = fetch_weather_data(LAT, LON, 7)
    if weather_raw:
        weather = process_weather_data(weather_raw)
        print(f"  Weather OK - keys: {list(weather.keys())}")
        print(f"  Current temp: {weather.get('current', {}).get('temperature')}")
        hourly = weather.get("hourly")
        if hourly is not None:
            print(f"  Hourly rows: {len(hourly)}, cols: {list(hourly.columns)}")
        daily = weather.get("daily")
        if daily is not None:
            print(f"  Daily rows: {len(daily)}, cols: {list(daily.columns)}")
    else:
        print("  Weather FAILED - returned None")
        weather = {}
except Exception as e:
    print(f"  Weather ERROR: {e}")
    traceback.print_exc()
    weather = {}

# Solar
print("\nFetching solar...")
try:
    solar_raw = fetch_solar_radiation_data(LAT, LON, 7)
    if solar_raw:
        solar = process_solar_data(solar_raw)
        print(f"  Solar OK - keys: {list(solar.keys())}")
        daily_s = solar.get("daily")
        if daily_s is not None:
            print(f"  Daily rows: {len(daily_s)}, cols: {list(daily_s.columns)}")
            print(f"  radiation_sum sample: {daily_s['radiation_sum'].head().tolist()}")
        else:
            print("  Solar daily is None!")
        hourly_s = solar.get("hourly")
        if hourly_s is not None:
            print(f"  Hourly rows: {len(hourly_s)}, cols: {list(hourly_s.columns)}")
        else:
            print("  Solar hourly is None!")
    else:
        print("  Solar FAILED - returned None")
        solar = {}
except Exception as e:
    print(f"  Solar ERROR: {e}")
    traceback.print_exc()
    solar = {}

# Air Quality
print("\nFetching air quality...")
try:
    aq_raw = fetch_air_quality_data(LAT, LON, 5)
    if aq_raw:
        aq = process_air_quality_data(aq_raw)
        print(f"  AQ OK - keys: {list(aq.keys())}")
        current_aq = aq.get("current", {})
        print(f"  Current AQ keys: {list(current_aq.keys())}")
        print(f"  US AQI: {current_aq.get('us_aqi')}")
        print(f"  EU AQI: {current_aq.get('european_aqi')}")
        hourly_aq = aq.get("hourly")
        if hourly_aq is not None:
            print(f"  Hourly rows: {len(hourly_aq)}, cols: {list(hourly_aq.columns)}")
        else:
            print("  AQ hourly is None!")
    else:
        print("  AQ FAILED - returned None")
        aq = {}
except Exception as e:
    print(f"  AQ ERROR: {e}")
    traceback.print_exc()
    aq = {}

# Test tab 2 logic (Wind)
print("\n=== TAB LOGIC TESTS ===")
print("Tab 2 (Wind Energy)...")
try:
    if weather and "hourly" in weather and weather["hourly"] is not None:
        hourly = weather["hourly"]
        avg_wind = hourly["wind_speed"].mean()
        wind_pot = estimate_wind_energy_potential(avg_wind)
        print(f"  Wind OK - avg: {avg_wind:.1f}, class: {wind_pot['wind_class']}")
    else:
        print("  Wind SKIP - no hourly data")
except Exception as e:
    print(f"  Wind ERROR: {e}")
    traceback.print_exc()

# Test tab 3 logic (Solar)
print("Tab 3 (Solar Energy)...")
try:
    if solar and "daily" in solar and solar["daily"] is not None:
        ds = solar["daily"]
        avg_rad = ds["radiation_sum"].mean() / 1000
        avg_psh = ds["peak_sun_hours"].mean()
        soiling = calculate_soiling_loss(region_name="Kinshasa", days_since_cleaning=30)
        sp = calculate_solar_potential(ghi_kwh_m2_day=avg_rad, soiling_loss=soiling["soiling_loss_percent"]/100, ambient_temp=28)
        ot = calculate_optimal_tilt(LAT)
        rc = get_solar_resource_class(avg_rad)
        print(f"  Solar OK - rad: {avg_rad:.2f}, PSH: {avg_psh:.1f}, class: {rc}")
        print(f"  Soiling: {soiling['soiling_loss_percent']:.1f}%, net: {sp['net_energy_kwh_day']:.1f} kWh")
    else:
        print("  Solar SKIP - no daily data")
except Exception as e:
    print(f"  Solar ERROR: {e}")
    traceback.print_exc()

# Test tab 4 logic (Air Quality)
print("Tab 4 (Air Quality)...")
try:
    if aq and "current" in aq:
        caq = aq["current"]
        aqi_val = caq.get("us_aqi") or caq.get("european_aqi", 0)
        print(f"  AQ OK - AQI: {aqi_val}, category: {caq.get('aqi_category')}")
        impact = analyze_air_quality_for_solar(aq)
        print(f"  Solar impact available: {impact.get('impact_available')}")
    else:
        print("  AQ SKIP - no current data")
except Exception as e:
    print(f"  AQ ERROR: {e}")
    traceback.print_exc()

# Test tab 5 logic (Alerts)
print("Tab 5 (Alerts)...")
try:
    if weather:
        alerts = analyze_weather_alerts(weather)
        aq_alerts = generate_air_quality_alerts(aq) if aq else []
        print(f"  Alerts OK - weather: {len(alerts)}, AQ: {len(aq_alerts)}")
        if weather:
            impacts = analyze_renewable_energy_impact(weather, alerts)
            print(f"  Impacts OK - solar: {len(impacts['solar'])}, wind: {len(impacts['wind'])}")
    else:
        print("  Alerts SKIP - no weather data")
except Exception as e:
    print(f"  Alerts ERROR: {e}")
    traceback.print_exc()

# Test tab 6 logic (Map)
print("Tab 6 (Map)...")
try:
    m = create_comprehensive_map(LAT, LON, "Kinshasa", weather, aq, solar, True)
    print(f"  Map OK - type: {type(m).__name__}")
except Exception as e:
    print(f"  Map ERROR: {e}")
    traceback.print_exc()

# Test tab 7 logic (Forecast Comparison)
print("Tab 7 (Forecast Comparison)...")
try:
    if weather and "hourly" in weather:
        comp = run_forecast_comparison(weather, "temperature", 24, 48)
        print(f"  Forecast OK - history: {len(comp['history'])}, api_future: {len(comp['api_future'])}")
        print(f"  Internal forecasts: {len(comp['internal_forecasts'])}")
        print(f"  Metrics: {list(comp['metrics'].keys())}")
    else:
        print("  Forecast SKIP - no hourly data")
except Exception as e:
    print(f"  Forecast ERROR: {e}")
    traceback.print_exc()

# Test plotly rendering
print("\n=== PLOTLY TESTS ===")
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=[1,2,3], y=[1,2,3], name="test"), secondary_y=False)
    fig.add_trace(go.Bar(x=[1,2,3], y=[3,2,1], name="test2"), secondary_y=True)
    print("make_subplots OK")
    
    fig2 = go.Figure(go.Indicator(mode="gauge+number", value=5.0, gauge={"axis": {"range": [0, 30]}}))
    print("Indicator gauge OK")
    
    import plotly.express as px
    fig3 = px.histogram(x=[10,20,30,40], nbins=4)
    print("px.histogram OK")
except Exception as e:
    print(f"PLOTLY ERROR: {e}")
    traceback.print_exc()

print("\n=== ALL TESTS COMPLETE ===")
