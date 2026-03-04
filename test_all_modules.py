#!/usr/bin/env python
"""
End-to-end smoke test for every module in the Renewable Energy Dashboard.
Tests both a predefined region (Casablanca) and a custom coordinate set.
"""
import sys, os, traceback, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL, SKIP = 0, 0, 0

def banner(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")

def check(label, fn):
    global PASS, FAIL, SKIP
    try:
        result = fn()
        if result == "SKIP":
            SKIP += 1
            print(f"  [SKIP] {label}")
        else:
            PASS += 1
            print(f"  [ OK ] {label}")
    except Exception as e:
        FAIL += 1
        print(f"  [FAIL] {label}: {e}")
        traceback.print_exc()

# ── Coordinates ─────────────────────────────────────────────
PREDEFINED = {"name": "Casablanca", "lat": 33.5731, "lon": -7.5898}
CUSTOM     = {"name": "Custom (Nairobi)", "lat": -1.2921, "lon": 36.8219}

# ====================================================================
# 1. weather_wind
# ====================================================================
banner("1. weather_wind")
from modules.weather_wind import fetch_weather_data, process_weather_data
weather_data = {}

def test_weather(coords):
    global weather_data
    raw = fetch_weather_data(coords["lat"], coords["lon"])
    assert raw is not None, "fetch returned None"
    wd = process_weather_data(raw)
    assert "hourly" in wd, "no hourly key"
    assert len(wd["hourly"]) > 0, "hourly df empty"
    weather_data[coords["name"]] = wd
    print(f"           -> {len(wd['hourly'])} hourly rows")

check(f"fetch+process [{PREDEFINED['name']}]", lambda: test_weather(PREDEFINED))
check(f"fetch+process [{CUSTOM['name']}]",     lambda: test_weather(CUSTOM))

# ====================================================================
# 2. solar_energy
# ====================================================================
banner("2. solar_energy")
from modules.solar_energy import (
    fetch_solar_radiation_data, fetch_nasa_power_data,
    process_solar_data, calculate_solar_potential,
)

def test_solar(coords):
    raw = fetch_solar_radiation_data(coords["lat"], coords["lon"])
    assert raw is not None, "solar radiation fetch returned None"
    sd = process_solar_data(raw)
    assert isinstance(sd, dict), f"process_solar_data returned {type(sd)}"
    # calculate_solar_potential takes a float (GHI kWh/m2/day), not a dict
    pot = calculate_solar_potential(ghi_kwh_m2_day=5.5)
    assert isinstance(pot, dict), f"calculate_solar_potential returned {type(pot)}"
    print(f"           -> solar potential: {pot.get('net_energy_kwh_day', 'N/A'):.1f} kWh/day")

check(f"solar radiation [{PREDEFINED['name']}]", lambda: test_solar(PREDEFINED))
check(f"solar radiation [{CUSTOM['name']}]",     lambda: test_solar(CUSTOM))

# NASA POWER (may be slow / rate-limited)
def test_nasa(coords):
    try:
        nd = fetch_nasa_power_data(coords["lat"], coords["lon"])
        if nd is None:
            return "SKIP"
        print(f"           -> NASA POWER returned ok")
    except Exception:
        return "SKIP"

check(f"NASA POWER [{PREDEFINED['name']}]", lambda: test_nasa(PREDEFINED))

# ====================================================================
# 3. air_quality
# ====================================================================
banner("3. air_quality")
from modules.air_quality import fetch_air_quality_data, process_air_quality_data

def test_aq(coords):
    raw = fetch_air_quality_data(coords["lat"], coords["lon"])
    assert raw is not None, "AQ fetch returned None"
    aq = process_air_quality_data(raw)
    assert aq is not None, "AQ process returned None"
    print(f"           -> AQ rows: {len(aq) if hasattr(aq, '__len__') else 'dict'}")

check(f"air quality [{PREDEFINED['name']}]", lambda: test_aq(PREDEFINED))
check(f"air quality [{CUSTOM['name']}]",     lambda: test_aq(CUSTOM))

# ====================================================================
# 4. extreme_weather
# ====================================================================
banner("4. extreme_weather")
from modules.extreme_weather import analyze_weather_alerts, analyze_renewable_energy_impact

alert_results = {}

def test_alerts_store(coords):
    name = coords["name"]
    wd = weather_data.get(name)
    if wd is None:
        return "SKIP"
    alerts = analyze_weather_alerts(wd)
    alert_results[name] = alerts
    assert isinstance(alerts, list)
    print(f"           -> {len(alerts)} alerts")

def test_impact(coords):
    name = coords["name"]
    wd = weather_data.get(name)
    alerts = alert_results.get(name, [])
    if wd is None:
        return "SKIP"
    impact = analyze_renewable_energy_impact(wd, alerts)
    assert impact is not None
    print(f"           -> impact type: {type(impact).__name__}")

check(f"alerts [{PREDEFINED['name']}]",    lambda: test_alerts_store(PREDEFINED))
check(f"alerts [{CUSTOM['name']}]",        lambda: test_alerts_store(CUSTOM))
check(f"RE impact [{PREDEFINED['name']}]", lambda: test_impact(PREDEFINED))
check(f"RE impact [{CUSTOM['name']}]",     lambda: test_impact(CUSTOM))

# ====================================================================
# 5. map_visualization
# ====================================================================
banner("5. map_visualization")
from modules.map_visualization import create_comprehensive_map, create_regions_overview_map

def test_map_comp(coords):
    wd = weather_data.get(coords["name"])
    if wd is None:
        return "SKIP"
    m = create_comprehensive_map(coords["lat"], coords["lon"], wd)
    assert m is not None, "map is None"
    print(f"           -> map created")

def test_overview_map():
    m = create_regions_overview_map()
    assert m is not None
    print(f"           -> overview map created")

check(f"comprehensive map [{PREDEFINED['name']}]", lambda: test_map_comp(PREDEFINED))
check(f"comprehensive map [{CUSTOM['name']}]",     lambda: test_map_comp(CUSTOM))
check("regions overview map", test_overview_map)

# ====================================================================
# 6. forecasting (SARIMAX, XGBoost, LSTM)
# ====================================================================
banner("6. forecasting (SARIMAX / XGBoost / LSTM)")
from modules.forecasting import (
    run_forecast_comparison, available_methods,
    persistence_forecast, sarimax_forecast, xgboost_forecast,
    lstm_forecast, ensemble_forecast, compute_accuracy_metrics,
)

avail = available_methods()
print(f"  Backend availability: {avail}")

def test_forecast(coords):
    wd = weather_data.get(coords["name"])
    if wd is None:
        return "SKIP"
    result = run_forecast_comparison(wd, target_col="temperature", horizon_hours=12, history_hours=48)
    assert "internal_forecasts" in result, "missing internal_forecasts"
    idf = result["internal_forecasts"]
    if idf.empty:
        print("           -> internal_forecasts empty (not enough data?)")
        return
    methods_found = idf["method"].unique().tolist()
    print(f"           -> methods: {methods_found}")
    if result["metrics"]:
        for mname, vals in result["metrics"].items():
            r2 = vals.get("R2", vals.get("R\u00b2", "-"))
            r2_str = f"{r2:.3f}" if isinstance(r2, (int, float)) else str(r2)
            print(f"              {mname}: MAE={vals.get('MAE',0):.2f}  RMSE={vals.get('RMSE',0):.2f}  R2={r2_str}")

check(f"forecast comparison [{PREDEFINED['name']}]", lambda: test_forecast(PREDEFINED))
check(f"forecast comparison [{CUSTOM['name']}]",     lambda: test_forecast(CUSTOM))

# Individual method tests with dummy data
import pandas as pd
import numpy as np

def make_dummy_series(n=120):
    t = pd.date_range("2025-01-01", periods=n, freq="h")
    vals = 20 + 5*np.sin(np.arange(n)*2*np.pi/24) + np.random.normal(0, 1, n)
    return pd.DataFrame({"time": t, "temperature": vals})

dummy = make_dummy_series(120)

def test_persistence():
    r = persistence_forecast(dummy, "temperature", 24)
    assert len(r) == 24 and "method" in r.columns
    print(f"           -> {len(r)} rows, method={r['method'].iloc[0]}")

def test_sarimax():
    r = sarimax_forecast(dummy, "temperature", 24)
    assert len(r) == 24 and "method" in r.columns
    print(f"           -> {len(r)} rows, method={r['method'].iloc[0]}")

def test_xgboost():
    r = xgboost_forecast(dummy, "temperature", 24)
    assert len(r) == 24 and "method" in r.columns, f"XGBoost returned {len(r)} rows"
    print(f"           -> {len(r)} rows, method={r['method'].iloc[0]}")

def test_lstm():
    r = lstm_forecast(dummy, "temperature", 24)
    assert len(r) == 24 and "method" in r.columns, f"LSTM returned {len(r)} rows"
    print(f"           -> {len(r)} rows, method={r['method'].iloc[0]}")

def test_ensemble():
    r = ensemble_forecast(dummy, "temperature", 24)
    methods = r["method"].unique().tolist()
    assert "Ensemble Mean" in methods, f"no Ensemble Mean in {methods}"
    print(f"           -> methods: {methods}, total rows: {len(r)}")

check("persistence (dummy)", test_persistence)
check("SARIMAX (dummy)",     test_sarimax)
check("XGBoost (dummy)",     test_xgboost)
check("LSTM (dummy)",        test_lstm)
check("ensemble (dummy)",    test_ensemble)

# Accuracy metrics
def test_metrics():
    # Properly split: first 96 rows as history, forecast into the remaining 24
    hist = dummy.iloc[:96].reset_index(drop=True)
    api_fut = dummy.iloc[96:][["time", "temperature"]].rename(columns={"temperature": "value"}).reset_index(drop=True)
    ens = ensemble_forecast(hist, "temperature", 24)
    m = compute_accuracy_metrics(api_fut, ens)
    assert isinstance(m, dict) and len(m) > 0, f"metrics result: {m}"
    for method, vals in m.items():
        print(f"           -> {method}: MAE={vals.get('MAE',0):.2f}  RMSE={vals.get('RMSE',0):.2f}")

check("accuracy metrics (dummy)", test_metrics)

# ====================================================================
# 7. wind_resource
# ====================================================================
banner("7. wind_resource")
from modules.wind_resource import (
    weibull_fit, extrapolate_wind_speed, TURBINE_CATALOG,
    capacity_factor as wind_cf,
)

def test_wind_resource(coords):
    wd = weather_data.get(coords["name"])
    if wd is None:
        return "SKIP"
    hourly = wd["hourly"]
    if "wind_speed" not in hourly.columns:
        return "SKIP"
    ws = hourly["wind_speed"].dropna().values
    if len(ws) < 10:
        return "SKIP"
    k, A = weibull_fit(ws)
    print(f"           -> Weibull k={k:.2f}, A={A:.2f}")
    ws100 = extrapolate_wind_speed(ws.mean(), 10, 100, alpha=0.14)
    print(f"           -> Extrapolated mean wind @ 100m: {ws100:.2f} km/h")
    turb = list(TURBINE_CATALOG.keys())[0]
    cf = wind_cf(ws, TURBINE_CATALOG[turb])
    print(f"           -> CF ({turb}): {cf:.2%}")

check(f"wind resource [{PREDEFINED['name']}]", lambda: test_wind_resource(PREDEFINED))
check(f"wind resource [{CUSTOM['name']}]",     lambda: test_wind_resource(CUSTOM))

# ====================================================================
# 8. solar_resource
# ====================================================================
banner("8. solar_resource")
from modules.solar_resource import (
    pv_power_output, compute_solar_lcoe, assess_solar_resource,
    PVModule, optimal_tilt_angle,
)

def test_solar_resource(coords):
    tilt = optimal_tilt_angle(coords["lat"])
    print(f"           -> optimal tilt: {tilt}")
    lcoe = compute_solar_lcoe(system_kwp=10, annual_kwh=15000)
    assert lcoe > 0, f"LCOE should be positive, got {lcoe}"
    print(f"           -> Solar LCOE: ${lcoe:.4f}/kWh")

check(f"solar resource [{PREDEFINED['name']}]", lambda: test_solar_resource(PREDEFINED))
check(f"solar resource [{CUSTOM['name']}]",     lambda: test_solar_resource(CUSTOM))

# ====================================================================
# 9. hybrid_hydrogen
# ====================================================================
banner("9. hybrid_hydrogen")
from modules.hybrid_hydrogen import (
    ELECTROLYSER_CATALOG, hybrid_dispatch, compute_lcoh,
)

def test_hydrogen():
    wind_kw = np.array([50 + 30*np.sin(2*np.pi*h/24) for h in range(24)])
    solar_kw = np.array([max(0, 40*np.sin(np.pi*(h-6)/12)) if 6 <= h <= 18 else 0 for h in range(24)])
    cat_name = list(ELECTROLYSER_CATALOG.keys())[0]
    elec = ELECTROLYSER_CATALOG[cat_name]
    disp = hybrid_dispatch(wind_kw, solar_kw, elec)
    assert disp is not None
    h2_kg = disp.get("total_h2_kg", 0)
    print(f"           -> dispatch: {h2_kg:.2f} kg H2 ({cat_name})")
    if h2_kg > 0:
        cost = compute_lcoh(elec, n_electrolysers=1, annual_h2_kg=h2_kg*365,
                            wind_capex=500000, solar_capex=200000)
        print(f"           -> LCOH: ${cost.get('lcoh', 0):.2f}/kg")

check("hydrogen dispatch + LCOH", test_hydrogen)

# ====================================================================
# Summary
# ====================================================================
banner("SUMMARY")
total = PASS + FAIL + SKIP
print(f"  PASS: {PASS}   FAIL: {FAIL}   SKIP: {SKIP}   TOTAL: {total}")
if FAIL == 0:
    print("  All tests passed!")
else:
    print("  Some tests failed - see details above.")
sys.exit(FAIL)
