# =============================================================================
# Advanced Forecasting Module
# SARIMAX  ·  XGBoost  ·  LSTM   (+ Persistence baseline & Ensemble)
# =============================================================================
#
# Methods implemented:
#   1. Persistence Model        – Naive baseline (last value repeated)
#   2. SARIMAX                  – Seasonal ARIMA with eXogenous capability
#   3. XGBoost                  – Gradient-boosted trees with lag features
#   4. LSTM (PyTorch)           – Recurrent neural network
#   5. Ensemble Mean            – Weighted average of all methods
#
# Each method returns a DataFrame with columns: time, value, method
# =============================================================================

import warnings
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Suppress convergence / future warnings from statsmodels during fitting
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Resolution helpers ---------------------------------------------------
_FREQ_TD = {
    "h": timedelta(hours=1),
    "15min": timedelta(minutes=15),
    "5min": timedelta(minutes=5),
    "1min": timedelta(minutes=1),
    "D": timedelta(days=1),
}
_FREQ_SPD = {
    "h": 24,
    "15min": 96,
    "5min": 288,
    "1min": 1440,
    "D": 1,
}  # steps per day

def _td(freq: str) -> timedelta:
    return _FREQ_TD.get(freq, timedelta(hours=1))

def _spd(freq: str) -> int:
    return _FREQ_SPD.get(freq, 24)

# ---------------------------------------------------------------------------
# Optional heavy imports – wrapped so the module loads even if a dep is missing
# ---------------------------------------------------------------------------
_HAS_SARIMAX = False
_HAS_XGB = False
_HAS_TORCH = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX
    _HAS_SARIMAX = True
except ImportError:
    pass

try:
    import xgboost as xgb
    # XGBRegressor needs sklearn; verify both are available
    from xgboost import XGBRegressor as _XGBRegressor  # noqa: F401
    _HAS_XGB = True
except (ImportError, LookupError, ModuleNotFoundError):
    pass

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    pass


# ========================================================================== #
# 1.  Persistence Model  (baseline)
# ========================================================================== #

def persistence_forecast(
    historical: pd.DataFrame,
    target_col: str,
    horizon_hours: int = 24,
    freq: str = "h",
) -> pd.DataFrame:
    """Last-observed-value repeated forward (naive baseline)."""
    if historical.empty or target_col not in historical.columns:
        return pd.DataFrame(columns=["time", "value", "method"])
    last_time = historical["time"].iloc[-1]
    last_value = float(historical[target_col].iloc[-1])
    future = pd.date_range(start=last_time + _td(freq),
                           periods=horizon_hours, freq=freq)
    return pd.DataFrame({"time": future, "value": last_value,
                         "method": "Persistence"})


# ========================================================================== #
# 2.  SARIMAX
# ========================================================================== #

def sarimax_forecast(
    historical: pd.DataFrame,
    target_col: str,
    horizon_hours: int = 24,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 0, 24),
    exog_cols: Optional[List[str]] = None,
    freq: str = "h",
) -> pd.DataFrame:
    """
    Seasonal ARIMA(X) forecast.

    Parameters
    ----------
    order : (p, d, q)  – non-seasonal ARIMA order
    seasonal_order : (P, D, Q, s)  – seasonal component (auto-set from freq)
    exog_cols : optional exogenous variable column names
    freq : pandas frequency string ('h', '15min', 'D')
    """
    if not _HAS_SARIMAX:
        return pd.DataFrame(columns=["time", "value", "method"])
    if historical.empty or target_col not in historical.columns:
        return pd.DataFrame(columns=["time", "value", "method"])

    spd = _spd(freq)
    min_len = max(48, spd * 2) if freq != "D" else 14
    if len(historical) < min_len:
        return pd.DataFrame(columns=["time", "value", "method"])

    series = historical[target_col].astype(float).values
    last_time = historical["time"].iloc[-1]

    # Override seasonal period to match resolution
    seasonal_order = (seasonal_order[0], seasonal_order[1],
                      seasonal_order[2], spd if spd > 1 else 7)

    # Exogenous matrix (if columns supplied)
    exog = None
    if exog_cols:
        valid = [c for c in exog_cols if c in historical.columns]
        if valid:
            exog = historical[valid].astype(float).values

    try:
        n = len(series)
        s = seasonal_order[3]
        # Fall back to plain ARIMA if not enough data for full seasonal cycle
        if n < 2 * s:
            seasonal_order = (0, 0, 0, 0)

        model = _SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            exog=exog,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=100)
        pred = fit.forecast(steps=horizon_hours)
    except Exception:
        return pd.DataFrame(columns=["time", "value", "method"])

    future_times = pd.date_range(start=last_time + _td(freq),
                                 periods=horizon_hours, freq=freq)
    return pd.DataFrame({"time": future_times, "value": pred,
                         "method": "SARIMAX"})


# ========================================================================== #
# 3.  XGBoost  (gradient-boosted trees with lag features)
# ========================================================================== #

def _build_xgb_features(
    series: np.ndarray,
    lags: List[int],
    frac_of_day: Optional[np.ndarray] = None,
) -> tuple:
    """Build feature matrix [lag_1 … lag_k, day_sin, day_cos]."""
    max_lag = max(lags)
    n = len(series)
    rows = []
    for i in range(max_lag, n):
        row = [series[i - lag] for lag in lags]
        if frac_of_day is not None:
            f = frac_of_day[i]
            row.append(np.sin(2 * np.pi * f))
            row.append(np.cos(2 * np.pi * f))
        rows.append(row)
    return np.array(rows, dtype=np.float32), max_lag


def xgboost_forecast(
    historical: pd.DataFrame,
    target_col: str,
    horizon_hours: int = 24,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.08,
    lags: Optional[List[int]] = None,
    freq: str = "h",
) -> pd.DataFrame:
    """
    XGBoost regression with lagged features + time-of-day cyclical encoding.
    Uses recursive multi-step prediction.  `freq` controls step size.
    """
    if not _HAS_XGB:
        return pd.DataFrame(columns=["time", "value", "method"])
    if historical.empty or target_col not in historical.columns:
        return pd.DataFrame(columns=["time", "value", "method"])

    spd = _spd(freq)
    if lags is None:
        if freq == "1min":
            lags = [1, 5, 15, 60, 120, 1440]   # 1m, 5m, 15m, 1h, 2h, 24h
        elif freq == "5min":
            lags = [1, 3, 12, 24, 48, 288]     # 5m, 15m, 1h, 2h, 4h, 24h
        elif freq == "15min":
            lags = [1, 2, 4, 12, 24, 96]       # 15m, 30m, 1h, 3h, 6h, 24h
        elif freq == "D":
            lags = [1, 2, 3, 7]                 # 1d, 2d, 3d, 7d
        else:
            lags = [1, 2, 3, 6, 12, 24]        # standard hourly
    max_lag = max(lags)
    if len(historical) < max_lag + 10:
        return pd.DataFrame(columns=["time", "value", "method"])

    series = historical[target_col].astype(float).values
    # Fractional position within the day (0-1), works for any resolution
    times = historical["time"]
    frac = (times.dt.hour + times.dt.minute / 60) / 24.0
    frac_arr = frac.values.astype(np.float64)

    X, ml = _build_xgb_features(series, lags, frac_arr)
    y = series[ml:]

    try:
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            verbosity=0,
            random_state=42,
        )
        model.fit(X, y)
    except Exception:
        return pd.DataFrame(columns=["time", "value", "method"])

    # Recursive multi-step prediction
    last_time = historical["time"].iloc[-1]
    extended = list(series)
    step_td = _td(freq)
    preds = []

    for step in range(horizon_hours):
        ft = last_time + step_td * (step + 1)
        cur_frac = (ft.hour + ft.minute / 60) / 24.0
        row = [extended[-lag] for lag in lags]
        row.append(np.sin(2 * np.pi * cur_frac))
        row.append(np.cos(2 * np.pi * cur_frac))
        x_in = np.array([row], dtype=np.float32)
        yhat = float(model.predict(x_in)[0])
        preds.append(yhat)
        extended.append(yhat)

    future_times = pd.date_range(start=last_time + step_td,
                                 periods=horizon_hours, freq=freq)
    return pd.DataFrame({"time": future_times, "value": preds,
                         "method": "XGBoost"})


# ========================================================================== #
# 4.  LSTM  (PyTorch – lightweight single-layer)
# ========================================================================== #

if _HAS_TORCH:

    class _LSTMNet(nn.Module):
        """Minimal LSTM for univariate time-series forecasting."""

        def __init__(self, input_size: int = 1, hidden_size: int = 48,
                     num_layers: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out


def lstm_forecast(
    historical: pd.DataFrame,
    target_col: str,
    horizon_hours: int = 24,
    lookback: int = 48,
    hidden_size: int = 48,
    epochs: int = 30,
    lr: float = 0.005,
    freq: str = "h",
) -> pd.DataFrame:
    """
    Train a lightweight LSTM on the historical series, then predict
    `horizon_hours` steps ahead using recursive one-step prediction.
    """
    if not _HAS_TORCH:
        return pd.DataFrame(columns=["time", "value", "method"])
    if historical.empty or target_col not in historical.columns:
        return pd.DataFrame(columns=["time", "value", "method"])
    # Scale lookback per resolution: finer resolutions need more steps, daily fewer
    if lookback == 48:  # caller used default – auto-adjust
        lookback = {"h": 48, "15min": 96, "5min": 120, "1min": 180, "D": 14}.get(freq, 48)
    if len(historical) < lookback + 10:
        return pd.DataFrame(columns=["time", "value", "method"])

    series = historical[target_col].astype(float).values

    # Normalise to [0, 1]
    s_min, s_max = float(series.min()), float(series.max())
    s_range = s_max - s_min
    if s_range < 1e-9:
        s_range = 1.0
    normed = (series - s_min) / s_range

    # Build sliding-window dataset
    X_seq, y_seq = [], []
    for i in range(len(normed) - lookback):
        X_seq.append(normed[i: i + lookback])
        y_seq.append(normed[i + lookback])

    X_t = torch.tensor(np.array(X_seq), dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(np.array(y_seq), dtype=torch.float32).unsqueeze(-1)

    # Model / optimiser
    model = _LSTMNet(input_size=1, hidden_size=hidden_size, num_layers=1)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    ds = X_t.shape[0]
    bs = min(64, ds)
    try:
        for _ep in range(epochs):
            perm = torch.randperm(ds)
            for start in range(0, ds, bs):
                idx = perm[start: start + bs]
                xb, yb = X_t[idx], y_t[idx]
                optimiser.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimiser.step()
    except Exception:
        return pd.DataFrame(columns=["time", "value", "method"])

    # Recursive prediction
    model.eval()
    window = list(normed[-lookback:])
    preds_norm: list = []
    with torch.no_grad():
        for _ in range(horizon_hours):
            x_in = torch.tensor([window[-lookback:]],
                                dtype=torch.float32).unsqueeze(-1)
            yhat = model(x_in).item()
            preds_norm.append(yhat)
            window.append(yhat)

    preds = np.array(preds_norm) * s_range + s_min
    last_time = historical["time"].iloc[-1]
    future_times = pd.date_range(start=last_time + _td(freq),
                                 periods=horizon_hours, freq=freq)
    return pd.DataFrame({"time": future_times, "value": preds,
                         "method": "LSTM"})


# ========================================================================== #
# Ensemble  (all available methods)
# ========================================================================== #

def ensemble_forecast(
    historical: pd.DataFrame,
    target_col: str,
    horizon_hours: int = 24,
    methods: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    freq: str = "h",
) -> pd.DataFrame:
    """
    Run selected forecast methods and compute a (weighted) ensemble mean.

    Parameters
    ----------
    methods : list of method names to include.  None = all available.
    weights : optional dict  method_name -> weight.  Defaults equal.
    freq : pandas frequency string ('h', '15min', 'D').

    Returns
    -------
    DataFrame (time, value, method) — one block per method + 'Ensemble Mean'.
    """
    all_methods: Dict[str, Any] = {
        "Persistence": persistence_forecast,
        "SARIMAX": sarimax_forecast,
        "XGBoost": xgboost_forecast,
        "LSTM": lstm_forecast,
    }

    if methods is not None:
        all_methods = {k: v for k, v in all_methods.items() if k in methods}

    frames: List[pd.DataFrame] = []
    for name, fn in all_methods.items():
        try:
            result = fn(historical, target_col, horizon_hours, freq=freq)
            if not result.empty:
                frames.append(result)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(columns=["time", "value", "method"])

    combined = pd.concat(frames, ignore_index=True)

    # Weighted ensemble mean
    if weights is None:
        ens = combined.groupby("time")["value"].mean().reset_index()
    else:
        def _wmean(grp):
            tw, ws = 0.0, 0.0
            for _, row in grp.iterrows():
                w = weights.get(row["method"], 1.0)
                ws += w * row["value"]
                tw += w
            return ws / max(tw, 1e-9)
        ens = combined.groupby("time").apply(_wmean).reset_index()
        ens.columns = ["time", "value"]

    ens["method"] = "Ensemble Mean"
    combined = pd.concat([combined, ens], ignore_index=True)
    return combined


# ========================================================================== #
# Accuracy metrics
# ========================================================================== #

def compute_accuracy_metrics(
    api_forecast: pd.DataFrame,
    internal_forecast: pd.DataFrame,
    api_col: str = "value",
    internal_col: str = "value",
) -> Dict[str, Dict[str, float]]:
    """
    Compare internal forecast methods against the API forecast.

    Returns dict[method_name] -> {MAE, RMSE, MAPE, Bias, R2}
    """
    metrics: Dict[str, Dict[str, float]] = {}
    if api_forecast.empty or internal_forecast.empty:
        return metrics

    for method_name, group in internal_forecast.groupby("method"):
        merged = pd.merge(
            api_forecast[["time", api_col]].rename(columns={api_col: "api"}),
            group[["time", internal_col]].rename(columns={internal_col: "internal"}),
            on="time", how="inner",
        )
        if merged.empty:
            continue

        errors = merged["internal"] - merged["api"]
        abs_errors = errors.abs()

        mae = float(abs_errors.mean())
        rmse = float(np.sqrt((errors ** 2).mean()))
        bias = float(errors.mean())

        # MAPE
        nonzero = merged["api"].abs() > 0.01
        mape = float((abs_errors[nonzero] / merged["api"][nonzero].abs()).mean()
                      * 100) if nonzero.any() else float("nan")

        # R-squared
        ss_res = float((errors ** 2).sum())
        ss_tot = float(((merged["api"] - merged["api"].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-9)

        metrics[str(method_name)] = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "Bias": round(bias, 2),
            "MAPE (%)": round(mape, 1),
            "R\u00b2": round(r2, 3),
        }

    return metrics


# ========================================================================== #
# Helpers
# ========================================================================== #

def api_forecast_from_weather(
    weather_data: Dict[str, Any], target_col: str,
) -> pd.DataFrame:
    """Extract API-provided hourly forecast for comparison."""
    hourly = weather_data.get("hourly")
    if hourly is None or not isinstance(hourly, pd.DataFrame):
        return pd.DataFrame(columns=["time", "value"])
    if target_col not in hourly.columns:
        return pd.DataFrame(columns=["time", "value"])
    df = hourly[["time", target_col]].dropna().copy()
    return df.rename(columns={target_col: "value"})


def available_methods() -> Dict[str, bool]:
    """Report which forecast back-ends are available."""
    return {
        "Persistence": True,
        "SARIMAX": _HAS_SARIMAX,
        "XGBoost": _HAS_XGB,
        "LSTM": _HAS_TORCH,
    }


# ========================================================================== #
# End-to-end comparison
# ========================================================================== #

def run_forecast_comparison(
    weather_data: Dict[str, Any],
    target_col: str,
    horizon_hours: int = 24,
    history_hours: int = 48,
    freq: str = "h",
    ts_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    1. Split time-series data into history + future windows.
    2. Run internal forecast methods on the history window.
    3. Compute accuracy metrics against the API future window.

    Parameters
    ----------
    ts_df : If provided, use this DataFrame directly instead of
            weather_data["hourly"].  Useful when the caller has already
            resolved the active resolution DataFrame.
    freq : pandas frequency string ('h', '15min', 'D').

    Returns
    -------
    dict  history, api_future, internal_forecasts, metrics, availability
    """
    empty = {"history": pd.DataFrame(), "api_future": pd.DataFrame(),
             "internal_forecasts": pd.DataFrame(), "metrics": {},
             "availability": available_methods()}

    source = ts_df if ts_df is not None else weather_data.get("hourly")
    if source is None or not isinstance(source, pd.DataFrame):
        return empty
    if target_col not in source.columns:
        return empty

    df = source[["time", target_col]].dropna().copy()
    total = len(df)

    history_end = min(history_hours, total - horizon_hours)
    if history_end < 6:
        history_end = total // 2
    future_start = history_end
    actual_horizon = total - future_start
    if actual_horizon < 1:
        return empty

    history_df = df.iloc[:history_end].reset_index(drop=True)
    future_df = df.iloc[future_start:].reset_index(drop=True)
    api_future = future_df.rename(columns={target_col: "value"})

    internal = ensemble_forecast(history_df, target_col, actual_horizon,
                                 freq=freq)
    metrics = compute_accuracy_metrics(api_future, internal)

    return {
        "history": history_df,
        "api_future": api_future,
        "internal_forecasts": internal,
        "metrics": metrics,
        "availability": available_methods(),
    }
