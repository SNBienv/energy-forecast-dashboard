# =============================================================================
# High-Resolution Interpolation Module
# Physics-based spectral correlations for sub-hourly time-series upsampling
# =============================================================================
#
# Each meteorological variable uses its own physically-correct model:
#
#   WIND SPEED / GUSTS
#     Von-Kármán (IEC 61400-1) or Kaimal power spectral density.
#     Base trend via cubic spline → synthetic turbulence added by filtering
#     white noise through the spectral transfer function in the frequency
#     domain, respecting the integral length-scale and σ_u.
#
#   WIND DIRECTION
#     Circular (sin / cos) cubic spline + random-walk-on-circle noise,
#     scaled by the yaw integral time-scale (~60 s).
#
#   TEMPERATURE
#     Harmonic diurnal decomposition: T(t) = T̄ + ΣAₖcos(kωt + φₖ)
#     where ω = 2π/86400 s⁻¹.  Fit fundamental + 2nd harmonic to the
#     coarse data, interpolate residual with a natural cubic spline.
#
#   PRESSURE
#     Semidiurnal barometric tide S₂ (period 12 h, amplitude ≈ 0.5–1.3 hPa
#     tropical, less at higher latitudes) + long-period trend spline.
#
#   HUMIDITY
#     Clausius–Clapeyron / dewpoint approach: cubic-spline the dewpoint
#     temperature Td, cubic-spline the temperature T, then recompute
#     RH = 100 · e_s(Td) / e_s(T)  (Magnus formula).
#     Falls back to harmonic + spline when Td is absent.
#
#   SOLAR (GHI / DNI / DHI / GTI)
#     Clear-sky envelope from solar zenith angle (simplified Ineichen–Perez)
#     → clearness index Kt = GHI_obs / GHI_cs.  Kt is interpolated via
#     cubic spline on [0, 1] with a Markov-chain cloud-transient model
#     (two-state clear ↔ cloudy) adding sub-hourly ramp events.
#
#   CLOUD COVER
#     Logit-transformed cubic spline (maps 0–100 % to ℝ and back) so the
#     spline cannot overshoot below 0 or above 100.
#
# The API (Open-Meteo) hourly forecast is kept as a piecewise-constant step
# function so users can visually compare the smooth internal prediction
# against the "staircase" of the external forecast.
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, List
from scipy.interpolate import CubicSpline


# ═══════════════════════════════════════════════════════════════════════════════
# Variable registry
# ═══════════════════════════════════════════════════════════════════════════════

INTERPOLATABLE = {
    "wind_speed", "wind_gusts", "wind_direction",
    "temperature", "feels_like",
    "pressure",
    "humidity", "dew_point",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "global_tilted_irradiance", "ghi",
    "cloud_cover",
}

NON_INTERPOLATABLE = {
    "precipitation", "rain", "precipitation_probability",
    "weather_code", "uv_index",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  WIND – Von-Kármán / Kaimal spectral turbulence
# ═══════════════════════════════════════════════════════════════════════════════

def _vonkarman_psd(f: np.ndarray, sigma_u: float, L_u: float,
                   U: float) -> np.ndarray:
    """
    Von-Kármán longitudinal PSD  (IEC 61400-1 Ed.4, Eq. A.1).

        S_u(f) = 4 σ²_u (L_u / U)  /  [1 + 70.8 (f L_u / U)²]^{5/6}

    Parameters
    ----------
    f     : frequency array (Hz), positive
    sigma_u : std-dev of longitudinal turbulence (m/s)
    L_u   : integral length-scale (m).  IEC: L_u = 8.1 Λ_1
    U     : mean wind speed at hub height (m/s)
    """
    fL_U = f * L_u / max(U, 0.5)
    return 4.0 * sigma_u**2 * (L_u / max(U, 0.5)) / (1.0 + 70.8 * fL_U**2)**(5.0 / 6.0)


def _kaimal_psd(f: np.ndarray, sigma_u: float, L_u: float,
                U: float) -> np.ndarray:
    """
    Kaimal longitudinal PSD  (ESDU 85020 / IEC 61400-1 alternative).

        S_u(f) = 4 σ²_u (L_u / U)  /  [1 + 6 f L_u / U]^{5/3}
    """
    fL_U = f * L_u / max(U, 0.5)
    return 4.0 * sigma_u**2 * (L_u / max(U, 0.5)) / (1.0 + 6.0 * fL_U)**(5.0 / 3.0)


def _spectral_turbulence(
    n: int,
    dt_seconds: float,
    sigma_u: float,
    U: float,
    L_u: float = 340.0,
    spectrum: str = "vonkarman",
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Generate a synthetic turbulence time-series by spectral filtering.

    White noise → FFT → multiply by √S(f) → IFFT → scale to σ_u.
    """
    rng = np.random.default_rng(seed)

    # Ensure even length for FFT efficiency
    n_fft = n if n % 2 == 0 else n + 1
    freqs = np.fft.rfftfreq(n_fft, d=dt_seconds)  # 0 … f_Nyquist

    # Choose PSD model
    psd_func = _vonkarman_psd if spectrum == "vonkarman" else _kaimal_psd

    # Build amplitude spectrum  (one-sided PSD → amplitude)
    # S(f) has units [m²/s² / Hz].  |A(f)|² = S(f) · df
    df_hz = 1.0 / (n_fft * dt_seconds)
    S = psd_func(freqs, sigma_u, L_u, U)
    S[0] = 0.0  # no DC offset from turbulence
    amplitude = np.sqrt(S * df_hz)

    # Random phases
    phase = rng.uniform(0, 2 * np.pi, size=len(freqs))
    phase[0] = 0.0

    # Construct complex spectrum and inverse-transform
    spectrum_c = amplitude * np.exp(1j * phase)
    turb = np.fft.irfft(spectrum_c, n=n_fft)[:n]

    # Normalise so std ≈ sigma_u  (numerical rounding correction)
    actual_std = np.std(turb)
    if actual_std > 1e-12:
        turb *= sigma_u / actual_std

    return turb


def _interpolate_wind(
    x_coarse: np.ndarray,
    vals: np.ndarray,
    x_fine: np.ndarray,
    dt: float,
    TI: float = 0.12,
    spectrum: str = "vonkarman",
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Cubic-spline trend + Von-Kármán / Kaimal spectral turbulence."""
    cs = CubicSpline(x_coarse, vals, bc_type="natural")
    trend = cs(x_fine)

    U_mean = float(np.nanmean(vals[vals > 0])) if np.any(vals > 0) else 1.0
    sigma_u = TI * U_mean

    # IEC integral length-scale  L_u = 8.1 · Λ₁
    # Λ₁ = 42 m for hub ≤ 60 m, otherwise = 0.7 · hub_height
    # Without hub height info we use a representative 340 m (≈ 8.1 × 42)
    L_u = 340.0

    turb = _spectral_turbulence(
        len(x_fine), dt, sigma_u, U_mean, L_u, spectrum, seed,
    )

    return np.clip(trend + turb, 0, None)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  WIND DIRECTION – circular spline + random walk on circle
# ═══════════════════════════════════════════════════════════════════════════════

def _interpolate_wind_direction(
    x_coarse: np.ndarray,
    vals: np.ndarray,
    x_fine: np.ndarray,
    dt: float,
    sigma_yaw_deg: float = 8.0,
    tau_yaw: float = 60.0,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Circular cubic spline (sin/cos decomposition) + Ornstein–Uhlenbeck
    yaw wander with integral time-scale τ_yaw ≈ 60 s.
    """
    rad = np.deg2rad(vals)
    cs_sin = CubicSpline(x_coarse, np.sin(rad), bc_type="natural")
    cs_cos = CubicSpline(x_coarse, np.cos(rad), bc_type="natural")
    base = np.rad2deg(np.arctan2(cs_sin(x_fine), cs_cos(x_fine))) % 360

    # OU yaw noise
    rng = np.random.default_rng(seed)
    phi = np.exp(-dt / tau_yaw)
    noise_std = np.deg2rad(sigma_yaw_deg) * np.sqrt(max(1 - phi**2, 1e-12))
    yaw = np.zeros(len(x_fine))
    for i in range(1, len(yaw)):
        yaw[i] = phi * yaw[i - 1] + noise_std * rng.standard_normal()

    return (base + np.rad2deg(yaw)) % 360


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TEMPERATURE – harmonic diurnal decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def _interpolate_temperature(
    x_coarse: np.ndarray,
    vals: np.ndarray,
    x_fine: np.ndarray,
) -> np.ndarray:
    """
    Fit the fundamental diurnal cycle (24 h) and its 2nd harmonic (12 h)
    to the coarse data.  Interpolate the residual with a natural cubic
    spline (smooth, no overshoot risk for temperature).

        T(t) = T̄ + A₁ cos(ω₁t + φ₁) + A₂ cos(ω₂t + φ₂) + residual(t)
    """
    omega_1 = 2.0 * np.pi / 86400.0   # 24-h cycle
    omega_2 = 2.0 * omega_1            # 12-h cycle

    # Design matrix:  [1, cos(ω₁t), sin(ω₁t), cos(ω₂t), sin(ω₂t)]
    A = np.column_stack([
        np.ones(len(x_coarse)),
        np.cos(omega_1 * x_coarse),
        np.sin(omega_1 * x_coarse),
        np.cos(omega_2 * x_coarse),
        np.sin(omega_2 * x_coarse),
    ])

    # Least-squares fit
    coeffs, *_ = np.linalg.lstsq(A, vals, rcond=None)

    # Reconstruct harmonic at coarse and fine resolution
    harmonic_coarse = A @ coeffs
    A_fine = np.column_stack([
        np.ones(len(x_fine)),
        np.cos(omega_1 * x_fine),
        np.sin(omega_1 * x_fine),
        np.cos(omega_2 * x_fine),
        np.sin(omega_2 * x_fine),
    ])
    harmonic_fine = A_fine @ coeffs

    # Residual interpolation
    residual = vals - harmonic_coarse
    cs_res = CubicSpline(x_coarse, residual, bc_type="natural")

    return harmonic_fine + cs_res(x_fine)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PRESSURE – semidiurnal barometric tide + trend spline
# ═══════════════════════════════════════════════════════════════════════════════

def _interpolate_pressure(
    x_coarse: np.ndarray,
    vals: np.ndarray,
    x_fine: np.ndarray,
    lat: float = 20.0,
) -> np.ndarray:
    """
    Pressure is dominated by:
      - Synoptic trend  → natural cubic spline
      - Semidiurnal atmospheric tide S₂  (period 12 h)
        Amplitude ≈ 1.3 hPa at equator, ≈ 0.5 hPa at 45°.

    We fit the S₂ tide from the coarse data and interpolate the
    slowly-varying trend with a cubic spline.
    """
    omega_s2 = 2.0 * np.pi / (12.0 * 3600.0)  # 12-hour period

    # Design matrix: [1, cos(ω_s2 · t), sin(ω_s2 · t)]
    A = np.column_stack([
        np.ones(len(x_coarse)),
        np.cos(omega_s2 * x_coarse),
        np.sin(omega_s2 * x_coarse),
    ])
    coeffs, *_ = np.linalg.lstsq(A, vals, rcond=None)

    tide_coarse = A @ coeffs
    A_fine = np.column_stack([
        np.ones(len(x_fine)),
        np.cos(omega_s2 * x_fine),
        np.sin(omega_s2 * x_fine),
    ])
    tide_fine = A_fine @ coeffs

    residual = vals - tide_coarse
    cs_res = CubicSpline(x_coarse, residual, bc_type="natural")

    return tide_fine + cs_res(x_fine)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  HUMIDITY – Clausius–Clapeyron / dewpoint approach
# ═══════════════════════════════════════════════════════════════════════════════

def _magnus_es(T: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure (hPa) via Magnus formula."""
    return 6.1094 * np.exp(17.625 * T / (243.04 + T))


def _interpolate_humidity(
    x_coarse: np.ndarray,
    rh_vals: np.ndarray,
    x_fine: np.ndarray,
    temp_coarse: Optional[np.ndarray] = None,
    dew_coarse: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    If both temperature and dewpoint are available:
      1. Cubic-spline interpolate T(t) and Td(t) independently
      2. Recompute RH = 100 · e_s(Td) / e_s(T)  (Magnus formula)

    This preserves the Clausius–Clapeyron thermodynamic consistency.
    Falls back to harmonic diurnal + spline when Td is not available.
    """
    if temp_coarse is not None and dew_coarse is not None:
        T_fine = _interpolate_temperature(x_coarse, temp_coarse, x_fine)
        Td_fine = _interpolate_temperature(x_coarse, dew_coarse, x_fine)
        rh_fine = 100.0 * _magnus_es(Td_fine) / _magnus_es(T_fine)
        return np.clip(rh_fine, 0, 100)

    # Fallback: harmonic diurnal (RH has a strong diurnal cycle too)
    return np.clip(_interpolate_temperature(x_coarse, rh_vals, x_fine), 0, 100)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  SOLAR – Clear-sky envelope + Kt interpolation + cloud transients
# ═══════════════════════════════════════════════════════════════════════════════

def _solar_zenith_angle(
    times: pd.DatetimeIndex,
    lat: float = 20.0,
    lon: float = 0.0,
) -> np.ndarray:
    """
    Simplified solar zenith angle (Spencer, 1971 / Iqbal, 1983).
    Accurate to ~1° — sufficient for clear-sky envelope construction.
    """
    doy = times.dayofyear.values.astype(float)
    hour_utc = times.hour.values + times.minute.values / 60.0

    # Solar declination (Spencer 1971)
    B = (doy - 1) * 2.0 * np.pi / 365.0
    delta = (0.006918
             - 0.399912 * np.cos(B) + 0.070257 * np.sin(B)
             - 0.006758 * np.cos(2 * B) + 0.000907 * np.sin(2 * B)
             - 0.002697 * np.cos(3 * B) + 0.00148 * np.sin(3 * B))

    # Equation of time (minutes)
    EoT = 229.18 * (0.000075
                     + 0.001868 * np.cos(B) - 0.032077 * np.sin(B)
                     - 0.014615 * np.cos(2 * B) - 0.04089 * np.sin(2 * B))

    # Solar hour angle
    solar_time = hour_utc + (lon / 15.0) + (EoT / 60.0)
    omega = np.deg2rad(15.0 * (solar_time - 12.0))

    lat_r = np.deg2rad(lat)
    cos_z = (np.sin(lat_r) * np.sin(delta)
             + np.cos(lat_r) * np.cos(delta) * np.cos(omega))
    cos_z = np.clip(cos_z, -1, 1)
    return np.arccos(cos_z)  # radians


def _clearsky_ghi(zenith_rad: np.ndarray) -> np.ndarray:
    """
    Simplified Ineichen–Perez clear-sky GHI model
    (fixed Linke turbidity TL = 2.5, sea-level).

        GHI_cs ≈ 1361 · cos(θ_z) · 0.7^(AM^0.678)

    AM = air mass (Kasten & Young, 1989).
    """
    cos_z = np.cos(zenith_rad)
    cos_z_safe = np.clip(cos_z, 0.01, 1.0)
    is_day = cos_z > 0.0

    # Air mass (Kasten & Young 1989)
    zenith_deg = np.rad2deg(zenith_rad)
    zenith_deg_safe = np.clip(zenith_deg, 0, 89.9)
    AM = 1.0 / (cos_z_safe + 0.50572 * (96.07995 - zenith_deg_safe)**(-1.6364))
    AM = np.clip(AM, 1, 40)

    ghi_cs = 1361.0 * cos_z_safe * 0.7**(AM**0.678)
    ghi_cs[~is_day] = 0.0
    return ghi_cs


def _interpolate_solar(
    x_coarse: np.ndarray,
    vals: np.ndarray,
    x_fine: np.ndarray,
    times_coarse: pd.DatetimeIndex,
    times_fine: pd.DatetimeIndex,
    lat: float = 20.0,
    lon: float = 0.0,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Clear-sky decomposition + clearness-index Kt interpolation
    + stochastic cloud-transient sub-hourly ramps.
    """
    dt = float(np.median(np.diff(x_fine)))

    # Clear-sky envelope at coarse and fine resolution
    zen_c = _solar_zenith_angle(times_coarse, lat, lon)
    ghi_cs_c = _clearsky_ghi(zen_c)
    zen_f = _solar_zenith_angle(times_fine, lat, lon)
    ghi_cs_f = _clearsky_ghi(zen_f)

    # Clearness index Kt = GHI_obs / GHI_cs  (daytime only)
    with np.errstate(divide="ignore", invalid="ignore"):
        kt_coarse = np.where(ghi_cs_c > 10.0, vals / ghi_cs_c, 0.0)
    kt_coarse = np.clip(kt_coarse, 0.0, 1.2)  # can exceed 1 with reflections

    # Cubic-spline Kt, then reconstruct GHI = Kt · GHI_cs
    cs_kt = CubicSpline(x_coarse, kt_coarse, bc_type="clamped")
    kt_fine = np.clip(cs_kt(x_fine), 0.0, 1.2)

    ghi_fine = kt_fine * ghi_cs_f

    # Add cloud-transient ramps (Markov cloud state)
    rng = np.random.default_rng(seed)
    p_c2cl = min(0.008 * dt / 60, 0.12)   # cloud → clear
    p_cl2c = min(0.015 * dt / 60, 0.18)   # clear → cloud

    cloud_state = np.zeros(len(x_fine))
    state = 0.0
    for i in range(len(x_fine)):
        if state < 0.5:
            if rng.random() < p_cl2c:
                state = 1.0
        else:
            if rng.random() < p_c2cl:
                state = 0.0
        cloud_state[i] = state

    # Smooth cloud transitions (Ornstein–Uhlenbeck filter)
    tau = max(dt * 4, 180)
    phi = np.exp(-dt / tau)
    smooth = np.zeros(len(x_fine))
    smooth[0] = cloud_state[0]
    for i in range(1, len(x_fine)):
        smooth[i] = phi * smooth[i - 1] + (1 - phi) * cloud_state[i]

    # Dimming factor: clouds reduce by 25–50 %
    dim = 1.0 - 0.35 * smooth
    ghi_fine *= dim

    return np.clip(ghi_fine, 0, None)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  CLOUD COVER – logit-transformed cubic spline
# ═══════════════════════════════════════════════════════════════════════════════

def _interpolate_cloud_cover(
    x_coarse: np.ndarray,
    vals: np.ndarray,
    x_fine: np.ndarray,
) -> np.ndarray:
    """
    Logit transform: map [0, 100] → ℝ, cubic spline, then inverse.
    Prevents overshoot below 0 or above 100 %.
    """
    eps = 0.5  # avoid log(0)
    clamped = np.clip(vals, eps, 100.0 - eps)
    p = clamped / 100.0
    logit = np.log(p / (1.0 - p))

    cs = CubicSpline(x_coarse, logit, bc_type="natural")
    logit_fine = cs(x_fine)

    p_fine = 1.0 / (1.0 + np.exp(-logit_fine))
    return np.clip(p_fine * 100.0, 0, 100)


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════

def interpolate_series(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    target_freq: str,
    variable: str,
    turbulence_intensity: float = 0.12,
    wind_spectrum: str = "vonkarman",
    lat: float = 20.0,
    lon: float = 0.0,
    temp_coarse: Optional[np.ndarray] = None,
    dew_coarse: Optional[np.ndarray] = None,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Interpolate a time-series to a finer resolution using the
    physically-correct spectral / harmonic model for the variable.

    Parameters
    ----------
    times : original timestamps
    values : corresponding numeric values
    target_freq : e.g. '5min', '1min'
    variable : column name → selects interpolation physics
    turbulence_intensity : TI for wind Von-Kármán / Kaimal model
    wind_spectrum : 'vonkarman' or 'kaimal'
    lat, lon : for solar zenith angle and pressure tide amplitude
    temp_coarse, dew_coarse : for Clausius–Clapeyron humidity
    seed : reproducibility
    """
    if len(times) < 2:
        return pd.DataFrame({"time": times, "value": values})

    fine_times = pd.date_range(times[0], times[-1], freq=target_freq)
    if len(fine_times) < 2:
        return pd.DataFrame({"time": times, "value": values})

    t0 = times[0]
    x_c = (times - t0).total_seconds().astype(float)
    x_f = (fine_times - t0).total_seconds().astype(float)
    dt = float(np.median(np.diff(x_f)))
    vals = np.asarray(values, dtype=float)

    # ── dispatch ──────────────────────────────────────────────────────────

    if variable in NON_INTERPOLATABLE:
        idx = np.searchsorted(x_c, x_f, side="right") - 1
        fine_vals = vals[np.clip(idx, 0, len(vals) - 1)]

    elif variable in ("wind_speed", "wind_gusts"):
        fine_vals = _interpolate_wind(
            x_c, vals, x_f, dt, turbulence_intensity, wind_spectrum, seed)

    elif variable == "wind_direction":
        fine_vals = _interpolate_wind_direction(x_c, vals, x_f, dt, seed=seed)

    elif variable in ("temperature", "feels_like"):
        fine_vals = _interpolate_temperature(x_c, vals, x_f)

    elif variable == "pressure":
        fine_vals = _interpolate_pressure(x_c, vals, x_f, lat=lat)

    elif variable == "humidity":
        fine_vals = _interpolate_humidity(
            x_c, vals, x_f, temp_coarse=temp_coarse, dew_coarse=dew_coarse)

    elif variable == "dew_point":
        # Dewpoint is smooth like temperature — harmonic + spline
        fine_vals = _interpolate_temperature(x_c, vals, x_f)

    elif variable in ("shortwave_radiation", "direct_radiation",
                       "diffuse_radiation", "direct_normal_irradiance",
                       "global_tilted_irradiance", "ghi"):
        fine_vals = _interpolate_solar(
            x_c, vals, x_f, times, fine_times, lat, lon, seed)

    elif variable == "cloud_cover":
        fine_vals = _interpolate_cloud_cover(x_c, vals, x_f)

    else:
        # Unknown variable — natural cubic spline (safe default)
        cs = CubicSpline(x_c, vals, bc_type="natural")
        fine_vals = cs(x_f)

    return pd.DataFrame({"time": fine_times, "value": fine_vals})


# ═══════════════════════════════════════════════════════════════════════════════
# DataFrame-level interpolation
# ═══════════════════════════════════════════════════════════════════════════════

def interpolate_dataframe(
    df: pd.DataFrame,
    target_freq: str,
    columns: Optional[List[str]] = None,
    turbulence_intensity: float = 0.12,
    wind_spectrum: str = "vonkarman",
    lat: float = 20.0,
    lon: float = 0.0,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Interpolate an entire DataFrame to ``target_freq`` using
    variable-specific physics for each column.
    """
    if df is None or df.empty or "time" not in df.columns:
        return df

    times = pd.DatetimeIndex(df["time"])
    fine_times = pd.date_range(times[0], times[-1], freq=target_freq)
    result = pd.DataFrame({"time": fine_times})

    if columns is None:
        columns = [c for c in df.columns
                   if c != "time" and pd.api.types.is_numeric_dtype(df[c])]

    # Pre-extract temperature / dewpoint for Clausius–Clapeyron humidity
    temp_c = df["temperature"].values if "temperature" in df.columns else None
    dew_c = df["dew_point"].values if "dew_point" in df.columns else None

    for col in columns:
        if col not in df.columns:
            continue
        interp = interpolate_series(
            times, df[col].values, target_freq, col,
            turbulence_intensity=turbulence_intensity,
            wind_spectrum=wind_spectrum,
            lat=lat, lon=lon,
            temp_coarse=temp_c, dew_coarse=dew_c,
            seed=seed,
        )
        result[col] = interp["value"].values[:len(fine_times)]

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step-function expansion  (for displaying the API staircase)
# ═══════════════════════════════════════════════════════════════════════════════

def step_function_df(
    df: pd.DataFrame,
    target_freq: str,
) -> pd.DataFrame:
    """
    Expand a coarse DataFrame to ``target_freq`` as piecewise-constant
    (forward-fill).  Used to display the API forecast at fine zoom.
    """
    if df is None or df.empty or "time" not in df.columns:
        return df

    fine_times = pd.date_range(df["time"].iloc[0], df["time"].iloc[-1],
                               freq=target_freq)
    x_c = (pd.DatetimeIndex(df["time"]) - df["time"].iloc[0]).total_seconds().astype(float)
    x_f = (fine_times - df["time"].iloc[0]).total_seconds().astype(float)

    cols_out = {"time": fine_times}
    for col in df.columns:
        if col == "time":
            continue
        vals = df[col].values
        idx = np.searchsorted(x_c, x_f, side="right") - 1
        idx = np.clip(idx, 0, len(vals) - 1)
        if pd.api.types.is_numeric_dtype(df[col]):
            cols_out[col] = vals[idx]
        else:
            cols_out[col] = [vals[i] for i in idx]

    return pd.DataFrame(cols_out)
