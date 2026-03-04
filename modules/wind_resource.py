# =============================================================================
# Wind Resource Assessment Module  (YouWind-style)
# Bastankhah-Porté-Agel Wake Model  +  Jensen comparison
# Based on: SUMAILI NDEBA Bienvenu, PhD Thesis 2026
# =============================================================================

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Physical constants & Morocco defaults
# ---------------------------------------------------------------------------
RHO_0 = 1.225            # ISA sea-level density  [kg/m³]
GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Turbine catalogue (low-wind  + utility-scale)
# ---------------------------------------------------------------------------
@dataclass
class Turbine:
    name: str
    rated_kw: float
    rotor_d: float         # m
    hub_height: float      # m
    cut_in: float          # m/s
    rated_ws: float        # m/s
    cut_out: float         # m/s
    ct: float              # thrust coefficient at rated
    spacing_d: float = 7.0 # default inter-turbine spacing in D

    @property
    def swept_area(self) -> float:
        return math.pi * (self.rotor_d / 2) ** 2

    @property
    def specific_power(self) -> float:
        """W/m² of swept area."""
        return self.rated_kw * 1000 / self.swept_area


TURBINE_CATALOG: Dict[str, Turbine] = {
    # --- Low-wind (thesis Benguerir pilot) ---
    "LW-3kW D5.5":    Turbine("LW-3kW D5.5",     3,   5.5,  24, 2.0,  7.0,  25, 0.84, 6),
    "LW-5kW D7.0":    Turbine("LW-5kW D7.0",     5,   7.0,  24, 2.0,  7.0,  25, 0.82, 6),
    "LW-10kW D10":    Turbine("LW-10kW D10",    10,  10.0,  24, 2.0,  7.0,  25, 0.82, 8),
    "LW-10kW D12":    Turbine("LW-10kW D12",    10,  12.0,  30, 2.0,  7.0,  25, 0.80, 8),
    "LW-15kW D12":    Turbine("LW-15kW D12",    15,  12.0,  30, 2.0,  7.5,  30, 0.80, 8),
    "LW-20kW D16":    Turbine("LW-20kW D16",    20,  16.0,  36, 2.0,  7.5,  30, 0.78, 7),
    # --- Standard small-wind ---
    "Aeolos-H 3kW":   Turbine("Aeolos-H 3kW",    3,   4.0,  18, 2.5,  9.0,  45, 0.82, 6),
    "Bergey Excel 10": Turbine("Bergey Excel 10", 10,   7.0,  30, 2.5, 11.0,  60, 0.78, 6),
    # --- Utility-scale Morocco (generic) ---
    "Gamesa G132":     Turbine("Gamesa G132",   3300, 132.0,  84, 3.0, 12.5,  25, 0.80, 7),
    "Vestas V110":     Turbine("Vestas V110",   2000, 110.0,  80, 3.0, 12.0,  25, 0.82, 7),
    "Vestas V150":     Turbine("Vestas V150",   4200, 150.0, 105, 3.0, 11.5,  25, 0.79, 7),
    "Enercon E126":    Turbine("Enercon E126",  7580, 127.0, 135, 3.0, 13.0,  25, 0.80, 8),
}


# ---------------------------------------------------------------------------
# Morocco site presets
# ---------------------------------------------------------------------------
MOROCCO_SITES: Dict[str, Dict[str, Any]] = {
    "Benguerir (GEP/UM6P)": {
        "lat": 32.224, "lon": -7.884, "elev": 446, "rho": 1.146,
        "weibull_k": 1.72, "weibull_c": 3.85, "alpha": 0.137,
        "climate": "semi_arid",
    },
    "Meskala (Essaouira)": {
        "lat": 30.85, "lon": -9.48, "elev": 530, "rho": 1.218,
        "weibull_k": 2.30, "weibull_c": 7.60, "alpha": 0.135,
        "climate": "semi_arid",
    },
    "Dakhla": {
        "lat": 23.685, "lon": -15.958, "elev": 7, "rho": 1.220,
        "weibull_k": 2.80, "weibull_c": 8.50, "alpha": 0.110,
        "climate": "desert",
    },
    "Tarfaya": {
        "lat": 27.936, "lon": -12.926, "elev": 20, "rho": 1.222,
        "weibull_k": 2.60, "weibull_c": 9.20, "alpha": 0.100,
        "climate": "desert",
    },
    "Tanger-Med": {
        "lat": 35.780, "lon": -5.800, "elev": 30, "rho": 1.220,
        "weibull_k": 2.20, "weibull_c": 6.90, "alpha": 0.130,
        "climate": "coastal",
    },
}


# ---------------------------------------------------------------------------
# 1.  Weibull distribution fitting  (Method of Moments)
# ---------------------------------------------------------------------------
def weibull_fit(ws: np.ndarray) -> Tuple[float, float]:
    """
    Fit Weibull k and c parameters using Method of Moments.
    Returns (k, c).
    """
    ws = ws[ws > 0]
    if len(ws) < 10:
        return 2.0, float(ws.mean()) if len(ws) > 0 else 5.0

    mu = ws.mean()
    sigma = ws.std(ddof=0)
    if mu < 1e-6:
        return 2.0, 1.0

    cv = sigma / mu
    ks = np.linspace(0.5, 10.0, 3000)
    from scipy.special import gamma as _gamma_fn
    try:
        vals = np.array([
            math.sqrt(_gamma_fn(1 + 2.0 / k) / _gamma_fn(1 + 1.0 / k) ** 2 - 1)
            for k in ks
        ])
    except Exception:
        # Fallback without scipy
        vals = np.array([_approx_gamma_ratio(k) for k in ks])

    idx = np.argmin(np.abs(vals - cv))
    k = float(ks[idx])
    try:
        c = float(mu / _gamma_fn(1 + 1.0 / k))
    except Exception:
        c = float(mu / math.gamma(1 + 1.0 / k))
    return k, c


def _approx_gamma_ratio(k: float) -> float:
    """Approximate CV from Weibull k (no scipy fallback)."""
    g1 = math.gamma(1 + 1.0 / k)
    g2 = math.gamma(1 + 2.0 / k)
    return math.sqrt(g2 / g1 ** 2 - 1)


def weibull_pdf(ws: np.ndarray, k: float, c: float) -> np.ndarray:
    """Weibull probability density function."""
    ws = np.asarray(ws, dtype=float)
    pdf = np.zeros_like(ws)
    mask = ws > 0
    pdf[mask] = (k / c) * (ws[mask] / c) ** (k - 1) * np.exp(-(ws[mask] / c) ** k)
    return pdf


def weibull_cdf(ws: np.ndarray, k: float, c: float) -> np.ndarray:
    """Weibull cumulative distribution function."""
    ws = np.asarray(ws, dtype=float)
    return 1.0 - np.exp(-(np.maximum(ws, 0) / c) ** k)


# ---------------------------------------------------------------------------
# 2.  Height extrapolation  (power law)
# ---------------------------------------------------------------------------
def extrapolate_wind_speed(
    u_ref: float,
    z_ref: float,
    z_target: float,
    alpha: float = 0.14,
) -> float:
    """Power-law wind-speed extrapolation to target height."""
    if z_ref <= 0 or z_target <= 0:
        return u_ref
    return u_ref * (z_target / z_ref) ** alpha


def compute_shear_exponent(u1: float, z1: float, u2: float, z2: float) -> float:
    """Estimate shear exponent α from two measurement heights."""
    if u1 <= 0 or u2 <= 0 or z1 <= 0 or z2 <= 0 or z1 == z2:
        return 0.14
    return math.log(u2 / u1) / math.log(z2 / z1)


# ---------------------------------------------------------------------------
# 3.  Power curve  (cubic ramp)
# ---------------------------------------------------------------------------
def power_curve(
    ws: np.ndarray,
    turbine: Turbine,
) -> np.ndarray:
    """
    Compute turbine power output [kW] for each wind speed value.
    Uses cubic ramp between cut-in and rated.
    """
    ws = np.asarray(ws, dtype=float)
    p = np.zeros_like(ws)
    ramp = (ws >= turbine.cut_in) & (ws < turbine.rated_ws)
    rated = (ws >= turbine.rated_ws) & (ws <= turbine.cut_out)
    p[ramp] = turbine.rated_kw * (
        (ws[ramp] - turbine.cut_in) / (turbine.rated_ws - turbine.cut_in)
    ) ** 3
    p[rated] = turbine.rated_kw
    return p


def capacity_factor(ws: np.ndarray, turbine: Turbine) -> float:
    """Mean capacity factor from a wind-speed time series."""
    p = power_curve(ws, turbine)
    return float(p.mean() / turbine.rated_kw) if turbine.rated_kw > 0 else 0.0


# ---------------------------------------------------------------------------
# 4.  Wind power density
# ---------------------------------------------------------------------------
def wind_power_density(ws: np.ndarray, rho: float = RHO_0) -> float:
    """Mean wind power density [W/m²] = 0.5 ρ <v³>."""
    return float(0.5 * rho * np.mean(ws ** 3))


# ---------------------------------------------------------------------------
# 5.  Bastankhah-Porté-Agel wake model  (thesis implementation)
# ---------------------------------------------------------------------------
def bastankhah_wake_deficit(
    x_D: float,
    y_D: float,
    ct: float,
    k_w: float = 0.075,
    epsilon: Optional[float] = None,
) -> float:
    """
    Bastankhah & Porté-Agel (2014) Gaussian wake model.

    Args:
        x_D:  downstream distance normalised by rotor diameter  (x/D)
        y_D:  lateral offset normalised by rotor diameter  (y/D)
        ct:   thrust coefficient
        k_w:  wake expansion rate  (onshore ≈0.075, offshore ≈0.04)
        epsilon: initial wake width; if None, computed from Ct

    Returns:
        Fractional velocity deficit  ΔU/U∞  (0 = no deficit)
    """
    if x_D <= 0:
        return 0.0

    sqrt_1mCt = math.sqrt(max(1e-12, 1 - ct))
    if epsilon is None:
        beta = 0.5 * (1 + sqrt_1mCt) / max(1e-12, sqrt_1mCt)
        epsilon = 0.25 * math.sqrt(beta)

    sigma_y = k_w * x_D + epsilon

    arg = ct / (8.0 * sigma_y ** 2)
    if arg >= 1.0:
        delta_c = 1.0 - sqrt_1mCt
    else:
        delta_c = 1.0 - math.sqrt(1.0 - arg)

    deficit = delta_c * math.exp(-0.5 * (y_D / sigma_y) ** 2)
    return max(0.0, deficit)


def bastankhah_wake_field(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    ct: float,
    k_w: float = 0.075,
    nx: int = 200,
    ny: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2-D normalised wake velocity field  U/U∞  for a single turbine.

    Returns:
        (X, Y, U_ratio)  where X, Y in rotor-diameters, U_ratio ∈ [0, 1].
    """
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(xs, ys)
    U = np.ones_like(X)

    sqrt_1mCt = math.sqrt(max(1e-12, 1 - ct))
    beta = 0.5 * (1 + sqrt_1mCt) / max(1e-12, sqrt_1mCt)
    epsilon = 0.25 * math.sqrt(beta)

    for i in range(ny):
        for j in range(nx):
            xd, yd = X[i, j], Y[i, j]
            if xd > 0:
                deficit = bastankhah_wake_deficit(xd, yd, ct, k_w, epsilon)
                U[i, j] = max(0.0, 1.0 - deficit)
    return X, Y, U


# ---------------------------------------------------------------------------
# 6.  Jensen (Park) wake model  (for comparison)
# ---------------------------------------------------------------------------
def jensen_wake_deficit(
    x: float,
    R: float,
    ct: float,
    k_wake: float = 0.075,
) -> float:
    """
    Jensen/Park top-hat wake model.

    Args:
        x:  downstream distance  [m]
        R:  rotor radius  [m]
        ct: thrust coefficient
        k_wake: wake decay constant

    Returns:
        Velocity ratio  U_wake / U∞
    """
    if x <= 0:
        return 1.0
    denom = (1 + k_wake * x / R) ** 2
    return 1.0 - (1 - math.sqrt(max(0, 1 - ct))) / denom


# ---------------------------------------------------------------------------
# 7.  Wind farm wake superposition  (Bastankhah + RSS)
# ---------------------------------------------------------------------------
def compute_farm_wake(
    layout_xy: np.ndarray,
    wind_speed: float,
    wind_dir_deg: float,
    turbine: Turbine,
    k_w: float = 0.075,
) -> np.ndarray:
    """
    Compute effective wind speed at each turbine accounting for wakes.
    Uses root-sum-of-squares (RSS) superposition of Bastankhah deficits.

    Args:
        layout_xy: (N, 2) array of turbine (x, y) positions [m]
        wind_speed: free-stream wind speed [m/s]
        wind_dir_deg: wind direction [degrees, meteorological: 0=N, 90=E]
        turbine: Turbine dataclass
        k_w: wake expansion rate

    Returns:
        Array of effective wind speeds at each turbine [m/s]
    """
    N = len(layout_xy)
    D = turbine.rotor_d

    # Wind direction angle  (convert met→math)
    theta = math.radians(270 - wind_dir_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # Rotate layout to align x-axis with wind direction
    rotated = np.empty_like(layout_xy)
    for i in range(N):
        dx = layout_xy[i, 0]
        dy = layout_xy[i, 1]
        rotated[i, 0] = dx * cos_t + dy * sin_t
        rotated[i, 1] = -dx * sin_t + dy * cos_t

    u_eff = np.full(N, wind_speed)

    # Sort by rotated-x (upwind → downwind)
    order = np.argsort(rotated[:, 0])

    for idx_j in range(N):
        j = order[idx_j]
        deficit_sq_sum = 0.0
        for idx_i in range(idx_j):
            i = order[idx_i]
            dx = (rotated[j, 0] - rotated[i, 0]) / D  # x/D
            dy = (rotated[j, 1] - rotated[i, 1]) / D  # y/D
            if dx > 0:
                d = bastankhah_wake_deficit(dx, dy, turbine.ct, k_w)
                # Scale deficit by upwind turbine's local speed
                deficit_sq_sum += (d * u_eff[i] / wind_speed) ** 2
        u_eff[j] = wind_speed * max(0.0, 1.0 - math.sqrt(deficit_sq_sum))

    return u_eff


def farm_aep(
    layout_xy: np.ndarray,
    ws_hourly: np.ndarray,
    wd_hourly: np.ndarray,
    turbine: Turbine,
    rho: float = RHO_0,
    k_w: float = 0.075,
    n_sectors: int = 16,
) -> Dict[str, Any]:
    """
    Compute annual energy production (AEP) for a wind farm layout.
    Bins hourly data into directional sectors for efficiency.

    Returns dict with: aep_mwh, aep_no_wake_mwh, wake_loss_pct,
    capacity_factor, turbine_aeps, mean_farm_ws.
    """
    N = len(layout_xy)
    sector_width = 360.0 / n_sectors

    # Bin wind data by sector
    sector_edges = np.linspace(0, 360, n_sectors + 1)

    total_energy_kwh = np.zeros(N)
    total_energy_no_wake = 0.0
    hours_total = len(ws_hourly)

    for s in range(n_sectors):
        lo = sector_edges[s]
        hi = sector_edges[s + 1]
        if s == n_sectors - 1:
            mask = (wd_hourly >= lo) | (wd_hourly < sector_edges[1])
        else:
            mask = (wd_hourly >= lo) & (wd_hourly < hi)

        sector_ws = ws_hourly[mask]
        if len(sector_ws) == 0:
            continue

        mean_dir = (lo + hi) / 2.0
        mean_ws = float(sector_ws.mean())

        # Compute wake-adjusted speeds
        u_eff = compute_farm_wake(layout_xy, mean_ws, mean_dir, turbine, k_w)

        # Energy per turbine for this sector
        for i in range(N):
            p = power_curve(np.array([u_eff[i]]), turbine)[0]
            total_energy_kwh[i] += p * len(sector_ws)  # kW × hours

        # No-wake reference
        p_free = power_curve(np.array([mean_ws]), turbine)[0]
        total_energy_no_wake += p_free * N * len(sector_ws)

    aep_mwh = float(total_energy_kwh.sum() / 1000)
    aep_no_wake = float(total_energy_no_wake / 1000)
    wake_loss = (1 - aep_mwh / aep_no_wake) * 100 if aep_no_wake > 0 else 0

    rated_farm_kwh = turbine.rated_kw * N * hours_total
    cf = (total_energy_kwh.sum() / rated_farm_kwh) if rated_farm_kwh > 0 else 0

    return {
        "aep_mwh": aep_mwh,
        "aep_no_wake_mwh": aep_no_wake,
        "wake_loss_pct": wake_loss,
        "capacity_factor": float(cf),
        "turbine_aeps_mwh": (total_energy_kwh / 1000).tolist(),
        "mean_farm_ws_ms": float(ws_hourly.mean()),
        "n_turbines": N,
    }


# ---------------------------------------------------------------------------
# 8.  Layout generators
# ---------------------------------------------------------------------------
def generate_grid_layout(
    n_rows: int,
    n_cols: int,
    spacing_x_D: float,
    spacing_y_D: float,
    rotor_d: float,
) -> np.ndarray:
    """
    Generate a regular grid wind farm layout.
    Returns (N, 2) array of (x, y) positions [m].
    """
    positions = []
    for r in range(n_rows):
        for c in range(n_cols):
            x = c * spacing_x_D * rotor_d
            y = r * spacing_y_D * rotor_d
            positions.append([x, y])
    return np.array(positions)


def generate_staggered_layout(
    n_rows: int,
    n_cols: int,
    spacing_x_D: float,
    spacing_y_D: float,
    rotor_d: float,
) -> np.ndarray:
    """
    Generate a staggered (offset rows) wind farm layout.
    Even rows are offset by half the x-spacing.
    """
    positions = []
    for r in range(n_rows):
        x_offset = 0.5 * spacing_x_D * rotor_d if r % 2 else 0
        for c in range(n_cols):
            x = c * spacing_x_D * rotor_d + x_offset
            y = r * spacing_y_D * rotor_d
            positions.append([x, y])
    return np.array(positions)


# ---------------------------------------------------------------------------
# 9.  Economics  (LCOE / NPV / IRR)
# ---------------------------------------------------------------------------
def compute_lcoe(
    capex_per_kw: float,
    cf: float,
    discount_rate: float = 0.08,
    lifetime_yr: int = 20,
    opex_frac: float = 0.025,
    rho_site: float = RHO_0,
) -> float:
    """
    Levelised Cost of Energy  [$/kWh].
    LCOE = CAPEX_per_kW × KF / CF
    where KF = (CRF + OPEX_frac) × ρ₀ / (ρ_site × 8760)
    """
    if cf <= 0:
        return float("inf")
    crf = (discount_rate * (1 + discount_rate) ** lifetime_yr) / (
        (1 + discount_rate) ** lifetime_yr - 1
    )
    kf = (crf + opex_frac) * RHO_0 / (rho_site * 8760)
    return capex_per_kw * kf / cf


def compute_npv(
    capex_total: float,
    annual_revenue: float,
    annual_opex: float,
    discount_rate: float = 0.08,
    lifetime_yr: int = 20,
) -> float:
    """Net Present Value of a wind project [$]."""
    npv = -capex_total
    for t in range(1, lifetime_yr + 1):
        npv += (annual_revenue - annual_opex) / (1 + discount_rate) ** t
    return npv


def compute_simple_payback(
    capex_total: float,
    annual_revenue: float,
    annual_opex: float,
) -> float:
    """Simple payback period [years]."""
    net = annual_revenue - annual_opex
    if net <= 0:
        return float("inf")
    return capex_total / net


# ---------------------------------------------------------------------------
# 10. CAPEX estimation  (power-law scaling from thesis)
# ---------------------------------------------------------------------------
def estimate_capex_per_kw(
    rated_kw: float,
    turbine_type: str = "HAWT",
) -> float:
    """Scale-dependent CAPEX $/kW using thesis power-law model."""
    if turbine_type.upper() == "VAWT":
        c0, exp = 11000.0, 0.18
    else:
        c0, exp = 7500.0, 0.22
    return c0 * max(rated_kw, 1.0) ** (-exp)


# ---------------------------------------------------------------------------
# 11. IEC wind class assessment
# ---------------------------------------------------------------------------
def iec_wind_class(v_avg: float, ti_15: float = None) -> Dict[str, str]:
    """
    Classify site per IEC 61400-1:2019.
    v_avg: annual mean wind speed at hub height [m/s]
    ti_15: turbulence intensity at 15 m/s
    """
    # Wind class by Vref
    if v_avg <= 0:
        wind_class = "Insufficient data"
    elif v_avg * 5 <= 25:
        wind_class = "Class IV (V_ref ≤ 30 m/s)"
    elif v_avg * 5 <= 37.5:
        wind_class = "Class III (V_ref = 37.5 m/s)"
    elif v_avg * 5 <= 42.5:
        wind_class = "Class II (V_ref = 42.5 m/s)"
    else:
        wind_class = "Class I (V_ref = 50 m/s)"

    # Turbulence class
    turb_class = "N/A"
    if ti_15 is not None:
        if ti_15 <= 0.12:
            turb_class = "Class C (I_ref = 0.12)"
        elif ti_15 <= 0.14:
            turb_class = "Class B (I_ref = 0.14)"
        elif ti_15 <= 0.16:
            turb_class = "Class A (I_ref = 0.16)"
        else:
            turb_class = "Class A+ (I_ref > 0.16)"

    return {"wind_class": wind_class, "turbulence_class": turb_class}


# ---------------------------------------------------------------------------
# 12. Wind resource summary from hourly data
# ---------------------------------------------------------------------------
def assess_wind_resource(
    ws_hourly: np.ndarray,
    wd_hourly: np.ndarray,
    measurement_height: float = 10.0,
    hub_height: float = 80.0,
    alpha: float = 0.14,
    rho: float = RHO_0,
) -> Dict[str, Any]:
    """
    Comprehensive wind resource assessment from hourly data.
    Returns Weibull params, extrapolated speeds, power density, wind class, etc.
    """
    ws = np.asarray(ws_hourly, dtype=float)
    wd = np.asarray(wd_hourly, dtype=float)
    ws_valid = ws[ws > 0]

    # Weibull fit
    k, c = weibull_fit(ws_valid)

    # Extrapolate to hub height
    ws_hub = np.array([
        extrapolate_wind_speed(u, measurement_height, hub_height, alpha)
        for u in ws
    ])
    ws_hub_valid = ws_hub[ws_hub > 0]

    k_hub, c_hub = weibull_fit(ws_hub_valid) if len(ws_hub_valid) > 10 else (k, c)

    # Power density
    pd_meas = wind_power_density(ws_valid, rho)
    pd_hub = wind_power_density(ws_hub_valid, rho)

    # Wind rose  (16 sectors)
    n_sectors = 16
    sector_width = 360.0 / n_sectors
    wind_rose = []
    for s in range(n_sectors):
        lo = s * sector_width
        hi = lo + sector_width
        if s == 0:
            mask = (wd >= 360 - sector_width / 2) | (wd < sector_width / 2)
        else:
            mask = (wd >= lo - sector_width / 2) & (wd < hi - sector_width / 2)
        freq = float(mask.sum()) / len(wd) if len(wd) > 0 else 0
        mean_ws = float(ws[mask].mean()) if mask.sum() > 0 else 0
        wind_rose.append({
            "direction": lo,
            "direction_label": _dir_label(lo),
            "frequency": freq,
            "mean_ws": mean_ws,
        })

    # IEC classification (approximate)
    iec = iec_wind_class(float(ws_hub_valid.mean()) if len(ws_hub_valid) > 0 else 0)

    return {
        "measurement_height_m": measurement_height,
        "hub_height_m": hub_height,
        "shear_exponent": alpha,
        "mean_ws_measured": float(ws_valid.mean()) if len(ws_valid) > 0 else 0,
        "mean_ws_hub": float(ws_hub_valid.mean()) if len(ws_hub_valid) > 0 else 0,
        "max_ws_measured": float(ws_valid.max()) if len(ws_valid) > 0 else 0,
        "weibull_k_measured": k,
        "weibull_c_measured": c,
        "weibull_k_hub": k_hub,
        "weibull_c_hub": c_hub,
        "power_density_measured_wm2": pd_meas,
        "power_density_hub_wm2": pd_hub,
        "iec_class": iec,
        "wind_rose": wind_rose,
        "density_kg_m3": rho,
    }


def _dir_label(deg: float) -> str:
    labels = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return labels[round(deg / 22.5) % 16]


# ---------------------------------------------------------------------------
# 13. Wind farm optimisation (simple greedy swap)
# ---------------------------------------------------------------------------
def optimise_layout_greedy(
    initial_layout: np.ndarray,
    ws_hourly: np.ndarray,
    wd_hourly: np.ndarray,
    turbine: Turbine,
    k_w: float = 0.075,
    n_iterations: int = 200,
    step_m: float = 50.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Greedy local-search layout optimisation.
    Moves one turbine at a time in random directions; keeps improvements.

    Returns (optimised_layout, results_dict).
    """
    layout = initial_layout.copy()
    best_aep = farm_aep(layout, ws_hourly, wd_hourly, turbine, k_w=k_w)["aep_mwh"]

    rng = np.random.default_rng(42)
    n_turbines = len(layout)

    for iteration in range(n_iterations):
        idx = rng.integers(0, n_turbines)
        angle = rng.uniform(0, 2 * math.pi)
        dx = step_m * math.cos(angle)
        dy = step_m * math.sin(angle)

        trial = layout.copy()
        trial[idx, 0] += dx
        trial[idx, 1] += dy

        # Check minimum spacing
        ok = True
        for j in range(n_turbines):
            if j == idx:
                continue
            dist = math.sqrt((trial[idx, 0] - trial[j, 0]) ** 2 +
                             (trial[idx, 1] - trial[j, 1]) ** 2)
            if dist < 2.0 * turbine.rotor_d:
                ok = False
                break
        if not ok:
            continue

        trial_aep = farm_aep(trial, ws_hourly, wd_hourly, turbine, k_w=k_w)["aep_mwh"]
        if trial_aep > best_aep:
            layout = trial
            best_aep = trial_aep

    result = farm_aep(layout, ws_hourly, wd_hourly, turbine, k_w=k_w)
    return layout, result
