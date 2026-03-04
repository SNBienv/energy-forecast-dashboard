# =============================================================================
# Solar Resource Assessment Module
# PV system sizing, irradiance analysis, and energy yield estimation
# Morocco-focused with multi-site presets
# =============================================================================

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
SOLAR_CONSTANT = 1361.0   # W/m²  (AM0)
STC_IRRADIANCE = 1000.0   # W/m²  (Standard Test Conditions)
STC_TEMP = 25.0            # °C
NOCT = 45.0                # °C (Nominal Operating Cell Temperature)


# ---------------------------------------------------------------------------
# PV Module catalogue
# ---------------------------------------------------------------------------
@dataclass
class PVModule:
    name: str
    rated_wp: float         # Watt-peak per module
    efficiency: float       # decimal (0-1)
    temp_coeff: float       # %/°C (negative, e.g. -0.40)
    area_m2: float          # module area
    degradation_yr: float   # annual degradation rate (decimal)
    technology: str         # mono-Si, poly-Si, CdTe, CIGS, HJT, bifacial

    @property
    def specific_power(self) -> float:
        return self.rated_wp / self.area_m2


PV_CATALOG: Dict[str, PVModule] = {
    "Generic Mono-Si 400W":   PVModule("Generic Mono-Si 400W",   400, 0.205, -0.35, 1.95, 0.005, "mono-Si"),
    "Generic Mono-Si 550W":   PVModule("Generic Mono-Si 550W",   550, 0.213, -0.34, 2.58, 0.004, "mono-Si"),
    "LONGi Hi-MO 6 580W":    PVModule("LONGi Hi-MO 6 580W",    580, 0.224, -0.29, 2.59, 0.004, "mono-Si"),
    "JinkoSolar Tiger Neo 620W": PVModule("JinkoSolar Tiger Neo 620W", 620, 0.227, -0.30, 2.73, 0.004, "mono-Si"),
    "Bifacial 550W":          PVModule("Bifacial 550W",          550, 0.210, -0.32, 2.62, 0.004, "bifacial"),
    "First Solar CdTe 450W":  PVModule("First Solar CdTe 450W",450, 0.190, -0.28, 2.37, 0.003, "CdTe"),
    "Generic Poly-Si 340W":   PVModule("Generic Poly-Si 340W", 340, 0.175, -0.40, 1.94, 0.006, "poly-Si"),
    "HJT 700W":               PVModule("HJT 700W",              700, 0.230, -0.26, 3.04, 0.003, "HJT"),
}


# ---------------------------------------------------------------------------
# Morocco solar site presets (typical GHI from PV-GIS / SolarGIS)
# ---------------------------------------------------------------------------
MOROCCO_SOLAR_SITES: Dict[str, Dict[str, Any]] = {
    "Ouarzazate (Noor)": {
        "lat": 30.92, "lon": -6.90, "elev": 1160,
        "ghi_kwh_m2_yr": 2150, "dni_kwh_m2_yr": 2520,
        "avg_temp_c": 20.5, "climate": "desert",
    },
    "Benguerir (GEP)": {
        "lat": 32.224, "lon": -7.884, "elev": 446,
        "ghi_kwh_m2_yr": 1980, "dni_kwh_m2_yr": 2200,
        "avg_temp_c": 21.0, "climate": "semi_arid",
    },
    "Midelt": {
        "lat": 32.68, "lon": -4.73, "elev": 1520,
        "ghi_kwh_m2_yr": 2050, "dni_kwh_m2_yr": 2380,
        "avg_temp_c": 16.0, "climate": "semi_arid",
    },
    "Laâyoune": {
        "lat": 27.15, "lon": -13.20, "elev": 10,
        "ghi_kwh_m2_yr": 2200, "dni_kwh_m2_yr": 2450,
        "avg_temp_c": 22.0, "climate": "desert",
    },
    "Tata": {
        "lat": 29.75, "lon": -7.97, "elev": 700,
        "ghi_kwh_m2_yr": 2250, "dni_kwh_m2_yr": 2600,
        "avg_temp_c": 22.5, "climate": "desert",
    },
    "Errachidia": {
        "lat": 31.93, "lon": -4.43, "elev": 1060,
        "ghi_kwh_m2_yr": 2100, "dni_kwh_m2_yr": 2400,
        "avg_temp_c": 20.0, "climate": "desert",
    },
    "Casablanca": {
        "lat": 33.57, "lon": -7.59, "elev": 60,
        "ghi_kwh_m2_yr": 1820, "dni_kwh_m2_yr": 1950,
        "avg_temp_c": 18.5, "climate": "coastal",
    },
    "Dakhla": {
        "lat": 23.68, "lon": -15.96, "elev": 7,
        "ghi_kwh_m2_yr": 2180, "dni_kwh_m2_yr": 2350,
        "avg_temp_c": 21.5, "climate": "desert",
    },
}


# ---------------------------------------------------------------------------
# 1. Solar geometry
# ---------------------------------------------------------------------------
def solar_declination(day_of_year: int) -> float:
    """Solar declination angle [degrees]."""
    return 23.45 * math.sin(math.radians(360 / 365 * (284 + day_of_year)))


def hour_angle(solar_hour: float) -> float:
    """Hour angle [degrees]. solar_hour=12 → 0°."""
    return 15.0 * (solar_hour - 12.0)


def solar_elevation(lat: float, day_of_year: int, solar_hour: float) -> float:
    """Solar elevation angle [degrees]."""
    dec = math.radians(solar_declination(day_of_year))
    lat_r = math.radians(lat)
    ha = math.radians(hour_angle(solar_hour))
    sin_elev = (math.sin(lat_r) * math.sin(dec) +
                math.cos(lat_r) * math.cos(dec) * math.cos(ha))
    return math.degrees(math.asin(max(-1, min(1, sin_elev))))


def optimal_tilt_angle(lat: float) -> Dict[str, float]:
    """
    Estimate optimal fixed-tilt angle for PV panels.
    Returns annual optimal and seasonal recommendations.
    """
    abs_lat = abs(lat)
    annual = abs_lat * 0.76 + 3.1   # Jacobson approximation
    summer = max(0, abs_lat - 15)
    winter = min(90, abs_lat + 15)
    return {
        "annual_optimal": round(annual, 1),
        "summer_optimal": round(summer, 1),
        "winter_optimal": round(winter, 1),
    }


def day_length_hours(lat: float, day_of_year: int) -> float:
    """Theoretical day length [hours]."""
    dec = math.radians(solar_declination(day_of_year))
    lat_r = math.radians(lat)
    cos_ha = -math.tan(lat_r) * math.tan(dec)
    if cos_ha < -1:
        return 24.0
    if cos_ha > 1:
        return 0.0
    return 2.0 * math.degrees(math.acos(cos_ha)) / 15.0


# ---------------------------------------------------------------------------
# 2.  Irradiance on tilted surface  (isotropic sky model)
# ---------------------------------------------------------------------------
def irradiance_on_tilt(
    ghi: float,
    dni: float,
    dhi: float,
    solar_zenith_deg: float,
    tilt_deg: float,
    azimuth_deg: float = 180.0,   # south-facing
    albedo: float = 0.25,
) -> float:
    """
    Total irradiance on a tilted surface [W/m²].
    Uses isotropic diffuse model.
    """
    cos_z = math.cos(math.radians(solar_zenith_deg))
    cos_tilt = math.cos(math.radians(tilt_deg))

    # Beam on tilt (simplified — assumes facing equator)
    if cos_z > 0.01:
        beam_tilt = dni * max(0, math.cos(math.radians(abs(tilt_deg - solar_zenith_deg))))
    else:
        beam_tilt = 0

    # Diffuse (isotropic)
    diffuse_tilt = dhi * (1 + cos_tilt) / 2

    # Ground-reflected
    reflected = ghi * albedo * (1 - cos_tilt) / 2

    return beam_tilt + diffuse_tilt + reflected


# ---------------------------------------------------------------------------
# 3.  PV cell temperature
# ---------------------------------------------------------------------------
def cell_temperature(
    ambient_temp: float,
    irradiance_wm2: float,
    wind_speed: float = 1.0,
    noct: float = NOCT,
) -> float:
    """
    Estimate PV cell temperature [°C].
    NOCT model with wind correction.
    """
    t_cell = ambient_temp + (noct - 20) * (irradiance_wm2 / 800)
    # Wind cooling factor
    if wind_speed > 1.0:
        t_cell -= 0.5 * (wind_speed - 1.0)
    return t_cell


# ---------------------------------------------------------------------------
# 4.  PV power output
# ---------------------------------------------------------------------------
def pv_power_output(
    irradiance_wm2: float,
    module: PVModule,
    cell_temp_c: float = STC_TEMP,
    n_modules: int = 1,
    inverter_efficiency: float = 0.96,
    soiling_loss: float = 0.03,
    mismatch_loss: float = 0.02,
    wiring_loss: float = 0.02,
) -> float:
    """
    Compute AC power output [W] from PV system.

    Args:
        irradiance_wm2: POA irradiance
        module: PV module dataclass
        cell_temp_c: cell temperature [°C]
        n_modules: number of modules
        inverter_efficiency: DC-AC conversion efficiency
        soiling_loss: fraction (0-1)
        mismatch_loss: fraction
        wiring_loss: fraction

    Returns:
        AC power output [W]
    """
    # Temperature derating
    delta_t = cell_temp_c - STC_TEMP
    temp_factor = 1 + (module.temp_coeff / 100) * delta_t  # temp_coeff is %/°C

    # DC power per module
    p_dc = module.rated_wp * (irradiance_wm2 / STC_IRRADIANCE) * max(0, temp_factor)

    # System losses
    system_factor = (1 - soiling_loss) * (1 - mismatch_loss) * (1 - wiring_loss)

    # Total AC
    p_ac = p_dc * n_modules * inverter_efficiency * system_factor
    return max(0.0, p_ac)


# ---------------------------------------------------------------------------
# 5.  Solar resource assessment from hourly data
# ---------------------------------------------------------------------------
def assess_solar_resource(
    ghi_hourly: np.ndarray,
    temp_hourly: np.ndarray,
    lat: float,
    lon: float,
    module: PVModule = None,
    n_modules: int = 100,
    tilt_deg: float = None,
) -> Dict[str, Any]:
    """
    Comprehensive solar resource assessment.

    Args:
        ghi_hourly: Global Horizontal Irradiance array [W/m²]
        temp_hourly: Ambient temperature array [°C]
        lat: latitude
        lon: longitude
        module: PV module (default: Generic Mono-Si 550W)
        n_modules: number of modules
        tilt_deg: fixed tilt (default: optimal for latitude)

    Returns:
        Dict with energy yield, capacity factor, PR, specific yield, etc.
    """
    if module is None:
        module = PV_CATALOG["Generic Mono-Si 550W"]
    if tilt_deg is None:
        tilt_deg = optimal_tilt_angle(lat)["annual_optimal"]

    ghi = np.asarray(ghi_hourly, dtype=float)
    temp = np.asarray(temp_hourly, dtype=float)
    n_hours = len(ghi)

    # --- Hourly energy calc ---
    energy_wh = np.zeros(n_hours)
    for h in range(n_hours):
        if ghi[h] <= 0:
            continue
        t_cell = cell_temperature(temp[h], ghi[h])
        p_ac = pv_power_output(ghi[h], module, t_cell, n_modules)
        energy_wh[h] = p_ac  # W × 1h = Wh

    total_kwh = float(energy_wh.sum() / 1000)
    system_kwp = module.rated_wp * n_modules / 1000

    # Peak sun hours
    psh = float(ghi[ghi > 0].sum() / STC_IRRADIANCE) if (ghi > 0).any() else 0
    avg_daily_psh = psh / max(1, n_hours / 24)

    # Capacity factor
    cf = total_kwh / (system_kwp * n_hours) if system_kwp > 0 and n_hours > 0 else 0

    # Performance Ratio (simplified)
    total_insolation_kwh = float(ghi.sum() / 1000) * module.area_m2 * n_modules
    pr = total_kwh / (total_insolation_kwh * module.efficiency) if total_insolation_kwh > 0 else 0

    # Specific yield
    specific_yield = total_kwh / system_kwp if system_kwp > 0 else 0

    # Scale to annual
    hours_per_year = 8760
    scale = hours_per_year / n_hours if n_hours > 0 else 1
    annual_kwh = total_kwh * scale

    return {
        "system_kwp": system_kwp,
        "n_modules": n_modules,
        "module_name": module.name,
        "tilt_deg": tilt_deg,
        "total_kwh": total_kwh,
        "annual_kwh_est": annual_kwh,
        "avg_daily_kwh": total_kwh / max(1, n_hours / 24),
        "capacity_factor": float(cf),
        "performance_ratio": float(min(pr, 1.0)),
        "specific_yield_kwh_kwp": specific_yield,
        "peak_sun_hours_total": psh,
        "avg_daily_psh": avg_daily_psh,
        "avg_ghi_wm2": float(ghi.mean()),
        "max_ghi_wm2": float(ghi.max()),
        "avg_temp_c": float(temp.mean()),
    }


# ---------------------------------------------------------------------------
# 6.  Solar LCOE
# ---------------------------------------------------------------------------
def compute_solar_lcoe(
    system_kwp: float,
    annual_kwh: float,
    capex_per_kwp: float = 800.0,     # $/kWp (utility-scale Morocco 2025)
    discount_rate: float = 0.06,
    lifetime_yr: int = 25,
    opex_per_kwp_yr: float = 12.0,    # $/kWp/yr
    degradation_rate: float = 0.005,
) -> float:
    """LCOE for a solar PV system [$/kWh]."""
    capex = system_kwp * capex_per_kwp
    total_energy = 0.0
    total_cost = capex

    for t in range(1, lifetime_yr + 1):
        degraded_kwh = annual_kwh * (1 - degradation_rate) ** t
        total_energy += degraded_kwh / (1 + discount_rate) ** t
        total_cost += (system_kwp * opex_per_kwp_yr) / (1 + discount_rate) ** t

    return total_cost / max(1, total_energy)


# ---------------------------------------------------------------------------
# 7.  Monthly solar profile estimation
# ---------------------------------------------------------------------------
def estimate_monthly_profile(
    lat: float,
    annual_ghi_kwh_m2: float = 2000.0,
) -> pd.DataFrame:
    """
    Estimate monthly GHI and energy output profile from latitude.
    Uses a sinusoidal model calibrated for Morocco.
    """
    months = list(range(1, 13))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    daily_avg = annual_ghi_kwh_m2 / 365.0

    # Day-of-year mid-month
    doy_mid = []
    cum = 0
    for d in days_in_month:
        doy_mid.append(cum + d // 2)
        cum += d

    monthly_ghi = []
    for i, doy in enumerate(doy_mid):
        dl = day_length_hours(lat, doy)
        # Seasonal factor: longer days → more irradiance
        factor = dl / 12.0
        monthly_ghi.append(daily_avg * factor * days_in_month[i])

    # Normalise to match annual total
    total = sum(monthly_ghi)
    if total > 0:
        monthly_ghi = [g * annual_ghi_kwh_m2 / total for g in monthly_ghi]

    return pd.DataFrame({
        "month": months,
        "month_name": month_names,
        "days": days_in_month,
        "ghi_kwh_m2": monthly_ghi,
        "daily_avg_kwh_m2": [g / d for g, d in zip(monthly_ghi, days_in_month)],
    })


# ---------------------------------------------------------------------------
# 8.  Soiling model for Morocco
# ---------------------------------------------------------------------------
def morocco_soiling_loss(
    climate: str,
    days_since_rain: int = 14,
    days_since_cleaning: int = 30,
) -> Dict[str, Any]:
    """
    Soiling loss model calibrated for Moroccan climates.
    Returns soiling loss fraction and cleaning recommendation.
    """
    daily_rates = {
        "desert": 0.008,       # 0.8%/day
        "semi_arid": 0.005,    # 0.5%/day
        "coastal": 0.003,      # 0.3%/day
        "tropical": 0.002,
    }
    rate = daily_rates.get(climate, 0.005)

    # Rain partially cleans panels
    effective_days = min(days_since_rain, days_since_cleaning)
    soiling = min(0.30, rate * effective_days)

    # Recommended cleaning interval (when loss > 5%)
    recommended_interval = max(7, int(0.05 / rate)) if rate > 0 else 90

    return {
        "soiling_loss_pct": soiling * 100,
        "transmission_factor": 1 - soiling,
        "daily_rate_pct": rate * 100,
        "effective_dirty_days": effective_days,
        "recommended_cleaning_days": recommended_interval,
        "climate": climate,
    }
