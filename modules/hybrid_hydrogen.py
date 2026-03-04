# =============================================================================
# Hybrid Wind+Solar System  &  Green Hydrogen Production Module
# Morocco-focused: electrolysis sizing, LCOH, storage, and dispatch
# =============================================================================

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Electrolyser catalogue
# ---------------------------------------------------------------------------
@dataclass
class Electrolyser:
    name: str
    technology: str           # PEM, Alkaline, SOEC
    rated_kw: float           # nominal electrical input [kW]
    efficiency: float         # HHV efficiency (decimal, 0-1)
    specific_energy: float    # kWh/kg H2  (electrical)
    water_consumption: float  # litres/kg H2
    capex_per_kw: float       # $/kW
    opex_pct: float           # annual OPEX as % of CAPEX
    lifetime_yr: int
    min_load: float           # minimum part-load fraction
    ramp_rate: float          # fraction/min (1.0 = instant)
    degradation_yr: float     # annual efficiency degradation (decimal)
    stack_life_hours: int     # hours before stack replacement

    @property
    def h2_rate_kg_hr(self) -> float:
        """Max H2 production rate [kg/h]."""
        return self.rated_kw / self.specific_energy


ELECTROLYSER_CATALOG: Dict[str, Electrolyser] = {
    "PEM 1 MW": Electrolyser(
        "PEM 1 MW", "PEM", 1000, 0.60, 55.0, 10.0,
        1200, 3.0, 20, 0.10, 1.0, 0.01, 80000,
    ),
    "PEM 5 MW": Electrolyser(
        "PEM 5 MW", "PEM", 5000, 0.62, 53.0, 9.5,
        1000, 2.5, 20, 0.10, 1.0, 0.01, 80000,
    ),
    "PEM 10 MW": Electrolyser(
        "PEM 10 MW", "PEM", 10000, 0.63, 52.5, 9.5,
        900, 2.5, 25, 0.10, 1.0, 0.008, 90000,
    ),
    "Alkaline 2 MW": Electrolyser(
        "Alkaline 2 MW", "Alkaline", 2000, 0.65, 50.0, 10.5,
        800, 3.5, 25, 0.20, 0.5, 0.008, 90000,
    ),
    "Alkaline 10 MW": Electrolyser(
        "Alkaline 10 MW", "Alkaline", 10000, 0.67, 49.0, 10.0,
        650, 3.0, 25, 0.20, 0.5, 0.007, 100000,
    ),
    "SOEC 1 MW (pilot)": Electrolyser(
        "SOEC 1 MW (pilot)", "SOEC", 1000, 0.80, 40.0, 9.0,
        2500, 4.0, 15, 0.30, 0.3, 0.015, 40000,
    ),
}


# ---------------------------------------------------------------------------
# Morocco Green Hydrogen context (GGGI / MASEN / IRENA data)
# ---------------------------------------------------------------------------
MOROCCO_H2_CONTEXT = {
    "national_target_GW_2030": 6,         # GW electrolyser capacity
    "national_target_GW_2050": 40,
    "export_target_Mt_yr_2030": 0.3,      # Million tonnes H2/yr
    "water_cost_m3": 1.50,                # $/m³ (desalinated)
    "electricity_cost_kwh": 0.04,         # $/kWh (RE-dedicated)
    "h2_transport_cost_kg": 1.50,         # $/kg (pipeline)
    "h2_market_price_kg": 4.00,           # $/kg (green premium)
    "co2_avoided_kg_per_kg_h2": 10.0,     # kg CO2 per kg H2 (vs grey)
    "carbon_credit_usd_ton": 30.0,
    "green_ammonia_conversion_kwh_kg": 0.6,  # kWh/kg NH3 from H2
}

# Moroccan RE zones for hybrid projects
MOROCCO_H2_ZONES: Dict[str, Dict[str, Any]] = {
    "Guelmim-Oued Noun": {
        "lat": 29.28, "lon": -10.06,
        "wind_cf": 0.35, "solar_cf": 0.22,
        "ghi_kwh_m2_yr": 2200, "mean_ws_ms": 8.5,
        "water_source": "desalination",
    },
    "Dakhla-Oued Ed-Dahab": {
        "lat": 23.68, "lon": -15.96,
        "wind_cf": 0.45, "solar_cf": 0.24,
        "ghi_kwh_m2_yr": 2180, "mean_ws_ms": 9.5,
        "water_source": "desalination",
    },
    "Laâyoune-Sakia El Hamra": {
        "lat": 27.15, "lon": -13.20,
        "wind_cf": 0.40, "solar_cf": 0.23,
        "ghi_kwh_m2_yr": 2200, "mean_ws_ms": 9.0,
        "water_source": "desalination",
    },
    "Midelt Region": {
        "lat": 32.68, "lon": -4.73,
        "wind_cf": 0.28, "solar_cf": 0.21,
        "ghi_kwh_m2_yr": 2050, "mean_ws_ms": 6.5,
        "water_source": "atlas_runoff",
    },
    "Tanger-Tétouan (offshore)": {
        "lat": 35.78, "lon": -5.80,
        "wind_cf": 0.38, "solar_cf": 0.18,
        "ghi_kwh_m2_yr": 1750, "mean_ws_ms": 8.0,
        "water_source": "desalination",
    },
    "Essaouira-Meskala": {
        "lat": 30.85, "lon": -9.48,
        "wind_cf": 0.32, "solar_cf": 0.21,
        "ghi_kwh_m2_yr": 2000, "mean_ws_ms": 7.5,
        "water_source": "mixed",
    },
}


# ---------------------------------------------------------------------------
# 1.  Hybrid system power dispatch  (hourly)
# ---------------------------------------------------------------------------
def hybrid_dispatch(
    wind_power_kw: np.ndarray,
    solar_power_kw: np.ndarray,
    electrolyser: Electrolyser,
    n_electrolysers: int = 1,
    battery_kwh: float = 0.0,
    battery_efficiency: float = 0.90,
    grid_export_limit_kw: float = 0.0,
) -> Dict[str, Any]:
    """
    Hourly dispatch simulation for a hybrid wind+solar plant feeding an electrolyser.

    Args:
        wind_power_kw: hourly wind power output [kW]
        solar_power_kw: hourly solar power output [kW]
        electrolyser: electrolyser specs
        n_electrolysers: number of parallel units
        battery_kwh: battery storage capacity [kWh] (0 = none)
        battery_efficiency: round-trip efficiency
        grid_export_limit_kw: max grid export (0 = no export)

    Returns:
        Dict with hourly arrays and summary metrics.
    """
    n_hours = len(wind_power_kw)
    total_elec_kw = electrolyser.rated_kw * n_electrolysers
    min_elec_kw = electrolyser.min_load * total_elec_kw

    # Output arrays
    h2_kg = np.zeros(n_hours)
    elec_input_kw = np.zeros(n_hours)
    curtailed_kw = np.zeros(n_hours)
    grid_export_kw = np.zeros(n_hours)
    battery_soc = np.zeros(n_hours)
    battery_charge = np.zeros(n_hours)
    battery_discharge = np.zeros(n_hours)

    soc = 0.0  # battery state of charge [kWh]

    for h in range(n_hours):
        total_re = wind_power_kw[h] + solar_power_kw[h]

        available = total_re

        # Battery discharge if RE insufficient
        if available < min_elec_kw and soc > 0 and battery_kwh > 0:
            needed = min_elec_kw - available
            discharge = min(needed, soc * math.sqrt(battery_efficiency))
            available += discharge
            soc -= discharge / math.sqrt(battery_efficiency)
            battery_discharge[h] = discharge

        # Electrolyser input (clamp to rated)
        if available >= min_elec_kw:
            e_input = min(available, total_elec_kw)
        else:
            e_input = 0.0

        elec_input_kw[h] = e_input
        h2_kg[h] = e_input / electrolyser.specific_energy

        surplus = available - e_input

        # Battery charging
        if surplus > 0 and battery_kwh > 0:
            charge = min(surplus, (battery_kwh - soc) / math.sqrt(battery_efficiency))
            soc += charge * math.sqrt(battery_efficiency)
            battery_charge[h] = charge
            surplus -= charge

        # Grid export
        if surplus > 0 and grid_export_limit_kw > 0:
            export = min(surplus, grid_export_limit_kw)
            grid_export_kw[h] = export
            surplus -= export

        curtailed_kw[h] = max(0, surplus)
        battery_soc[h] = soc

    total_h2 = float(h2_kg.sum())
    total_re_kwh = float((wind_power_kw + solar_power_kw).sum())
    elec_total_kwh = float(elec_input_kw.sum())
    curtailed_total = float(curtailed_kw.sum())

    elec_cf = elec_total_kwh / (total_elec_kw * n_hours) if total_elec_kw * n_hours > 0 else 0
    re_utilisation = 1 - curtailed_total / total_re_kwh if total_re_kwh > 0 else 0

    return {
        "total_h2_kg": total_h2,
        "total_h2_tonnes": total_h2 / 1000,
        "annual_h2_tonnes_est": total_h2 / 1000 * (8760 / n_hours) if n_hours > 0 else 0,
        "electrolyser_cf": float(elec_cf),
        "re_utilisation_pct": float(re_utilisation * 100),
        "total_re_mwh": total_re_kwh / 1000,
        "wind_share_pct": float(wind_power_kw.sum() / total_re_kwh * 100) if total_re_kwh > 0 else 0,
        "solar_share_pct": float(solar_power_kw.sum() / total_re_kwh * 100) if total_re_kwh > 0 else 0,
        "curtailed_mwh": curtailed_total / 1000,
        "curtailment_pct": float(curtailed_total / total_re_kwh * 100) if total_re_kwh > 0 else 0,
        "grid_export_mwh": float(grid_export_kw.sum() / 1000),
        "water_consumption_m3": total_h2 * electrolyser.water_consumption / 1000,
        "hourly": {
            "h2_kg": h2_kg,
            "elec_input_kw": elec_input_kw,
            "curtailed_kw": curtailed_kw,
            "battery_soc_kwh": battery_soc,
            "grid_export_kw": grid_export_kw,
        },
    }


# ---------------------------------------------------------------------------
# 2.  LCOH  (Levelised Cost of Hydrogen)
# ---------------------------------------------------------------------------
def compute_lcoh(
    electrolyser: Electrolyser,
    n_electrolysers: int,
    annual_h2_kg: float,
    wind_capex: float = 0.0,        # total wind CAPEX [$]
    solar_capex: float = 0.0,       # total solar CAPEX [$]
    battery_capex: float = 0.0,
    desal_capex: float = 0.0,       # desalination plant [$]
    wind_opex_yr: float = 0.0,
    solar_opex_yr: float = 0.0,
    discount_rate: float = 0.07,
    lifetime_yr: int = 25,
    stack_replacement_cost: float = 0.40,  # fraction of elec CAPEX
    water_cost_m3: float = 1.50,
) -> Dict[str, float]:
    """
    Compute LCOH [$/kg H2] with full cost breakdown.
    """
    elec_capex = electrolyser.capex_per_kw * electrolyser.rated_kw * n_electrolysers
    total_capex = elec_capex + wind_capex + solar_capex + battery_capex + desal_capex

    elec_opex_yr = elec_capex * (electrolyser.opex_pct / 100)

    # Stack replacement (once at ~half-life)
    stack_cost = elec_capex * stack_replacement_cost
    stack_year = max(1, lifetime_yr // 2)

    # Water cost
    water_yr = (annual_h2_kg * electrolyser.water_consumption / 1000) * water_cost_m3

    # NPV of costs
    total_cost_npv = total_capex
    total_h2_npv = 0.0

    for t in range(1, lifetime_yr + 1):
        df = (1 + discount_rate) ** t
        annual_opex = elec_opex_yr + wind_opex_yr + solar_opex_yr + water_yr
        total_cost_npv += annual_opex / df
        if t == stack_year:
            total_cost_npv += stack_cost / df
        # Degraded H2 production
        h2_t = annual_h2_kg * (1 - electrolyser.degradation_yr) ** t
        total_h2_npv += h2_t / df

    lcoh = total_cost_npv / max(1, total_h2_npv)

    # Cost breakdown per kg
    crf = (discount_rate * (1 + discount_rate) ** lifetime_yr) / (
        (1 + discount_rate) ** lifetime_yr - 1
    )
    capex_component = total_capex * crf / max(1, annual_h2_kg)
    opex_component = (elec_opex_yr + wind_opex_yr + solar_opex_yr) / max(1, annual_h2_kg)
    water_component = water_yr / max(1, annual_h2_kg)

    return {
        "lcoh_usd_kg": float(lcoh),
        "capex_component": float(capex_component),
        "opex_component": float(opex_component),
        "water_component": float(water_component),
        "total_capex_musd": total_capex / 1e6,
        "electrolyser_capex_musd": elec_capex / 1e6,
        "wind_capex_musd": wind_capex / 1e6,
        "solar_capex_musd": solar_capex / 1e6,
    }


# ---------------------------------------------------------------------------
# 3.  Quick hybrid sizing tool  (Morocco presets)
# ---------------------------------------------------------------------------
def quick_hybrid_sizing(
    target_h2_tonnes_yr: float = 1.0,
    zone_name: str = "Guelmim-Oued Noun",
    electrolyser_name: str = "PEM 5 MW",
    wind_capex_per_kw: float = 1200.0,
    solar_capex_per_kwp: float = 700.0,
    wind_solar_ratio: float = 0.60,     # 60% wind, 40% solar
    battery_hours: float = 2.0,
    discount_rate: float = 0.07,
    h2_selling_price: float = None,
) -> Dict[str, Any]:
    """
    Quick parametric sizing of a hybrid wind+solar+H2 plant.
    Uses zone capacity factors and typical Morocco parameters.
    """
    zone = MOROCCO_H2_ZONES.get(zone_name)
    if zone is None:
        zone = list(MOROCCO_H2_ZONES.values())[0]

    elec = ELECTROLYSER_CATALOG.get(electrolyser_name)
    if elec is None:
        elec = ELECTROLYSER_CATALOG["PEM 5 MW"]

    if h2_selling_price is None:
        h2_selling_price = MOROCCO_H2_CONTEXT["h2_market_price_kg"]

    # Required annual H2 in kg
    annual_h2_kg = target_h2_tonnes_yr * 1000

    # Required electrolyser energy input
    annual_elec_kwh = annual_h2_kg * elec.specific_energy

    # Size RE to match electrolyser (accounting for CF)
    wind_cf = zone["wind_cf"]
    solar_cf = zone["solar_cf"]

    # Weighted average CF
    combined_cf = wind_solar_ratio * wind_cf + (1 - wind_solar_ratio) * solar_cf
    if combined_cf <= 0:
        combined_cf = 0.20

    # Required RE capacity (with 15% oversizing for curtailment/losses)
    re_kw_needed = annual_elec_kwh / (combined_cf * 8760) * 1.15
    wind_kw = re_kw_needed * wind_solar_ratio
    solar_kwp = re_kw_needed * (1 - wind_solar_ratio)

    # Electrolyser sizing
    elec_kw_needed = annual_elec_kwh / (8760 * 0.85)  # target 85% elec CF
    n_electrolysers = max(1, math.ceil(elec_kw_needed / elec.rated_kw))
    total_elec_kw = n_electrolysers * elec.rated_kw

    # Battery
    battery_kwh = total_elec_kw * battery_hours
    battery_capex_per_kwh = 250.0  # $/kWh (2025 utility-scale)

    # CAPEX
    wind_capex = wind_kw * wind_capex_per_kw
    solar_capex = solar_kwp * solar_capex_per_kwp
    elec_capex = total_elec_kw * elec.capex_per_kw
    bat_capex = battery_kwh * battery_capex_per_kwh

    # Desalination (if needed)
    water_m3_yr = annual_h2_kg * elec.water_consumption / 1000
    desal_capex = water_m3_yr * 3.0 * 1000 if zone.get("water_source") == "desalination" else 0

    total_capex = wind_capex + solar_capex + elec_capex + bat_capex + desal_capex

    # OPEX
    wind_opex_yr = wind_capex * 0.025
    solar_opex_yr = solar_capex * 0.015
    elec_opex_yr = elec_capex * (elec.opex_pct / 100)
    total_opex_yr = wind_opex_yr + solar_opex_yr + elec_opex_yr

    # LCOH
    lcoh_result = compute_lcoh(
        elec, n_electrolysers, annual_h2_kg,
        wind_capex, solar_capex, bat_capex, desal_capex,
        wind_opex_yr, solar_opex_yr, discount_rate,
    )

    # Revenue & economics
    annual_revenue = annual_h2_kg * h2_selling_price
    co2_credit = (annual_h2_kg * MOROCCO_H2_CONTEXT["co2_avoided_kg_per_kg_h2"]
                  / 1000 * MOROCCO_H2_CONTEXT["carbon_credit_usd_ton"])
    total_annual_income = annual_revenue + co2_credit

    crf = (discount_rate * (1 + discount_rate) ** 25) / ((1 + discount_rate) ** 25 - 1)
    npv = -total_capex
    for t in range(1, 26):
        npv += (total_annual_income - total_opex_yr) / (1 + discount_rate) ** t

    payback = total_capex / max(1, total_annual_income - total_opex_yr)

    return {
        "zone": zone_name,
        "target_h2_tonnes_yr": target_h2_tonnes_yr,
        "wind_capacity_mw": wind_kw / 1000,
        "solar_capacity_mwp": solar_kwp / 1000,
        "electrolyser_capacity_mw": total_elec_kw / 1000,
        "n_electrolysers": n_electrolysers,
        "electrolyser_type": elec.name,
        "battery_mwh": battery_kwh / 1000,
        "wind_cf": wind_cf,
        "solar_cf": solar_cf,
        "combined_re_cf": combined_cf,
        "annual_h2_kg": annual_h2_kg,
        "annual_h2_tonnes": target_h2_tonnes_yr,
        "water_m3_yr": water_m3_yr,
        "capex_breakdown": {
            "wind_musd": wind_capex / 1e6,
            "solar_musd": solar_capex / 1e6,
            "electrolyser_musd": elec_capex / 1e6,
            "battery_musd": bat_capex / 1e6,
            "desalination_musd": desal_capex / 1e6,
            "total_musd": total_capex / 1e6,
        },
        "opex_yr_musd": total_opex_yr / 1e6,
        "lcoh_usd_kg": lcoh_result["lcoh_usd_kg"],
        "h2_selling_price": h2_selling_price,
        "annual_revenue_musd": annual_revenue / 1e6,
        "co2_credit_yr_usd": co2_credit,
        "npv_musd": npv / 1e6,
        "payback_yr": payback,
        "co2_avoided_tonnes_yr": annual_h2_kg * MOROCCO_H2_CONTEXT["co2_avoided_kg_per_kg_h2"] / 1000,
    }


# ---------------------------------------------------------------------------
# 4.  Sensitivity analysis
# ---------------------------------------------------------------------------
def lcoh_sensitivity(
    base_params: Dict[str, Any],
    electrolyser_name: str = "PEM 5 MW",
    capex_range: List[float] = None,
    discount_rates: List[float] = None,
    elec_prices: List[float] = None,
) -> pd.DataFrame:
    """
    Parametric LCOH sensitivity sweep.
    Returns DataFrame with LCOH for each parameter combination.
    """
    if capex_range is None:
        capex_range = [800, 1000, 1200, 1500, 2000]
    if discount_rates is None:
        discount_rates = [0.05, 0.06, 0.07, 0.08, 0.10]
    if elec_prices is None:
        elec_prices = [700, 900, 1100, 1300]

    elec = ELECTROLYSER_CATALOG.get(electrolyser_name, ELECTROLYSER_CATALOG["PEM 5 MW"])
    annual_h2 = base_params.get("annual_h2_kg", 1000000)

    rows = []
    for wind_capex_kw in capex_range:
        for dr in discount_rates:
            for elec_price in elec_prices:
                modified_elec = Electrolyser(
                    elec.name, elec.technology, elec.rated_kw, elec.efficiency,
                    elec.specific_energy, elec.water_consumption, elec_price,
                    elec.opex_pct, elec.lifetime_yr, elec.min_load,
                    elec.ramp_rate, elec.degradation_yr, elec.stack_life_hours,
                )
                wind_cap = base_params.get("wind_capacity_mw", 10) * 1000
                solar_cap = base_params.get("solar_capacity_mwp", 5) * 1000
                result = compute_lcoh(
                    modified_elec,
                    base_params.get("n_electrolysers", 1),
                    annual_h2,
                    wind_capex=wind_cap * wind_capex_kw,
                    solar_capex=solar_cap * 700,
                    discount_rate=dr,
                )
                rows.append({
                    "wind_capex_$/kW": wind_capex_kw,
                    "discount_rate": dr,
                    "elec_capex_$/kW": elec_price,
                    "lcoh_$/kg": result["lcoh_usd_kg"],
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5.  Green ammonia conversion  (bonus: NH3 from H2)
# ---------------------------------------------------------------------------
def green_ammonia_production(
    h2_kg_yr: float,
    nh3_efficiency: float = 0.90,
) -> Dict[str, float]:
    """
    Estimate green ammonia (NH3) production from H2.
    Haber-Bosch: 3H2 + N2 → 2NH3
    Stoichiometric: 1 kg H2 → 5.56 kg NH3 (with efficiency)
    """
    nh3_kg_yr = h2_kg_yr * 5.56 * nh3_efficiency
    return {
        "nh3_tonnes_yr": nh3_kg_yr / 1000,
        "conversion_efficiency": nh3_efficiency,
        "h2_input_tonnes_yr": h2_kg_yr / 1000,
        "energy_input_mwh_yr": h2_kg_yr * MOROCCO_H2_CONTEXT["green_ammonia_conversion_kwh_kg"] / 1000,
    }


# ---------------------------------------------------------------------------
# 6.  Complementarity index  (wind–solar)
# ---------------------------------------------------------------------------
def complementarity_index(
    wind_power: np.ndarray,
    solar_power: np.ndarray,
) -> Dict[str, float]:
    """
    Compute hourly complementarity between wind and solar.
    CI = 1 - corr(wind, solar)  →  high CI = good complementarity.
    Also computes variability reduction.
    """
    if len(wind_power) < 2 or len(solar_power) < 2:
        return {"complementarity_index": 0, "correlation": 0, "variability_reduction_pct": 0}

    # Normalise
    w = wind_power / max(wind_power.max(), 1e-6)
    s = solar_power / max(solar_power.max(), 1e-6)

    corr = float(np.corrcoef(w, s)[0, 1]) if np.std(w) > 0 and np.std(s) > 0 else 0
    ci = 1.0 - corr

    # Variability reduction
    cv_wind = float(np.std(wind_power) / max(np.mean(wind_power), 1e-6))
    cv_solar = float(np.std(solar_power) / max(np.mean(solar_power), 1e-6))
    combined = wind_power + solar_power
    cv_combined = float(np.std(combined) / max(np.mean(combined), 1e-6))
    avg_cv = (cv_wind + cv_solar) / 2
    var_reduction = (1 - cv_combined / max(avg_cv, 1e-6)) * 100 if avg_cv > 0 else 0

    return {
        "complementarity_index": float(ci),
        "correlation": float(corr),
        "variability_reduction_pct": float(var_reduction),
        "cv_wind": cv_wind,
        "cv_solar": cv_solar,
        "cv_combined": cv_combined,
    }


# ---------------------------------------------------------------------------
# 7.  Morocco H2 roadmap comparison
# ---------------------------------------------------------------------------
def morocco_roadmap_comparison(
    project_h2_tonnes_yr: float,
    project_lcoh: float,
) -> Dict[str, Any]:
    """Compare project metrics against Morocco's national H2 targets."""
    ctx = MOROCCO_H2_CONTEXT
    target_2030_tonnes = ctx["export_target_Mt_yr_2030"] * 1e6
    share_of_2030 = project_h2_tonnes_yr / target_2030_tonnes * 100

    competitive = project_lcoh <= 3.0
    market_viable = project_lcoh <= ctx["h2_market_price_kg"]

    return {
        "project_h2_tonnes_yr": project_h2_tonnes_yr,
        "project_lcoh": project_lcoh,
        "morocco_2030_target_Mt": ctx["export_target_Mt_yr_2030"],
        "share_of_2030_target_pct": share_of_2030,
        "is_competitive_global": competitive,
        "is_market_viable": market_viable,
        "co2_avoided_tonnes_yr": project_h2_tonnes_yr * ctx["co2_avoided_kg_per_kg_h2"],
        "benchmark_lcoh_grey": 1.5,
        "benchmark_lcoh_blue": 2.5,
        "benchmark_lcoh_green_target": 2.0,
        "lcoh_gap_vs_grey": project_lcoh - 1.5,
        "water_m3_yr_est": project_h2_tonnes_yr * 1000 * 10 / 1000,
    }
