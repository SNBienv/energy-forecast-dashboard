# =============================================================================
# Copernicus Climate Data Store (CDS) Module
# ERA5 reanalysis data via the official cdsapi
# =============================================================================
#
# This module mirrors the interface of weather_wind.py and solar_energy.py
# so data from Copernicus can be used as a drop-in alternative to Open-Meteo.
#
# Products used
# -------------
#   • reanalysis-era5-single-levels    (hourly, ~31 km grid)
#     – 2 m temperature, 2 m dewpoint, 10 m u/v wind, surface pressure,
#       total cloud cover, total precipitation, surface solar radiation
#       downwards (SSRD), surface net solar radiation (SSR), mean sea-level
#       pressure, etc.
#
# The CDS API is authenticated via the ~/.cdsapirc file that the user already
# possesses.
#
# IMPORTANT: CDS requests may take several minutes.  The module caches
# downloaded GRIB/NetCDF files locally so repeated requests are instant.
# =============================================================================

from __future__ import annotations

import os
import math
import tempfile
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

# Local cache directory for downloaded CDS files
_CACHE_DIR = Path(tempfile.gettempdir()) / "cds_cache"
_CACHE_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CDS credential helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_cds_credentials() -> Optional[Dict[str, str]]:
    """Return CDS credentials from Streamlit secrets or ~/.cdsapirc."""
    # 1) Try Streamlit secrets (for cloud deployment)
    try:
        import streamlit as st
        sec = st.secrets.get("cds", {})
        url = sec.get("url", "").strip()
        key = sec.get("key", "").strip()
        if url and key:
            return {"url": url, "key": key}
    except Exception:
        pass
    # 2) Try environment variables
    url = os.environ.get("CDSAPI_URL", "").strip()
    key = os.environ.get("CDSAPI_KEY", "").strip()
    if url and key:
        return {"url": url, "key": key}
    # 3) Fall back to ~/.cdsapirc (cdsapi default)
    rc = Path.home() / ".cdsapirc"
    if rc.exists():
        return None  # let cdsapi.Client() read it automatically
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _cache_path(request_dict: dict, ext: str = ".nc") -> Path:
    """Deterministic local file path for a CDS request."""
    key = hashlib.sha256(str(sorted(request_dict.items())).encode()).hexdigest()[:16]
    return _CACHE_DIR / f"era5_{key}{ext}"


def _wind_components_to_speed_dir(u: np.ndarray, v: np.ndarray):
    """Convert u/v wind components (m/s) → speed (km/h) and direction (°)."""
    speed_ms = np.sqrt(u**2 + v**2)
    speed_kmh = speed_ms * 3.6
    direction = (270 - np.rad2deg(np.arctan2(v, u))) % 360
    return speed_kmh, direction


def _kelvin_to_celsius(T: np.ndarray) -> np.ndarray:
    return T - 273.15


def _pa_to_hpa(p: np.ndarray) -> np.ndarray:
    return p / 100.0


def _j_per_m2_to_w_per_m2(accumulated_j: np.ndarray, dt_seconds: float = 3600.0) -> np.ndarray:
    """Convert accumulated J/m² (ERA5 convention) to mean W/m²."""
    return accumulated_j / dt_seconds


def _fractional_to_percent(frac: np.ndarray) -> np.ndarray:
    return frac * 100.0


def _m_to_mm(m: np.ndarray) -> np.ndarray:
    """Convert metres of water to mm (1 m = 1000 mm)."""
    return m * 1000.0


def _compute_rh(T_celsius: np.ndarray, Td_celsius: np.ndarray) -> np.ndarray:
    """Relative humidity from temperature and dewpoint (Magnus formula)."""
    e_s = 6.1094 * np.exp(17.625 * T_celsius / (243.04 + T_celsius))
    e_d = 6.1094 * np.exp(17.625 * Td_celsius / (243.04 + Td_celsius))
    return np.clip(100.0 * e_d / e_s, 0, 100)


def _compute_feels_like(T: np.ndarray, rh: np.ndarray, ws_kmh: np.ndarray) -> np.ndarray:
    """Simplified apparent temperature (Steadman / wind-chill hybrid)."""
    # Heat index for T > 27 °C
    hi = T.copy()
    hot = T > 27
    if np.any(hot):
        hi[hot] = (-8.784695 + 1.61139411 * T[hot] + 2.338549 * rh[hot]
                   - 0.14611605 * T[hot] * rh[hot]
                   - 0.012308094 * T[hot]**2 - 0.016424828 * rh[hot]**2
                   + 0.002211732 * T[hot]**2 * rh[hot]
                   + 0.00072546 * T[hot] * rh[hot]**2
                   - 0.000003582 * T[hot]**2 * rh[hot]**2)
    # Wind chill for T < 10 °C and wind > 4.8 km/h
    cold = (T < 10) & (ws_kmh > 4.8)
    if np.any(cold):
        hi[cold] = (13.12 + 0.6215 * T[cold]
                    - 11.37 * ws_kmh[cold]**0.16
                    + 0.3965 * T[cold] * ws_kmh[cold]**0.16)
    return hi


# ═══════════════════════════════════════════════════════════════════════════════
# CDS download  (single-level ERA5)
# ═══════════════════════════════════════════════════════════════════════════════

def _download_era5(
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    variables: List[str],
    product: str = "reanalysis-era5-single-levels",
) -> Optional[Path]:
    """
    Submit a CDS API request and download the result as NetCDF.

    Returns the local file path, or None on failure.
    Uses a local cache so identical requests are not repeated.
    """
    try:
        import cdsapi
    except ImportError:
        print("cdsapi is not installed.  pip install cdsapi")
        return None

    # Build date/time lists
    dates = pd.date_range(start.date(), end.date(), freq="D")
    years = sorted({str(d.year) for d in dates})
    months = sorted({f"{d.month:02d}" for d in dates})
    days = sorted({f"{d.day:02d}" for d in dates})
    hours = [f"{h:02d}:00" for h in range(24)]

    # Small bounding box (CDS allows area=[N, W, S, E])
    delta = 0.3  # ~33 km buffer
    area = [
        round(lat + delta, 2),
        round(lon - delta, 2),
        round(lat - delta, 2),
        round(lon + delta, 2),
    ]

    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": years,
        "month": months,
        "day": days,
        "time": hours,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }

    out_path = _cache_path(request, ".nc")
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path  # cache hit

    try:
        creds = _get_cds_credentials()
        if creds:
            client = cdsapi.Client(url=creds["url"], key=creds["key"], quiet=True)
        else:
            client = cdsapi.Client(quiet=True)
        client.retrieve(product, request, str(out_path))
        return out_path
    except Exception as exc:
        print(f"CDS download error: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ZIP handling (CDS may wrap NetCDF in a zip despite "unarchived")
# ═══════════════════════════════════════════════════════════════════════════════

import zipfile


def _unzip_if_needed(path: Path) -> Path:
    """If *path* is actually a ZIP archive, extract the first .nc inside
    and return the path to the extracted file.  Otherwise return *path*."""
    if not path.exists() or path.stat().st_size == 0:
        return path
    # Quick magic-number check: ZIP files start with PK\x03\x04
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic != b"PK\x03\x04":
        return path  # already a raw NetCDF
    try:
        with zipfile.ZipFile(path, "r") as zf:
            nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
            if not nc_names:
                return path  # nothing to extract
            extracted = path.parent / nc_names[0].replace("/", "_")
            if not extracted.exists():
                with zf.open(nc_names[0]) as src, open(extracted, "wb") as dst:
                    dst.write(src.read())
            return extracted
    except Exception:
        return path


# ═══════════════════════════════════════════════════════════════════════════════
# NetCDF → DataFrame helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _read_era5_nc(path: Path, lat: float, lon: float) -> Optional[pd.DataFrame]:
    """Read an ERA5 NetCDF and return a DataFrame for the nearest grid point."""
    # CDS sometimes wraps NetCDF in a ZIP archive — extract first
    path = _unzip_if_needed(path)

    try:
        import xarray as xr
    except ImportError:
        # Fallback: try netCDF4 directly
        try:
            import netCDF4
        except ImportError:
            print("Neither xarray nor netCDF4 is installed.")
            return None
        return _read_era5_nc_netcdf4(path, lat, lon)

    ds = xr.open_dataset(path, engine="netcdf4")

    # Select nearest grid point
    if "latitude" in ds.dims:
        ds = ds.sel(latitude=lat, longitude=lon, method="nearest")
    elif "lat" in ds.dims:
        ds = ds.sel(lat=lat, lon=lon, method="nearest")

    df = ds.to_dataframe().reset_index()

    # Normalise time column
    for col in ("valid_time", "time", "date"):
        if col in df.columns:
            df.rename(columns={col: "time"}, inplace=True)
            break

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)

    ds.close()
    return df


def _read_era5_nc_netcdf4(path: Path, lat: float, lon: float) -> Optional[pd.DataFrame]:
    """Minimal fallback reader using netCDF4."""
    import netCDF4 as nc
    ds = nc.Dataset(str(path))

    # Find nearest grid point
    lats = ds.variables.get("latitude", ds.variables.get("lat"))[:]
    lons = ds.variables.get("longitude", ds.variables.get("lon"))[:]
    ilat = int(np.argmin(np.abs(lats - lat)))
    ilon = int(np.argmin(np.abs(lons - lon)))

    time_var = ds.variables.get("valid_time", ds.variables.get("time"))
    times = nc.num2date(time_var[:], units=time_var.units,
                        calendar=getattr(time_var, "calendar", "standard"))
    times = pd.to_datetime([t.isoformat() for t in times])

    data = {"time": times}
    skip = {"latitude", "longitude", "lat", "lon", "time", "valid_time",
            "number", "expver"}
    for vname in ds.variables:
        if vname in skip:
            continue
        v = ds.variables[vname]
        if v.ndim >= 3:
            data[vname] = v[:, ilat, ilon]
        elif v.ndim == 1 and v.dimensions[0] in ("valid_time", "time"):
            data[vname] = v[:]

    ds.close()
    return pd.DataFrame(data).sort_values("time").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — Weather
# ═══════════════════════════════════════════════════════════════════════════════

_ERA5_WEATHER_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "instantaneous_10m_wind_gust",
    "mean_sea_level_pressure",
    "total_cloud_cover",
    "total_precipitation",
]

# Maps ERA5 short names in the NetCDF to our column convention
_ERA5_SHORT = {
    "t2m":   "temperature_2m",
    "d2m":   "dewpoint_2m",
    "u10":   "u10",
    "v10":   "v10",
    "i10fg": "wind_gusts_10m",
    "msl":   "pressure_msl",
    "tcc":   "cloud_cover",
    "tp":    "total_precipitation",
}


def fetch_weather_data_cds(
    lat: float,
    lon: float,
    days: int = 7,
    resolution: str = "hourly",
) -> Optional[Dict[str, Any]]:
    """
    Fetch weather data from Copernicus ERA5 reanalysis.

    Returns a dictionary in the same format as
    ``weather_wind.fetch_weather_data`` so it can be passed directly
    to ``process_weather_data``.
    """
    end = datetime.utcnow()
    # ERA5 has ~5 day latency; fall back to most recent available data
    end = end - timedelta(days=5)
    start = end - timedelta(days=max(days, 1))

    path = _download_era5(lat, lon, start, end, _ERA5_WEATHER_VARS)
    if path is None:
        return None

    raw = _read_era5_nc(path, lat, lon)
    if raw is None or raw.empty:
        return None

    # ── Unit conversions ──────────────────────────────────────────────────
    temp_c = _kelvin_to_celsius(raw.get("t2m", raw.get("temperature_2m", pd.Series(dtype=float))).values)
    dew_c  = _kelvin_to_celsius(raw.get("d2m", raw.get("dewpoint_2m", pd.Series(dtype=float))).values)
    u10 = raw.get("u10", pd.Series(0, index=raw.index)).values.astype(float)
    v10 = raw.get("v10", pd.Series(0, index=raw.index)).values.astype(float)
    ws_kmh, wdir = _wind_components_to_speed_dir(u10, v10)
    gusts_col = raw.get("i10fg", raw.get("wind_gusts_10m", pd.Series(0, index=raw.index)))
    gusts_kmh = gusts_col.values.astype(float) * 3.6  # m/s → km/h
    pressure_hpa = _pa_to_hpa(raw.get("msl", raw.get("pressure_msl", pd.Series(101325, index=raw.index))).values.astype(float))
    cloud_pct = _fractional_to_percent(raw.get("tcc", raw.get("cloud_cover", pd.Series(0, index=raw.index))).values.astype(float))
    precip_mm = _m_to_mm(raw.get("tp", raw.get("total_precipitation", pd.Series(0, index=raw.index))).values.astype(float))
    precip_mm = np.clip(precip_mm, 0, None)
    humidity = _compute_rh(temp_c, dew_c)
    feels_like = _compute_feels_like(temp_c, humidity, ws_kmh)

    times = raw["time"].tolist()

    # Build the same JSON-like dict that Open-Meteo returns
    result = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "GMT",
        "elevation": None,
        "_resolution": resolution,
        "_source": "copernicus",
        "hourly": {
            "time": [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in times],
            "temperature_2m": temp_c.tolist(),
            "apparent_temperature": feels_like.tolist(),
            "relative_humidity_2m": humidity.tolist(),
            "dew_point_2m": dew_c.tolist(),
            "precipitation_probability": [0] * len(times),   # not in ERA5
            "precipitation": precip_mm.tolist(),
            "rain": precip_mm.tolist(),                       # ERA5 tp ≈ rain
            "weather_code": [0] * len(times),                 # not in ERA5 directly
            "pressure_msl": pressure_hpa.tolist(),
            "cloud_cover": cloud_pct.tolist(),
            "visibility": [10000] * len(times),               # not in ERA5 single-level
            "wind_speed_10m": ws_kmh.tolist(),
            "wind_direction_10m": wdir.tolist(),
            "wind_gusts_10m": gusts_kmh.tolist(),
            "uv_index": [0] * len(times),                    # not in ERA5 single-level
        },
        "current": {
            "temperature_2m": float(temp_c[-1]),
            "apparent_temperature": float(feels_like[-1]),
            "relative_humidity_2m": float(humidity[-1]),
            "precipitation": float(precip_mm[-1]),
            "rain": float(precip_mm[-1]),
            "weather_code": 0,
            "cloud_cover": float(cloud_pct[-1]),
            "pressure_msl": float(pressure_hpa[-1]),
            "wind_speed_10m": float(ws_kmh[-1]),
            "wind_direction_10m": float(wdir[-1]),
            "wind_gusts_10m": float(gusts_kmh[-1]),
            "is_day": 1,
            "time": times[-1].isoformat() if hasattr(times[-1], "isoformat") else str(times[-1]),
        },
        "daily": {
            "time": [],
            "weather_code": [],
            "temperature_2m_max": [],
            "temperature_2m_min": [],
            "apparent_temperature_max": [],
            "apparent_temperature_min": [],
            "sunrise": [],
            "sunset": [],
            "daylight_duration": [],
            "sunshine_duration": [],
            "uv_index_max": [],
            "precipitation_sum": [],
            "rain_sum": [],
            "precipitation_hours": [],
            "precipitation_probability_max": [],
            "wind_speed_10m_max": [],
            "wind_gusts_10m_max": [],
            "wind_direction_10m_dominant": [],
        },
    }

    # Build daily aggregation from hourly arrays
    hdf = pd.DataFrame({
        "time": pd.to_datetime(result["hourly"]["time"]),
        "temp": temp_c,
        "feels": feels_like,
        "ws": ws_kmh,
        "gusts": gusts_kmh,
        "wdir": wdir,
        "precip": precip_mm,
    })
    hdf["_date"] = hdf["time"].dt.date
    for date, grp in hdf.groupby("_date"):
        ds = str(date)
        result["daily"]["time"].append(ds)
        result["daily"]["weather_code"].append(0)
        result["daily"]["temperature_2m_max"].append(float(grp["temp"].max()))
        result["daily"]["temperature_2m_min"].append(float(grp["temp"].min()))
        result["daily"]["apparent_temperature_max"].append(float(grp["feels"].max()))
        result["daily"]["apparent_temperature_min"].append(float(grp["feels"].min()))
        result["daily"]["sunrise"].append("")
        result["daily"]["sunset"].append("")
        result["daily"]["daylight_duration"].append(0)
        result["daily"]["sunshine_duration"].append(0)
        result["daily"]["uv_index_max"].append(0)
        result["daily"]["precipitation_sum"].append(float(grp["precip"].sum()))
        result["daily"]["rain_sum"].append(float(grp["precip"].sum()))
        result["daily"]["precipitation_hours"].append(int((grp["precip"] > 0.1).sum()))
        result["daily"]["precipitation_probability_max"].append(0)
        result["daily"]["wind_speed_10m_max"].append(float(grp["ws"].max()))
        result["daily"]["wind_gusts_10m_max"].append(float(grp["gusts"].max()))
        result["daily"]["wind_direction_10m_dominant"].append(float(grp["wdir"].mode().iloc[0]) if not grp["wdir"].mode().empty else 0)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — Solar
# ═══════════════════════════════════════════════════════════════════════════════

_ERA5_SOLAR_VARS = [
    "surface_solar_radiation_downwards",       # SSRD, J/m² accumulated
    "surface_net_solar_radiation",             # SSR,  J/m² accumulated
    "total_sky_direct_solar_radiation_at_surface",  # FDIR, J/m²
    "surface_solar_radiation_downward_clear_sky",   # SSRDC, J/m²
]


def fetch_solar_data_cds(
    lat: float,
    lon: float,
    days: int = 7,
    resolution: str = "hourly",
) -> Optional[Dict[str, Any]]:
    """
    Fetch solar radiation data from Copernicus ERA5.

    Returns a dictionary in the same format as
    ``solar_energy.fetch_solar_radiation_data`` so it can be passed to
    ``process_solar_data``.
    """
    end = datetime.utcnow() - timedelta(days=5)
    start = end - timedelta(days=max(days, 1))

    path = _download_era5(lat, lon, start, end, _ERA5_SOLAR_VARS)
    if path is None:
        return None

    raw = _read_era5_nc(path, lat, lon)
    if raw is None or raw.empty:
        return None

    # The ERA5 radiation fields are accumulated over each hour (J/m²).
    # Convert to mean W/m² by dividing by 3600 s.
    dt_s = 3600.0
    ssrd = _j_per_m2_to_w_per_m2(
        raw.get("ssrd", raw.get("surface_solar_radiation_downwards",
                pd.Series(0, index=raw.index))).values.astype(float), dt_s)
    ssrd = np.clip(ssrd, 0, None)

    ssr = _j_per_m2_to_w_per_m2(
        raw.get("ssr", raw.get("surface_net_solar_radiation",
                pd.Series(0, index=raw.index))).values.astype(float), dt_s)
    ssr = np.clip(ssr, 0, None)

    fdir = _j_per_m2_to_w_per_m2(
        raw.get("fdir", raw.get("total_sky_direct_solar_radiation_at_surface",
                pd.Series(0, index=raw.index))).values.astype(float), dt_s)
    fdir = np.clip(fdir, 0, None)

    ssrdc = _j_per_m2_to_w_per_m2(
        raw.get("ssrdc", raw.get("surface_solar_radiation_downward_clear_sky",
                pd.Series(0, index=raw.index))).values.astype(float), dt_s)
    ssrdc = np.clip(ssrdc, 0, None)

    # Diffuse = GHI(SSRD) − Direct(FDIR)
    diffuse = np.clip(ssrd - fdir, 0, None)

    # DNI ≈ fdir / cos(zenith).  For simplicity, use fdir directly as a proxy.
    dni = fdir.copy()

    times = raw["time"].tolist()

    result = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "GMT",
        "elevation": None,
        "_resolution": resolution,
        "_source": "copernicus",
        "hourly": {
            "time": [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in times],
            "shortwave_radiation": ssrd.tolist(),
            "direct_radiation": fdir.tolist(),
            "diffuse_radiation": diffuse.tolist(),
            "direct_normal_irradiance": dni.tolist(),
            "global_tilted_irradiance": ssrd.tolist(),   # ≈ GHI on flat surface
            "terrestrial_radiation": ssrdc.tolist(),
            "shortwave_radiation_instant": ssrd.tolist(),
        },
        "daily": {
            "time": [],
            "shortwave_radiation_sum": [],
            "sunrise": [],
            "sunset": [],
            "sunshine_duration": [],
            "daylight_duration": [],
        },
    }

    # Daily aggregation
    hdf = pd.DataFrame({"time": pd.to_datetime(result["hourly"]["time"]),
                         "ghi": ssrd})
    hdf["_date"] = hdf["time"].dt.date
    for date, grp in hdf.groupby("_date"):
        result["daily"]["time"].append(str(date))
        result["daily"]["shortwave_radiation_sum"].append(float(grp["ghi"].sum() * 0.0036))  # Wh→MJ
        result["daily"]["sunrise"].append("")
        result["daily"]["sunset"].append("")
        result["daily"]["sunshine_duration"].append(float((grp["ghi"] > 10).sum() * 3600))
        result["daily"]["daylight_duration"].append(float((grp["ghi"] > 0).sum() * 3600))

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Availability check
# ═══════════════════════════════════════════════════════════════════════════════

def cds_available() -> bool:
    """Return True if cdsapi can be imported AND credentials are available."""
    try:
        import cdsapi  # noqa: F401
    except Exception:
        return False
    # Check Streamlit secrets
    try:
        import streamlit as st
        sec = st.secrets.get("cds", {})
        if sec.get("url", "").strip() and sec.get("key", "").strip():
            return True
    except Exception:
        pass
    # Check env vars
    if os.environ.get("CDSAPI_URL", "").strip() and os.environ.get("CDSAPI_KEY", "").strip():
        return True
    # Check .cdsapirc
    return (Path.home() / ".cdsapirc").exists()
