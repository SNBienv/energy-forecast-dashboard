# =============================================================================
# Visitor Analytics Module
# Track and geolocate visitors accessing the dashboard
# =============================================================================

from __future__ import annotations

import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests
import pandas as pd

# Persistent log file
_LOG_DIR = Path(__file__).resolve().parent.parent / "data"
_LOG_DIR.mkdir(exist_ok=True)
_VISITOR_LOG = _LOG_DIR / "visitors.json"


# ═══════════════════════════════════════════════════════════════════════════════
# IP detection
# ═══════════════════════════════════════════════════════════════════════════════

def get_visitor_ip() -> str:
    """
    Attempt to discover the public IP of the current visitor.
    On Streamlit Cloud the server-side IP is returned (still useful
    for counting unique sessions). For local deploys this returns
    the machine's public IP.
    """
    for svc in [
        "https://api.ipify.org?format=json",
        "https://ifconfig.me/all.json",
        "https://ipinfo.io/json",
    ]:
        try:
            r = requests.get(svc, timeout=4)
            r.raise_for_status()
            data = r.json()
            ip = data.get("ip", data.get("ip_addr", ""))
            if ip:
                return ip
        except Exception:
            continue
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# Geolocation  (free, no key required — ip-api.com, 45 req/min)
# ═══════════════════════════════════════════════════════════════════════════════

_GEO_CACHE: Dict[str, Dict[str, Any]] = {}


def geolocate_ip(ip: str) -> Dict[str, Any]:
    """Return geolocation dict for *ip*.  Cached in memory."""
    if ip in _GEO_CACHE:
        return _GEO_CACHE[ip]
    if ip in ("unknown", "127.0.0.1", "::1", ""):
        return {"ip": ip, "country": "Local", "city": "localhost",
                "lat": 0.0, "lon": 0.0, "isp": ""}
    try:
        r = requests.get(
            f"http://ip-api.com/json/{ip}?fields=status,message,country,"
            "countryCode,region,regionName,city,lat,lon,timezone,isp,query",
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "success":
            geo = {
                "ip": ip,
                "country": data.get("country", ""),
                "country_code": data.get("countryCode", ""),
                "region": data.get("regionName", ""),
                "city": data.get("city", ""),
                "lat": data.get("lat", 0.0),
                "lon": data.get("lon", 0.0),
                "timezone": data.get("timezone", ""),
                "isp": data.get("isp", ""),
            }
            _GEO_CACHE[ip] = geo
            return geo
    except Exception:
        pass
    return {"ip": ip, "country": "Unknown", "city": "", "lat": 0.0,
            "lon": 0.0, "isp": ""}


# ═══════════════════════════════════════════════════════════════════════════════
# Persistent visitor log
# ═══════════════════════════════════════════════════════════════════════════════

def _load_log() -> List[Dict[str, Any]]:
    if _VISITOR_LOG.exists():
        try:
            return json.loads(_VISITOR_LOG.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_log(records: List[Dict[str, Any]]) -> None:
    _VISITOR_LOG.write_text(
        json.dumps(records, indent=2, default=str),
        encoding="utf-8",
    )


def _anon_hash(ip: str) -> str:
    """SHA-256 hash of the IP for privacy."""
    return hashlib.sha256(ip.encode()).hexdigest()[:12]


def log_visitor(ip: str, geo: Dict[str, Any], page: str = "main") -> None:
    """Append a visit entry to the persistent log."""
    records = _load_log()
    records.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ip_hash": _anon_hash(ip),
        "ip": ip,
        "country": geo.get("country", ""),
        "country_code": geo.get("country_code", ""),
        "region": geo.get("region", ""),
        "city": geo.get("city", ""),
        "lat": geo.get("lat", 0.0),
        "lon": geo.get("lon", 0.0),
        "isp": geo.get("isp", ""),
        "page": page,
    })
    _save_log(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Analytics helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_visitor_df() -> pd.DataFrame:
    """Load visitor log into a DataFrame."""
    records = _load_log()
    if not records:
        return pd.DataFrame(columns=[
            "timestamp", "ip_hash", "ip", "country", "country_code",
            "region", "city", "lat", "lon", "isp", "page",
        ])
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def visitor_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute key analytics from visitor DataFrame."""
    if df.empty:
        return {
            "total_visits": 0,
            "unique_visitors": 0,
            "countries": 0,
            "cities": 0,
            "top_countries": pd.Series(dtype=int),
            "top_cities": pd.Series(dtype=int),
            "visits_per_hour": pd.Series(dtype=int),
        }
    return {
        "total_visits": len(df),
        "unique_visitors": df["ip_hash"].nunique(),
        "countries": df["country"].nunique(),
        "cities": df["city"].nunique(),
        "top_countries": df["country"].value_counts().head(10),
        "top_cities": df["city"].value_counts().head(10),
        "visits_per_hour": (
            df.set_index("timestamp")
              .resample("h")
              .size()
              .rename("visits")
        ),
    }


def create_visitor_map(df: pd.DataFrame) -> "folium.Map":
    """Create a Folium map showing visitor locations."""
    import folium
    from folium.plugins import MarkerCluster

    m = folium.Map(
        location=[20, 0],
        zoom_start=2,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
    )

    if df.empty:
        return m

    # Aggregate visits per location
    locs = (
        df.groupby(["lat", "lon", "city", "country"])
          .agg(visits=("ip_hash", "size"), unique=("ip_hash", "nunique"))
          .reset_index()
    )
    locs = locs[(locs["lat"] != 0) | (locs["lon"] != 0)]  # drop unknowns

    cluster = MarkerCluster(name="Visitors").add_to(m)

    for _, row in locs.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=max(5, min(25, row["visits"])),
            color="#3498db",
            fill=True,
            fill_color="#2ecc71",
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{row['city']}, {row['country']}</b><br>"
                f"Visits: {row['visits']}<br>"
                f"Unique: {row['unique']}",
                max_width=200,
            ),
            tooltip=f"{row['city']} — {row['visits']} visits",
        ).add_to(cluster)

    # Layer control
    folium.LayerControl().add_to(m)
    return m
