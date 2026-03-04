# =============================================================================
# Interactive Map Visualization Module
# Uses Folium for creating interactive maps with weather and energy data layers
# =============================================================================

import folium
from folium import plugins
import json
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from datetime import datetime

from config import REGIONS, COLORS


# =============================================================================
# Morocco unified boundary (simplified polygon including southern provinces)
# This ensures Morocco is displayed as a single, whole territory.
# =============================================================================
MOROCCO_BOUNDARY_GEOJSON = {
    "type": "Feature",
    "properties": {
        "name": "Morocco",
        "name_ar": "المملكة المغربية",
        "sovereign": "Morocco"
    },
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            # Atlantic coast — north to south (full territory)
            [-5.40, 35.95],   # Tangier area
            [-5.80, 35.80],
            [-6.30, 35.20],   # Atlantic coast north
            [-6.80, 34.05],   # Rabat area
            [-7.60, 33.60],   # Casablanca
            [-8.50, 33.30],
            [-9.00, 32.60],   # El Jadida
            [-9.80, 32.20],
            [-9.85, 31.50],   # Safi
            [-9.80, 30.70],   # Agadir
            [-10.00, 30.00],
            [-10.05, 29.40],  # Guelmim
            [-10.50, 29.30],
            [-11.40, 28.50],  # Tan-Tan
            [-12.00, 27.80],
            [-12.50, 27.20],  # Laayoune area
            [-13.20, 26.50],
            [-13.50, 26.10],
            [-14.00, 25.50],
            [-14.50, 24.80],
            [-14.80, 24.20],
            [-15.20, 23.80],
            [-15.90, 23.50],  # Dakhla area
            [-16.00, 22.80],
            [-16.10, 22.30],
            [-16.00, 21.80],
            [-16.50, 21.30],
            [-17.05, 21.00],  # Southern tip (Cap Blanc / Nouadhibou)
            # Southern border — west to east (Morocco–Mauritania)
            [-16.50, 21.33],
            [-14.50, 21.33],
            [-13.17, 21.33],
            [-13.00, 21.33],
            # Southeastern border — diagonal to Algeria junction
            [-12.20, 22.00],
            [-11.40, 23.00],
            [-10.50, 24.20],
            [-9.80, 25.50],
            [-9.20, 26.40],
            # Eastern border — south to north (Morocco–Algeria)
            [-8.67, 27.29],   # Algeria border area
            [-8.67, 27.67],
            [-5.00, 29.00],
            [-3.60, 29.40],
            [-2.20, 29.60],
            [-1.75, 30.30],
            [-1.50, 31.50],
            [-1.30, 32.10],   # Eastern Morocco
            [-1.15, 32.60],
            [-1.20, 33.10],
            [-1.70, 33.50],
            [-1.80, 34.10],
            [-1.85, 34.60],   # Oujda area
            [-2.10, 34.90],
            [-2.20, 35.10],
            # Mediterranean coast — east to west
            [-2.40, 35.15],
            [-2.95, 35.30],   # Melilla area
            [-3.40, 35.20],
            [-3.80, 35.25],
            [-4.30, 35.20],
            [-4.60, 35.25],   # Al Hoceima
            [-5.00, 35.50],
            [-5.30, 35.85],
            [-5.40, 35.95],   # Close polygon at Tangier
        ]]
    }
}


def add_morocco_boundary(m: folium.Map, highlight: bool = True) -> folium.Map:
    """No-op — kept for backward compatibility.

    We no longer draw any overlay.  Instead we use tile providers
    that do not render disputed internal borders (Esri Dark Gray
    Canvas, Esri World Street Map, Esri Satellite).
    """
    return m


def create_base_map(
    center_lat: float,
    center_lon: float,
    zoom: int = 10,
    tiles: str = "OpenStreetMap"
) -> folium.Map:
    """
    Create a base Folium map.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        zoom: Initial zoom level
        tiles: Map tile layer
    
    Returns:
        Folium Map object
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=tiles,
        control_scale=True,
    )
    
    # Add additional tile layers
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Street Map",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Dark",
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m


def add_location_marker(
    m: folium.Map,
    lat: float,
    lon: float,
    name: str,
    weather_data: Optional[Dict[str, Any]] = None,
    air_quality_data: Optional[Dict[str, Any]] = None,
    solar_data: Optional[Dict[str, Any]] = None,
) -> folium.Map:
    """
    Add a location marker with weather popup.
    
    Args:
        m: Folium Map object
        lat: Latitude
        lon: Longitude
        name: Location name
        weather_data: Processed weather data
        air_quality_data: Processed air quality data
        solar_data: Processed solar data
    
    Returns:
        Updated Folium Map
    """
    # Build popup content
    popup_html = f"""
    <div style="font-family: Arial, sans-serif; width: 300px;">
        <h3 style="color: #2c3e50; margin-bottom: 10px;">{name}</h3>
        <p style="color: #7f8c8d; font-size: 12px;">📍 {lat:.4f}, {lon:.4f}</p>
    """
    
    # Add weather info if available
    if weather_data and "current" in weather_data:
        current = weather_data["current"]
        popup_html += f"""
        <hr style="margin: 10px 0;">
        <h4 style="color: #3498db;">🌤️ Current Weather</h4>
        <table style="width: 100%; font-size: 13px;">
            <tr>
                <td><strong>Temperature:</strong></td>
                <td>{current.get('temperature', 'N/A')}°C (feels like {current.get('feels_like', 'N/A')}°C)</td>
            </tr>
            <tr>
                <td><strong>Conditions:</strong></td>
                <td>{current.get('weather_icon', '')} {current.get('weather_description', 'N/A')}</td>
            </tr>
            <tr>
                <td><strong>Wind:</strong></td>
                <td>💨 {current.get('wind_speed', 'N/A')} km/h {current.get('wind_direction_name', '')}</td>
            </tr>
            <tr>
                <td><strong>Humidity:</strong></td>
                <td>💧 {current.get('humidity', 'N/A')}%</td>
            </tr>
            <tr>
                <td><strong>Pressure:</strong></td>
                <td>📊 {current.get('pressure', 'N/A')} hPa</td>
            </tr>
        </table>
        """
    
    # Add air quality info if available
    if air_quality_data and "current" in air_quality_data:
        aq = air_quality_data["current"]
        aqi = aq.get('us_aqi') or aq.get('european_aqi', 'N/A')
        aqi_color = aq.get('aqi_color', '#808080')
        popup_html += f"""
        <hr style="margin: 10px 0;">
        <h4 style="color: #27ae60;">🌬️ Air Quality</h4>
        <p><strong>AQI:</strong> 
           <span style="background-color: {aqi_color}; padding: 2px 8px; border-radius: 3px; color: white;">
               {aqi}
           </span>
           - {aq.get('aqi_category', 'N/A')}
        </p>
        <p style="font-size: 11px; color: #7f8c8d;">{aq.get('health_message', '')}</p>
        """
    
    popup_html += "</div>"
    
    # Create marker with popup
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=350),
        tooltip=f"{name} - Click for details",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)
    
    return m


def add_all_regions_markers(m: folium.Map) -> folium.Map:
    """Add markers for all predefined regions."""
    for name, coords in REGIONS.items():
        folium.Marker(
            location=[coords["lat"], coords["lon"]],
            popup=f"<b>{name}</b><br>{coords['country']}",
            tooltip=name,
            icon=folium.Icon(color="green", icon="leaf"),
        ).add_to(m)
    return m


def create_wind_rose_data(hourly_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare wind data for wind rose visualization.
    
    Args:
        hourly_data: Hourly weather dataframe
    
    Returns:
        Dictionary with wind rose data
    """
    if hourly_data is None or hourly_data.empty:
        return {}
    
    # Define direction bins
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    direction_bins = list(range(0, 361, 22))
    
    # Bin wind directions
    hourly_data["direction_bin"] = pd.cut(
        hourly_data["wind_direction"],
        bins=direction_bins,
        labels=directions,
        include_lowest=True
    )
    
    # Calculate frequency and average speed by direction
    wind_rose = hourly_data.groupby("direction_bin").agg({
        "wind_speed": ["mean", "max", "count"]
    }).reset_index()
    
    wind_rose.columns = ["direction", "mean_speed", "max_speed", "count"]
    wind_rose["frequency"] = wind_rose["count"] / wind_rose["count"].sum() * 100
    
    return {
        "data": wind_rose.to_dict("records"),
        "dominant_direction": wind_rose.loc[wind_rose["frequency"].idxmax(), "direction"],
        "average_speed": hourly_data["wind_speed"].mean(),
        "max_speed": hourly_data["wind_speed"].max(),
    }


def add_wind_overlay(
    m: folium.Map,
    lat: float,
    lon: float,
    wind_speed: float,
    wind_direction: float,
    radius: int = 5000,
) -> folium.Map:
    """
    Add a wind indicator overlay to the map.
    
    Args:
        m: Folium Map
        lat: Latitude
        lon: Longitude
        wind_speed: Wind speed in km/h
        wind_direction: Wind direction in degrees
        radius: Display radius in meters
    """
    # Color based on wind speed
    if wind_speed < 20:
        color = "#3498db"  # Light blue - calm
    elif wind_speed < 40:
        color = "#2ecc71"  # Green - moderate
    elif wind_speed < 60:
        color = "#f39c12"  # Orange - strong
    else:
        color = "#e74c3c"  # Red - dangerous
    
    # Add wind circle
    folium.Circle(
        location=[lat, lon],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.2,
        popup=f"Wind: {wind_speed:.0f} km/h from {wind_direction:.0f}°",
        tooltip=f"💨 {wind_speed:.0f} km/h",
    ).add_to(m)
    
    return m


def add_heatmap_layer(
    m: folium.Map,
    data_points: List[Tuple[float, float, float]],
    name: str = "Data Heatmap",
    radius: int = 20,
) -> folium.Map:
    """
    Add a heatmap layer to the map.
    
    Args:
        m: Folium Map
        data_points: List of (lat, lon, intensity) tuples
        name: Layer name
        radius: Heatmap point radius
    """
    heat_data = [[point[0], point[1], point[2]] for point in data_points]
    
    plugins.HeatMap(
        heat_data,
        name=name,
        radius=radius,
        blur=15,
        max_zoom=13,
    ).add_to(m)
    
    return m


def add_solar_overlay(
    m: folium.Map,
    lat: float,
    lon: float,
    ghi_value: float,
    radius: int = 3000,
) -> folium.Map:
    """
    Add a solar irradiance indicator to the map.
    
    Args:
        m: Folium Map
        lat: Latitude
        lon: Longitude
        ghi_value: Global Horizontal Irradiance (W/m²)
        radius: Display radius
    """
    # Color based on solar irradiance
    if ghi_value < 200:
        color = "#95a5a6"  # Gray - low
    elif ghi_value < 500:
        color = "#f1c40f"  # Yellow - moderate
    elif ghi_value < 800:
        color = "#e67e22"  # Orange - good
    else:
        color = "#e74c3c"  # Red - excellent
    
    folium.Circle(
        location=[lat, lon],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.3,
        popup=f"Solar Irradiance: {ghi_value:.0f} W/m²",
        tooltip=f"☀️ {ghi_value:.0f} W/m²",
    ).add_to(m)
    
    return m


def add_aqi_overlay(
    m: folium.Map,
    lat: float,
    lon: float,
    aqi: float,
    category: str,
) -> folium.Map:
    """Add an air quality indicator to the map."""
    from modules.air_quality import get_aqi_color
    
    color = get_aqi_color(aqi)
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=15,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=f"AQI: {aqi:.0f} - {category}",
        tooltip=f"AQI: {aqi:.0f}",
    ).add_to(m)
    
    return m


def create_comprehensive_map(
    lat: float,
    lon: float,
    name: str,
    weather_data: Optional[Dict[str, Any]] = None,
    air_quality_data: Optional[Dict[str, Any]] = None,
    solar_data: Optional[Dict[str, Any]] = None,
    show_all_regions: bool = False,
) -> folium.Map:
    """
    Create a comprehensive map with all available data layers.
    
    Args:
        lat: Center latitude
        lon: Center longitude
        name: Location name
        weather_data: Processed weather data
        air_quality_data: Processed air quality data
        solar_data: Processed solar data
        show_all_regions: Whether to show all predefined regions
    
    Returns:
        Folium Map with all layers
    """
    # Create base map
    m = create_base_map(lat, lon, zoom=11)
    
    # Add Morocco unified boundary overlay
    add_morocco_boundary(m)
    
    # Add mini map
    plugins.MiniMap(toggle_display=True).add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Add location marker with data popup
    add_location_marker(m, lat, lon, name, weather_data, air_quality_data, solar_data)
    
    # Add wind overlay if available
    if weather_data and "current" in weather_data:
        current = weather_data["current"]
        wind_speed = current.get("wind_speed", 0)
        wind_direction = current.get("wind_direction", 0)
        add_wind_overlay(m, lat, lon, wind_speed, wind_direction)
    
    # Add AQI overlay if available
    if air_quality_data and "current" in air_quality_data:
        aq = air_quality_data["current"]
        aqi = aq.get("us_aqi") or aq.get("european_aqi", 0)
        category = aq.get("aqi_category", "Unknown")
        # Offset slightly so it doesn't overlap with main marker
        add_aqi_overlay(m, lat + 0.01, lon + 0.01, aqi, category)
    
    # Show all predefined regions if requested
    if show_all_regions:
        add_all_regions_markers(m)
    
    # Add draw control for custom measurements
    plugins.Draw(export=True).add_to(m)
    
    # Add mouse position display
    plugins.MousePosition().add_to(m)
    
    # Add measure control
    plugins.MeasureControl(position="topleft").add_to(m)
    
    return m


def create_regions_overview_map() -> folium.Map:
    """Create an overview map showing all predefined regions."""
    # Calculate center of all regions
    avg_lat = sum(r["lat"] for r in REGIONS.values()) / len(REGIONS)
    avg_lon = sum(r["lon"] for r in REGIONS.values()) / len(REGIONS)
    
    m = create_base_map(avg_lat, avg_lon, zoom=4)
    
    # Add Morocco unified boundary overlay
    add_morocco_boundary(m, highlight=True)
    
    # Add markers for all regions
    for name, coords in REGIONS.items():
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4>{name}</h4>
            <p><strong>Country:</strong> {coords['country']}</p>
            <p><strong>Coordinates:</strong><br>
               {coords['lat']:.4f}°, {coords['lon']:.4f}°</p>
            <p><strong>Timezone:</strong> {coords['timezone']}</p>
        </div>
        """
        
        folium.Marker(
            location=[coords["lat"], coords["lon"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"📍 {name}",
            icon=folium.Icon(
                color="red" if "Morocco" in coords["country"] else "blue",
                icon="info-sign"
            ),
        ).add_to(m)
    
    # Connect regions with a polyline (for visualization)
    region_coords = [[r["lat"], r["lon"]] for r in REGIONS.values()]
    
    # Fit bounds to show all markers
    m.fit_bounds([[min(c[0] for c in region_coords), min(c[1] for c in region_coords)],
                  [max(c[0] for c in region_coords), max(c[1] for c in region_coords)]])
    
    plugins.MiniMap().add_to(m)
    plugins.Fullscreen().add_to(m)
    
    return m


def get_map_html(m: folium.Map) -> str:
    """Get HTML representation of a Folium map for embedding."""
    return m._repr_html_()


# =============================================================================
# 50 km Town Map  (zoomable local area)
# =============================================================================

def create_town_map(
    lat: float,
    lon: float,
    name: str = "",
    size_km: float = 50.0,
    weather_data: Optional[Dict[str, Any]] = None,
) -> folium.Map:
    """
    Create a 50 km × 50 km map centred on the chosen coordinates.

    The map bounds are computed from the geographic distance so the
    viewport is always a square of ``size_km`` km.  Scroll-zoom and
    drag are enabled so the user can manually zoom into streets.

    Parameters
    ----------
    lat, lon : centre point
    name     : location label
    size_km  : side length of the square viewport (default 50 km)
    weather_data : optional weather dict to annotate the marker
    """
    import math

    half_km = size_km / 2.0
    # Approximate ° per km at given latitude
    deg_lat = half_km / 111.32
    deg_lon = half_km / (111.32 * math.cos(math.radians(lat)))

    sw = [lat - deg_lat, lon - deg_lon]
    ne = [lat + deg_lat, lon + deg_lon]

    # Compute zoom level so the viewport spans ~size_km.
    # At zoom z each tile covers circ/2^z km.  The viewport shows ~2 tiles,
    # so we want  2 · circ/2^z ≈ size_km  →  z ≈ log2(2·circ/size_km).
    circ_km = 40075 * math.cos(math.radians(lat))
    zoom = int(round(math.log2(2.0 * circ_km / size_km)))
    zoom = max(9, min(zoom, 16))  # clamp to sensible range

    m = folium.Map(
        location=[lat, lon],
        zoom_start=zoom,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        control_scale=True,
        zoom_control=True,
        max_bounds=True,
    )

    # Additional tile layers
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Dark",
    ).add_to(m)

    # Centre marker
    popup_parts = [f"<b>{name}</b><br>({lat:.4f}°, {lon:.4f}°)"]
    if weather_data and "hourly" in weather_data:
        h = weather_data["hourly"]
        if isinstance(h, pd.DataFrame) and not h.empty:
            if "temperature" in h.columns:
                popup_parts.append(
                    f"Temp: {h['temperature'].iloc[-1]:.1f} °C")
            if "wind_speed" in h.columns:
                popup_parts.append(
                    f"Wind: {h['wind_speed'].iloc[-1]:.1f} km/h")
    popup_html = "<br>".join(popup_parts)

    folium.Marker(
        [lat, lon],
        popup=folium.Popup(popup_html, max_width=280),
        tooltip=f"📍 {name}",
        icon=folium.Icon(color="red", icon="star"),
    ).add_to(m)

    # 50 km bounding rectangle (visual cue)
    folium.Rectangle(
        bounds=[sw, ne],
        color="#e74c3c",
        weight=2,
        fill=False,
        dash_array="6",
        tooltip=f"{size_km:.0f} km × {size_km:.0f} km area",
    ).add_to(m)

    folium.LayerControl().add_to(m)
    plugins.MiniMap(toggle_display=True).add_to(m)
    plugins.Fullscreen().add_to(m)

    return m


# =============================================================================
# GWA / GSA–style Resource Maps  (wind speed & solar irradiance heatmaps)
# =============================================================================

import numpy as np


def _generate_resource_grid(
    lat: float,
    lon: float,
    grid_km: float = 50.0,
    resolution_pts: int = 30,
    base_value: float = 5.0,
    noise_std: float = 1.0,
    seed: int = 42,
) -> List[List]:
    """
    Generate a synthetic grid of resource values around (lat, lon).

    In a production system this would fetch ERA5 or MERRA-2 reanalysis
    data; here we create a plausible spatial field by combining:
    - a linear gradient (terrain / coastal effect)
    - a Gaussian hill centred near the point of interest
    - spatially-correlated noise (smoothed with a simple kernel)
    """
    import math
    rng = np.random.default_rng(seed)

    half_km = grid_km / 2.0
    dlat = half_km / 111.32
    dlon = half_km / (111.32 * math.cos(math.radians(lat)))

    lats = np.linspace(lat - dlat, lat + dlat, resolution_pts)
    lons = np.linspace(lon - dlon, lon + dlon, resolution_pts)

    # Gradient (simulating coastal / elevation effect)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    gradient = 0.3 * (lon_grid - lon) / dlon  # east–west gradient

    # Gaussian bump near centre
    r2 = ((lat_grid - lat) / dlat) ** 2 + ((lon_grid - lon) / dlon) ** 2
    bump = 0.8 * np.exp(-2.0 * r2)

    # Spatially correlated noise (simple box smoothing of white noise)
    raw = rng.normal(0, noise_std, (resolution_pts, resolution_pts))
    kernel_size = max(3, resolution_pts // 8)
    from scipy.ndimage import uniform_filter
    smooth_noise = uniform_filter(raw, size=kernel_size)

    field = base_value + gradient + bump + smooth_noise
    field = np.clip(field, 0, None)

    # Return as list of [lat, lon, value] for HeatMap
    points = []
    for i in range(resolution_pts):
        for j in range(resolution_pts):
            points.append([float(lats[i]), float(lons[j]), float(field[i, j])])
    return points


def create_wind_resource_map(
    lat: float,
    lon: float,
    name: str = "",
    weather_data: Optional[Dict[str, Any]] = None,
    grid_km: float = 50.0,
) -> folium.Map:
    """
    GWA-style wind speed resource heatmap (50 km square).

    Colour ramp: blue (low) → green → yellow → red (high), resembling
    the Global Wind Atlas mean wind speed map.
    """
    m = create_town_map(lat, lon, name, size_km=grid_km, weather_data=weather_data)

    # Determine representative mean wind speed from data
    base_ws = 5.0
    if weather_data and "hourly" in weather_data:
        h = weather_data["hourly"]
        if isinstance(h, pd.DataFrame) and "wind_speed" in h.columns:
            base_ws = float(h["wind_speed"].mean())

    points = _generate_resource_grid(
        lat, lon, grid_km=grid_km, base_value=base_ws,
        noise_std=base_ws * 0.15, seed=42,
    )

    from folium.plugins import HeatMap
    HeatMap(
        points,
        name="Wind Speed (m/s)",
        min_opacity=0.35,
        max_zoom=13,
        radius=18,
        blur=22,
        gradient={
            "0.0": "#3288bd",  # blue
            "0.3": "#66c2a5",  # green
            "0.5": "#fee08b",  # yellow
            "0.7": "#fdae61",  # orange
            "1.0": "#d53e4f",  # red
        },
    ).add_to(m)

    # Legend
    _add_resource_legend(m, "Mean Wind Speed",
                         ["< 3 m/s", "3–5", "5–7", "7–9", "> 9"],
                         ["#3288bd", "#66c2a5", "#fee08b", "#fdae61", "#d53e4f"])
    return m


def create_solar_resource_map(
    lat: float,
    lon: float,
    name: str = "",
    solar_data: Optional[Dict[str, Any]] = None,
    grid_km: float = 50.0,
) -> folium.Map:
    """
    GSA-style solar irradiance resource heatmap (50 km square).

    Colour ramp: purple (low) → blue → yellow → red (high), resembling
    the Global Solar Atlas GHI layer.
    """
    m = create_town_map(lat, lon, name, size_km=grid_km)

    base_ghi = 5.0  # kWh/m²/day
    if solar_data and "hourly" in solar_data:
        h = solar_data["hourly"]
        if isinstance(h, pd.DataFrame) and "shortwave_radiation" in h.columns:
            daily_mean = h["shortwave_radiation"].mean() * 24 / 1000  # W → kWh/day
            if daily_mean > 0:
                base_ghi = float(daily_mean)

    points = _generate_resource_grid(
        lat, lon, grid_km=grid_km, base_value=base_ghi,
        noise_std=base_ghi * 0.10, seed=99,
    )

    from folium.plugins import HeatMap
    HeatMap(
        points,
        name="Solar GHI (kWh/m²/day)",
        min_opacity=0.35,
        max_zoom=13,
        radius=18,
        blur=22,
        gradient={
            "0.0": "#4a1486",  # purple
            "0.25": "#2b83ba", # blue
            "0.5": "#ffffbf",  # yellow
            "0.75": "#fdae61", # orange
            "1.0": "#d7191c",  # red
        },
    ).add_to(m)

    _add_resource_legend(m, "Global Horizontal Irradiance",
                         ["< 3 kWh", "3–4.5", "4.5–5.5", "5.5–6.5", "> 6.5"],
                         ["#4a1486", "#2b83ba", "#ffffbf", "#fdae61", "#d7191c"])
    return m


def _add_resource_legend(
    m: folium.Map,
    title: str,
    labels: List[str],
    colors: List[str],
) -> None:
    """Inject an HTML legend into the map (bottom-right corner)."""
    rows = "".join(
        f'<li><span style="background:{c};width:18px;height:12px;'
        f'display:inline-block;margin-right:5px;border:1px solid #555;">'
        f'</span>{lab}</li>'
        for c, lab in zip(colors, labels)
    )
    html = f"""
    <div style="
        position:fixed; bottom:30px; right:30px; z-index:1000;
        background:rgba(0,0,0,0.75); padding:10px 14px; border-radius:6px;
        font-size:12px; color:#eee; line-height:1.6;
    ">
        <b>{title}</b>
        <ul style="list-style:none;padding:4px 0 0 0;margin:0;">{rows}</ul>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))
