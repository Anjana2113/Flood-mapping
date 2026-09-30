import os
import sys
import rasterio
import numpy as np
import folium
from folium import plugins

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def build_combined_interactive_web_map():
    print("=" * 60)
    print("🌐 COMBINING BONUS 1 & BONUS 2 INTO INTERACTIVE WEB MAP")
    print("=" * 60)

    susc_tif = os.path.join("outputs", "maps", "Kuttanad_Flood_Susceptibility_Map.tif")
    risk_tif = os.path.join("outputs", "maps", "Kuttanad_Flood_Risk_Map.tif")

    if not os.path.exists(susc_tif) or not os.path.exists(risk_tif):
        raise FileNotFoundError("Missing susceptibility or risk raster! Run 04 and 06 first.")

    # 1. Read Susceptibility Raster
    print(f"\n1. Reading Susceptibility Raster: {susc_tif}")
    with rasterio.open(susc_tif) as src_susc:
        susc_data = src_susc.read(1)

    # 2. Read Risk Raster
    print(f"2. Reading Flood Risk Raster: {risk_tif}")
    with rasterio.open(risk_tif) as src_risk:
        risk_data = src_risk.read(1)

    # Bounding Box for Kuttanad [south, west, north, east]
    south, west, north, east = 9.15, 76.25, 9.75, 76.65

    # 3. Create Base Folium Map
    print("\n3. Building dual-layer Folium web map...")
    m = folium.Map(
        location=[9.45, 76.43],
        zoom_start=11,
        tiles=None
    )

    # Add Reliable Unblocked Basemaps
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://carto.com/">CARTO</a>',
        name='CartoDB Street Map (Clean)',
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google Maps',
        name='Google Satellite Hybrid',
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite Imagery',
        overlay=False,
        control=True
    ).add_to(m)

    # --- Susceptibility Color Overlay ---
    susc_colors = {
        0: [0, 0, 0, 0],         # Transparent
        1: [43, 131, 186, 160],   # Blue - Low
        2: [171, 221, 164, 180],  # Green - Moderate
        3: [253, 174, 97, 200],   # Orange - High
        4: [215, 25, 28, 220]     # Red - Very High
    }
    susc_rgba = np.zeros((susc_data.shape[0], susc_data.shape[1], 4), dtype=np.uint8)
    for val, col in susc_colors.items():
        susc_rgba[susc_data == val] = col

    folium.raster_layers.ImageOverlay(
        image=susc_rgba,
        bounds=[[south, west], [north, east]],
        opacity=0.75,
        name='🌊 1. Flood Susceptibility Layer (Hazard)',
        show=True
    ).add_to(m)

    # --- Risk Color Overlay ---
    risk_colors = {
        0: [0, 0, 0, 0],         # Transparent
        1: [255, 255, 178, 160], # Light Yellow - Low Risk
        2: [254, 204, 92, 180],  # Yellow - Moderate Risk
        3: [253, 141, 60, 200],  # Orange - High Risk
        4: [189, 0, 38, 230]     # Dark Red - Critical Risk
    }
    risk_rgba = np.zeros((risk_data.shape[0], risk_data.shape[1], 4), dtype=np.uint8)
    for val, col in risk_colors.items():
        risk_rgba[risk_data == val] = col

    folium.raster_layers.ImageOverlay(
        image=risk_rgba,
        bounds=[[south, west], [north, east]],
        opacity=0.80,
        name='🛡️ 2. Flood Risk Layer (Hazard × Exposure)',
        show=False
    ).add_to(m)

    # Interactive Map Tools
    folium.LayerControl(collapsed=False).add_to(m)
    plugins.Fullscreen().add_to(m)
    plugins.MeasureControl(position='topleft', primary_length_unit='kilometers').add_to(m)

    # Dual Legend Box HTML
    legend_html = '''
     <div style="position: fixed; 
                 bottom: 30px; left: 30px; width: 250px; height: 210px; 
                 border:2px solid grey; z-index:9999; font-size:11px;
                 background-color:white; opacity: 0.92; padding: 10px;
                 border-radius: 8px; font-family: sans-serif;">
     <b>🌊 Kuttanad Web Map Legend</b><br><br>
     <b>Susceptibility Levels:</b><br>
     <i style="background:#2b83ba; width:12px; height:12px; float:left; margin-right:6px; opacity:0.8;"></i> Low (<25%)
     <i style="background:#abdda4; width:12px; height:12px; float:left; margin-right:6px; margin-left:8px; opacity:0.8;"></i> Moderate<br>
     <i style="background:#fdae61; width:12px; height:12px; float:left; margin-right:6px; opacity:0.8;"></i> High (50-75%)
     <i style="background:#d7191c; width:12px; height:12px; float:left; margin-right:6px; margin-left:8px; opacity:0.8;"></i> Very High (>75%)<br><br>
     <b>Flood Risk Levels:</b><br>
     <i style="background:#ffffb2; width:12px; height:12px; float:left; margin-right:6px;"></i> Low Risk 
     <i style="background:#fecc5c; width:12px; height:12px; float:left; margin-right:6px; margin-left:8px;"></i> Moderate Risk<br>
     <i style="background:#fd8d3c; width:12px; height:12px; float:left; margin-right:6px;"></i> High Risk 
     <i style="background:#bd0026; width:12px; height:12px; float:left; margin-right:6px; margin-left:8px;"></i> Critical Risk<br>
     </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save Interactive Map HTML
    out_html = os.path.join("outputs", "maps", "kuttanad_interactive_map.html")
    m.save(out_html)
    print(f"\n✅ Updated map with CartoDB & Google Satellite basemaps at: {out_html}")

if __name__ == "__main__":
    build_combined_interactive_web_map()
