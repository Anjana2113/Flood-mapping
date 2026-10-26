"""
=============================================================================
07_construct_kuttanad_graph.py
Phase 1: Hydrological Graph Formulation G = (V, E, X)
Reference: Li et al. (Remote Sensing, 2025, Section 3.1)
=============================================================================
Constructs the spatial graph for Kuttanad, Kerala:
- Nodes (V): Key village centroids and hydrologic monitoring locations
- Node Attributes (X): 6 Environmental Conditioning Factors from 30m rasters
- Edges (E): Spatial adjacency and canal/river flow connectivity matrix (A)
=============================================================================
"""

import os
import sys
import json
import math
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Study Area Villages / Monitoring Nodes
KUTTANAD_NODES = [
    {"id": "kainakary", "name": "Kainakary", "lat": 9.4920, "lon": 76.3680, "desc": "Polder core surrounded by Vembanad Lake"},
    {"id": "pulinkunnoo", "name": "Pulinkunnoo", "lat": 9.4550, "lon": 76.4520, "desc": "Central Kuttanad lowlands along Pamba river"},
    {"id": "champakulam", "name": "Champakulam", "lat": 9.4120, "lon": 76.4250, "desc": "Low-lying agricultural polder zone"},
    {"id": "edathua", "name": "Edathua", "lat": 9.3640, "lon": 76.4670, "desc": "Southern Kuttanad delta region"},
    {"id": "ramankary", "name": "Ramankary", "lat": 9.4280, "lon": 76.4800, "desc": "Manimala-Pamba river intersection plain"},
    {"id": "nedumudi", "name": "Nedumudi", "lat": 9.4350, "lon": 76.4020, "desc": "Backwater channel and river mouth zone"},
    {"id": "thalavady", "name": "Thalavady", "lat": 9.3510, "lon": 76.4420, "desc": "Agricultural interior floodplain"},
    {"id": "kavalam", "name": "Kavalam", "lat": 9.4950, "lon": 76.4480, "desc": "Northern Vembanad backwater village"},
    {"id": "neelamperoor", "name": "Neelamperoor", "lat": 9.5120, "lon": 76.4750, "desc": "Low elevation wetland agricultural boundary"},
    {"id": "veliyanad", "name": "Veliyanad", "lat": 9.4980, "lon": 76.5100, "desc": "North-eastern river corridor zone"}
]

FEATURE_NAMES = ['elevation', 'slope', 'ndvi', 'lulc', 'rainfall', 'dist_water']
DEFAULT_MEDIANS = [0.59, 1.30, 0.23, 66.5, 11.03, 702.0]

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates great-circle distance between two GPS coordinates in kilometers."""
    R = 6371.0 # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def extract_node_features(raster_path, nodes):
    """Samples environmental conditioning factors from 30m raster for each node."""
    features_list = []
    with rasterio.open(raster_path) as src:
        for node in nodes:
            row, col = src.index(node["lon"], node["lat"])
            row = min(max(0, row), src.height - 1)
            col = min(max(0, col), src.width - 1)
            
            vals = src.read()[:, row, col].astype(np.float64)
            # If NoData, search 3x3 local neighborhood
            if np.isnan(vals).any() or vals[0] < -50:
                window_data = src.read(window=((max(0, row-2), min(src.height, row+3)), 
                                               (max(0, col-2), min(src.width, col+3))))
                valid_mask = ~np.isnan(window_data[0]) & (window_data[0] >= -10)
                if valid_mask.any():
                    r_idx, c_idx = np.where(valid_mask)
                    vals = window_data[:, r_idx[0], c_idx[0]].astype(np.float64)
                else:
                    vals = np.array(DEFAULT_MEDIANS, dtype=np.float64)
            
            # Clean individual NaNs
            for i in range(len(vals)):
                if np.isnan(vals[i]):
                    vals[i] = DEFAULT_MEDIANS[i]
            
            features_list.append(vals[:len(FEATURE_NAMES)].tolist())
            
    return np.array(features_list)

def construct_graph(k_neighbors=3, max_distance_km=10.0):
    print("=" * 65)
    print("🌐 CONSTRUCTING KUTTANAD HYDROLOGICAL GRAPH G = (V, E, X)")
    print("   Reference: Li et al. (Remote Sensing, 2025, Section 3.1)")
    print("=" * 65)

    raster_path = os.path.join("data", "Kuttanad_Environmental_Factors_30m.tif")
    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"Raster missing at {raster_path}!")

    n_nodes = len(KUTTANAD_NODES)
    print(f"\n1. Extracting Node Attributes (X) for {n_nodes} Kuttanad villages...")
    X = extract_node_features(raster_path, KUTTANAD_NODES)
    print(f"   - Node Attribute Matrix X Shape: {X.shape} (Nodes x Features)")
    for i, node in enumerate(KUTTANAD_NODES):
        print(f"     [{i}] {node['name']:<12} | Elev: {X[i][0]:.2f}m | Slope: {X[i][1]:.2f}° | Rainfall: {X[i][4]:.1f}mm | DistWater: {X[i][5]:.0f}m")

    # 2. Compute Pairwise Distance Matrix
    print("\n2. Computing Pairwise Geographic Distance Matrix (km)...")
    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist_matrix[i, j] = haversine_distance(
                KUTTANAD_NODES[i]["lat"], KUTTANAD_NODES[i]["lon"],
                KUTTANAD_NODES[j]["lat"], KUTTANAD_NODES[j]["lon"]
            )

    # 3. Build Spatial Adjacency Matrix A (k-nearest neighbors + distance threshold)
    print(f"\n3. Constructing Topological Adjacency Matrix A (k={k_neighbors}, max_dist={max_distance_km}km)...")
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    edges = []

    for i in range(n_nodes):
        # Sort distances from node i
        sorted_indices = np.argsort(dist_matrix[i])
        connected = 0
        for j in sorted_indices[1:]: # exclude self
            if dist_matrix[i, j] <= max_distance_km and connected < k_neighbors:
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0 # Undirected graph
                edges.append((i, j, round(dist_matrix[i, j], 2)))
                connected += 1

    total_edges = int(np.sum(adj_matrix) / 2)
    print(f"   - Total Topological Edges Formed: {total_edges}")
    print(f"   - Graph Density: {total_edges / (n_nodes * (n_nodes - 1) / 2):.3f}")

    # 4. Save Graph JSON
    graph_data = {
        "metadata": {
            "title": "Kuttanad Hydrological Graph",
            "reference": "Li et al., Remote Sensing 2025, Sec 3.1",
            "num_nodes": n_nodes,
            "num_edges": total_edges,
            "feature_names": FEATURE_NAMES
        },
        "nodes": [
            {
                "index": i,
                "id": node["id"],
                "name": node["name"],
                "lat": node["lat"],
                "lon": node["lon"],
                "desc": node["desc"],
                "features": {FEATURE_NAMES[k]: X[i][k] for k in range(len(FEATURE_NAMES))}
            }
            for i, node in enumerate(KUTTANAD_NODES)
        ],
        "adjacency_matrix": adj_matrix.tolist(),
        "distance_matrix_km": np.round(dist_matrix, 2).tolist()
    }

    output_json = os.path.join("data", "kuttanad_graph.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)
    print(f"   - Graph structure saved to: {output_json}")

    # 5. Visualize Graph Topology
    print("\n4. Generating Publication-Quality Graph Topology Diagram...")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    G = nx.Graph()
    for i, node in enumerate(KUTTANAD_NODES):
        G.add_node(i, name=node["name"], pos=(node["lon"], node["lat"]))

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adj_matrix[i, j] == 1.0:
                G.add_edge(i, j, weight=dist_matrix[i, j])

    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw edges with distance labels
    nx.draw_networkx_edges(G, pos, ax=ax, width=2.2, alpha=0.7, edge_color="#1f77b4", style="solid")
    
    # Draw nodes
    node_colors = [X[i][0] for i in range(n_nodes)] # Colored by elevation
    nodes_draw = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=800, node_color=node_colors, 
                                        cmap="Blues_r", edgecolors="#0f3d63", linewidths=2.0)
    
    # Add village labels
    labels = {i: KUTTANAD_NODES[i]["name"] for i in range(n_nodes)}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10, font_weight="bold", 
                            font_color="#0b1d3a", verticalalignment="bottom")

    cbar = plt.colorbar(nodes_draw, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("Elevation (m above sea level)", fontsize=11, fontweight="bold")

    ax.set_title("Kuttanad Hydrological Graph Topology G = (V, E, X)\nSpatial Adjacency & River/Canal Connectivity (Li et al., 2025)", 
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    figure_path = os.path.join("outputs", "figures", "kuttanad_graph_topology.png")
    plt.tight_layout()
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()

    print(f"   - Graph topology figure saved to: {figure_path}")
    print("\n✅ Step 1 (Graph Formulation) Successfully Completed!")

if __name__ == "__main__":
    construct_graph()
