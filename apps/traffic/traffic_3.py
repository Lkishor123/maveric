# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# --- MODIFIED TO OUTPUT UE DATA CSV FOR CCO (PER TICK) ---
# --- Includes dummy topology generation and consistent naming ---

import os
import sys
import json
import logging
import math # Added import
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
try:
    from scipy.spatial import Voronoi, voronoi_plot_2d
except ImportError:
    print("Warning: scipy not found. Voronoi calculation and plotting will fail.")
    Voronoi, voronoi_plot_2d = None, None
import matplotlib.pyplot as plt
try:
    from shapely.geometry import Polygon, box, MultiPolygon, Point, LineString
    from shapely.validation import make_valid
except ImportError:
    print("Error: Shapely library not found. Please install it: pip install Shapely")
    sys.exit(1)


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, # Set to DEBUG for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Path setup (USER MUST MODIFY IF NEEDED) ---
RADP_ROOT = os.getenv("MAVERIC_ROOT", "/home/lk/Projects/accelcq-repos/cloudly/github/maveric") # SET YOUR PATH HERE OR VIA ENV VAR
if not os.path.isdir(RADP_ROOT):
     potential_path = os.path.join(os.path.dirname(__file__), "..", "..")
     if os.path.isdir(os.path.join(potential_path, "radp")):
         RADP_ROOT = os.path.abspath(potential_path)
         print(f"Warning: RADP_ROOT not explicitly set or found. Assuming relative path: {RADP_ROOT}")
     else:
        raise FileNotFoundError(f"RADP_ROOT directory not found: {RADP_ROOT}. Please set the correct path or environment variable.")
sys.path.insert(0, RADP_ROOT)


# --- Imports from RADP library (with fallback definitions) ---
try:
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.utils.gis_tools import GISTools
    logger.info("Successfully imported constants and GISTools from RADP library.")
except ImportError as e:
    print(f"Warning: Error importing RADP modules: {e}. Using fallback definitions for constants.")
    class c: # Define fallback constants ONLY IF import fails
        CELL_ID = "cell_id"; CELL_LAT = "cell_lat"; CELL_LON = "cell_lon"
        CELL_AZ_DEG = "cell_az_deg"; CELL_TXPWR_DBM = "cell_txpwr_dbm"
        ECGI = "ecgi"; SITE_ID = "site_id"; CELL_NAME = "cell_name"
        ENODEB_ID = "enodeb_id"; TAC = "tac"; CELL_CARRIER_FREQ_MHZ = "cell_carrier_freq_mhz"
        LAT = "lat"; LON = "lon"
    class GISTools: # Minimal fallback for distance if needed
        @staticmethod
        def dist(coord1, coord2):
             R = 6371.0; lat1, lon1 = map(np.radians, coord1); lat2, lon2 = map(np.radians, coord2)
             dlon = lon2 - lon1; dlat = lat2 - lat1
             a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
             return R * 2 * np.arcsin(np.sqrt(a))




# --- Topology Generation Function ---
def generate_dummy_topology(
    num_sites: int,
    cells_per_site: int = 3,
    lat_range: Tuple[float, float] = (40.7, 40.8),
    lon_range: Tuple[float, float] = (-74.05, -73.95),
    start_ecgi: int = 1001,
    start_enodeb_id: int = 1,
    default_tac: int = 1,
    default_freq: int = 2100,
    default_power_dbm: float = 25.0,
    azimuth_step: int = 120,
) -> pd.DataFrame:
    """Generates a DataFrame with dummy topology data."""
    logger.info(f"Generating dummy topology for {num_sites} sites with {cells_per_site} cells each.")
    topology_data = []
    current_ecgi = start_ecgi
    current_enodeb_id = start_enodeb_id

    # Define column names using constants if available, otherwise fallback strings
    COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')
    COL_CELL_LAT = getattr(c, 'CELL_LAT', 'cell_lat')
    COL_CELL_LON = getattr(c, 'CELL_LON', 'cell_lon')
    COL_CELL_AZ_DEG = getattr(c, 'CELL_AZ_DEG', 'cell_az_deg')
    COL_CELL_TXPWR_DBM = getattr(c, 'CELL_TXPWR_DBM', 'cell_txpwr_dbm')
    COL_ECGI = getattr(c, 'ECGI', 'ecgi')
    COL_SITE_ID = getattr(c, 'SITE_ID', 'site_id')
    COL_CELL_NAME = getattr(c, 'CELL_NAME', 'cell_name')
    COL_ENODEB_ID = getattr(c, 'ENODEB_ID', 'enodeb_id')
    COL_TAC = getattr(c, 'TAC', 'tac')
    COL_CELL_CARRIER_FREQ_MHZ = getattr(c, 'CELL_CARRIER_FREQ_MHZ', 'cell_carrier_freq_mhz')

    for i in range(num_sites):
        site_lat = np.random.uniform(lat_range[0], lat_range[1])
        site_lon = np.random.uniform(lon_range[0], lon_range[1])
        site_id_str = f"Site{i+1}"

        for j in range(cells_per_site):
            cell_az = (j * azimuth_step) % 360
            cell_id_str = f"cell_{current_enodeb_id}_{cell_az}"
            cell_name_str = f"Cell{j+1}"

            row = {
                COL_ECGI: current_ecgi,
                COL_SITE_ID: site_id_str,
                COL_CELL_NAME: cell_name_str,
                COL_ENODEB_ID: current_enodeb_id,
                COL_CELL_AZ_DEG: cell_az,
                COL_TAC: default_tac,
                COL_CELL_LAT: site_lat,
                COL_CELL_LON: site_lon,
                COL_CELL_ID: cell_id_str,
                COL_CELL_CARRIER_FREQ_MHZ: default_freq,
                COL_CELL_TXPWR_DBM: default_power_dbm + np.random.uniform(-2, 2)
            }
            topology_data.append(row)
        current_ecgi += 1
        current_enodeb_id += 1

    df = pd.DataFrame(topology_data)
    column_order = [ # Define the exact desired order
        COL_ECGI, COL_SITE_ID, COL_CELL_NAME, COL_ENODEB_ID, COL_CELL_AZ_DEG, COL_TAC,
        COL_CELL_LAT, COL_CELL_LON, COL_CELL_ID, COL_CELL_CARRIER_FREQ_MHZ,
        COL_CELL_TXPWR_DBM
    ]
    df = df.reindex(columns=column_order) # Ensure order and presence of all columns
    logger.info(f"Generated dummy topology DataFrame with {len(df)} cells.")
    return df

# --- Helper Functions ---
def space_boundary(cell_topology_data: pd.DataFrame, buffer_percent=0.3) -> Dict:
    """Calculates the buffered bounding box."""
    if cell_topology_data.empty: raise ValueError("Cell topology data cannot be empty.")
    min_lat = cell_topology_data[c.CELL_LAT].min(); max_lat = cell_topology_data[c.CELL_LAT].max()
    min_lon = cell_topology_data[c.CELL_LON].min(); max_lon = cell_topology_data[c.CELL_LON].max()
    lat_range = max_lat - min_lat; lon_range = max_lon - min_lon
    lat_buffer = lat_range * buffer_percent if lat_range > 1e-9 else 0.1
    lon_buffer = lon_range * buffer_percent if lon_range > 1e-9 else 0.1
    logger.debug(f"Original Bounds: LON=[{min_lon:.4f}, {max_lon:.4f}], LAT=[{min_lat:.4f}, {max_lat:.4f}]")
    logger.debug(f"Buffer Percent: {buffer_percent*100}%, Lon Buffer: {lon_buffer:.4f}, Lat Buffer: {lat_buffer:.4f}")
    return {
        "min_lon_buffered": max(min_lon - lon_buffer, -180.0),
        "min_lat_buffered": max(min_lat - lat_buffer, -90.0),
        "max_lon_buffered": min(max_lon + lon_buffer, 180.0),
        "max_lat_buffered": min(max_lat + lat_buffer, 90.0),
    }

def _space_component_list_generator(spatial_params: Dict) -> Dict:
    """Generates dict mapping space type names to proportions."""
    space_types = spatial_params.get("types", []); proportions = spatial_params.get("proportions", [])
    if not space_types or not proportions: raise ValueError("Spatial params need 'types' and 'proportions'.")
    if len(space_types) != len(proportions): raise ValueError("Num types and proportions must match.")
    if not np.isclose(sum(proportions), 1.0): raise ValueError("Proportions must sum to 1.0")
    # *** USE TYPE NAMES AS KEYS ***
    space_list = {stype: prop for stype, prop in zip(space_types, proportions)}
    logger.debug(f"Generated space component list: {space_list}")
    return space_list

def _create_spatial_cells(cell_topology_data: pd.DataFrame, space_list: Dict, space_bound: Dict) -> List[Dict]:
    """Creates clipped Voronoi cells with assigned space types (using actual type names)."""
    if cell_topology_data.empty: logger.error("Topology empty."); return []
    logger.info(f"Creating spatial cells from {len(cell_topology_data)} points.")
    if len(cell_topology_data) < 4: logger.error(f"Need >= 4 points for Voronoi, have {len(cell_topology_data)}."); return []
    if Voronoi is None: logger.error("Scipy required for Voronoi."); return []
    points = cell_topology_data[[c.CELL_LON, c.CELL_LAT]].values
    try: vor = Voronoi(points)
    except Exception as e: logger.error(f"Voronoi failed: {e}"); return []
    spatial_cells = []; min_lon, max_lon, min_lat, max_lat = space_bound.get("min_lon_buffered"), space_bound.get("max_lon_buffered"), space_bound.get("min_lat_buffered"), space_bound.get("max_lat_buffered")
    if any(v is None for v in [min_lon, max_lon, min_lat, max_lat]): logger.error("Invalid space boundary dict."); return []
    logger.debug(f"Clipping boundary: LON=[{min_lon:.4f}, {max_lon:.4f}], LAT=[{min_lat:.4f}, {max_lat:.4f}]")
    try: boundary_polygon = box(min_lon, min_lat, max_lon, max_lat); assert boundary_polygon.is_valid
    except Exception as e: logger.error(f"Failed boundary polygon creation: {e}"); return []
    sk_inf, sk_inv, sk_emp, proc = 0, 0, 0, 0
    for i, region_idx_list in enumerate(vor.regions):
        if not region_idx_list or -1 in region_idx_list: sk_inf += 1; continue
        try:
            verts = vor.vertices[region_idx_list]
            if np.isnan(verts).any(): logger.warning(f"Skip region {i}: NaN vertices."); sk_inv += 1; continue
            if len(verts) < 3: logger.warning(f"Skip region {i}: <3 vertices."); sk_inv += 1; continue
            vor_poly = Polygon(verts)
            if not vor_poly.is_valid: vor_poly = make_valid(vor_poly)
            if not vor_poly.is_valid: logger.warning(f"Skip region {i}: Invalid polygon."); sk_inv += 1; continue
            polys_to_use = []
            if vor_poly.geom_type == 'Polygon': polys_to_use.append(vor_poly)
            elif vor_poly.geom_type == 'MultiPolygon': polys_to_use.extend(list(vor_poly.geoms))
            else: logger.warning(f"Skip region {i}: make_valid gave {vor_poly.geom_type}"); sk_inv += 1; continue

            valid_clipped_polys = []
            for p in polys_to_use:
                if p.geom_type != 'Polygon': continue
                clipped = p.intersection(boundary_polygon)
                if not clipped.is_empty and clipped.is_valid:
                    if clipped.geom_type == 'Polygon' and clipped.area > 1e-9: valid_clipped_polys.append(clipped)
                    elif clipped.geom_type == 'MultiPolygon':
                         for sub_poly in clipped.geoms:
                              if sub_poly.geom_type == 'Polygon' and sub_poly.is_valid and sub_poly.area > 1e-9: valid_clipped_polys.append(sub_poly)

            if not valid_clipped_polys: sk_emp += 1; continue

            space_keys = list(space_list.keys()); space_probs = list(space_list.values())
            if not np.isclose(sum(space_probs), 1.0): space_probs = np.array(space_probs) / sum(space_probs)
            assigned_type = np.random.choice(space_keys, p=space_probs) # *** USE ACTUAL TYPE NAME ***

            for final_poly in valid_clipped_polys:
                spatial_cells.append({
                    "bounds": list(final_poly.exterior.coords), "type": assigned_type, # *** STORE ACTUAL TYPE NAME ***
                    "cell_id": len(spatial_cells), "original_voronoi_region": i
                })
                proc += 1
        except IndexError: logger.warning(f"IndexError region {i}."); sk_inv += 1; continue
        except Exception as e: logger.error(f"Error region {i}: {e}."); sk_inv += 1; continue
    logger.info(f"Voronoi processing: Processed OK={proc}, Skip Inf={sk_inf}, Skip Invalid={sk_inv}, Skip EmptyClip={sk_emp}. Total cells={len(spatial_cells)}")
    if not spatial_cells: logger.error("No valid spatial cells created after clipping.")
    return spatial_cells

def _city_digitaltwin_generator(total_ue: int, spatial_params_for_city: Dict, time_params: Dict, cell_topology_data: pd.DataFrame) -> pd.DataFrame:
    """Generates UE locations using actual type names."""
    logger.info(f"Generating UE data for {total_ue} UEs over {time_params.get('total_ticks', 0)} ticks.")
    space_bound = space_boundary(cell_topology_data)
    space_list = _space_component_list_generator(spatial_params_for_city) # Now returns {'residential': 0.5, ...}
    spatial_cells = _create_spatial_cells(cell_topology_data, space_list, space_bound)
    if not spatial_cells: logger.error("No spatial cells; cannot generate UEs."); return pd.DataFrame()
    total_ticks = time_params.get("total_ticks", 1); ue_data = []; time_weights = time_params.get('time_weights', {})
    COL_LAT = getattr(c, 'LAT', 'lat'); COL_LON = getattr(c, 'LON', 'lon') # Use constants if available

    for tick in range(total_ticks):
        tick_weights = []; valid_cells_for_tick = []
        for cell in spatial_cells:
             cell_type = cell['type'] # *** THIS IS NOW 'residential', etc. ***
             # *** Lookup in time_weights uses 'residential', etc. ***
             type_weights = time_weights.get(cell_type, [1.0] * total_ticks)
             if tick < len(type_weights):
                  weight = type_weights[tick];
                  if weight > 0: tick_weights.append(weight); valid_cells_for_tick.append(cell)
        if not tick_weights or np.sum(tick_weights) == 0: logger.warning(f"No valid cells/weights for tick {tick}. Skipping."); continue
        probs = np.array(tick_weights) / np.sum(tick_weights)
        for ue_id in range(total_ue):
            chosen_cell = np.random.choice(valid_cells_for_tick, p=probs)
            try:
                polygon = Polygon(chosen_cell['bounds']); min_lon, min_lat, max_lon, max_lat = polygon.bounds
                attempts = 0; max_attempts = 100
                while attempts < max_attempts:
                    ue_lon = np.random.uniform(min_lon, max_lon); ue_lat = np.random.uniform(min_lat, max_lat)
                    point = Point(ue_lon, ue_lat)
                    if polygon.contains(point): break
                    attempts += 1
                else: logger.warning(f"Point generation failed for cell {chosen_cell['cell_id']}, UE {ue_id}, tick {tick}. Skipping."); continue
                ue_data.append({
                    "tick": tick, "ue_id": ue_id, COL_LAT: ue_lat, COL_LON: ue_lon,
                    "space_type": chosen_cell['type'], # Store actual type name
                    "voronoi_cell_id": chosen_cell['cell_id'],
                })
            except Exception as e: logger.error(f"Error placing UE {ue_id} tick {tick}: {e}"); continue
    if not ue_data: logger.warning("No UE data generated."); return pd.DataFrame()
    logger.info(f"Generated {len(ue_data)} UE data points.")
    return pd.DataFrame(ue_data)

# --- Analysis Functions (Simplified checks, ensure they use constants) ---
def _radp_model_rftwin(trafficload_ue_data: pd.DataFrame, site_config_data: pd.DataFrame, path_loss_exponent: float = 3.5, ref_rx_power: float = -50,) -> pd.DataFrame:
    if trafficload_ue_data.empty or site_config_data.empty: return pd.DataFrame()
    COL_LAT=getattr(c,'LAT','lat'); COL_LON=getattr(c,'LON','lon')
    COL_CELL_LAT=getattr(c,'CELL_LAT','cell_lat'); COL_CELL_LON=getattr(c,'CELL_LON','cell_lon')
    COL_CELL_ID=getattr(c,'CELL_ID','cell_id'); COL_CELL_TXPWR_DBM=getattr(c,'CELL_TXPWR_DBM','cell_txpwr_dbm')
    req_ue_cols = [COL_LAT, COL_LON, "tick", "ue_id"]; req_site_cols = [COL_CELL_LAT, COL_CELL_LON, COL_CELL_ID, COL_CELL_TXPWR_DBM]
    if not all(col in trafficload_ue_data.columns for col in req_ue_cols): logger.error(f"UE data missing cols: {[c for c in req_ue_cols if c not in trafficload_ue_data.columns]}"); return pd.DataFrame()
    if not all(col in site_config_data.columns for col in req_site_cols): logger.error(f"Site config missing cols: {[c for c in req_site_cols if c not in site_config_data.columns]}"); return pd.DataFrame()
    ue_rxpower_data = []
    for _, ue_row in trafficload_ue_data.iterrows():
        for _, cell_row in site_config_data.iterrows():
            try:
                dist_km = GISTools.dist((ue_row[COL_LAT], ue_row[COL_LON]), (cell_row[COL_CELL_LAT], cell_row[COL_CELL_LON]))
                dist_m = dist_km * 1000.0
                rx_power = ref_rx_power - 10 * path_loss_exponent * np.log10(dist_m) if dist_m > 1e-3 else cell_row[COL_CELL_TXPWR_DBM] # Avoid log(0) or tiny dist
                ue_rxpower_data.append({"tick": ue_row["tick"], "ue_id": ue_row["ue_id"], COL_CELL_ID: cell_row[COL_CELL_ID], "rx_power_dbm": rx_power})
            except Exception as e: logger.warning(f"RxPower calc error UE {ue_row['ue_id']}, Cell {cell_row[COL_CELL_ID]}: {e}")
    return pd.DataFrame(ue_rxpower_data)

def _determine_serving_cell(ue_rxpower_data: pd.DataFrame) -> pd.DataFrame:
    COL_CELL_ID=getattr(c,'CELL_ID','cell_id')
    if ue_rxpower_data.empty or 'rx_power_dbm' not in ue_rxpower_data.columns: return pd.DataFrame(columns=["tick", "ue_id", "serving_cell_id"])
    logger.debug("Determining serving cells...")
    try:
        idx = ue_rxpower_data.groupby(["tick", "ue_id"])["rx_power_dbm"].idxmax()
        serving_cell_data = ue_rxpower_data.loc[idx, ["tick", "ue_id", COL_CELL_ID]].copy().rename(columns={COL_CELL_ID: "serving_cell_id"})
        logger.debug(f"Determined {len(serving_cell_data)} serving cell assignments.")
        return serving_cell_data
    except Exception as e: logger.error(f"Serving cell determination error: {e}"); return pd.DataFrame(columns=["tick", "ue_id", "serving_cell_id"])

def _radp_metric_trafficload(trafficload_ue_data: pd.DataFrame, serving_cell_data: pd.DataFrame) -> Dict[int, float]:
    if trafficload_ue_data.empty or serving_cell_data.empty: return {}
    ue_data_merged = pd.merge(trafficload_ue_data[["tick", "ue_id"]], serving_cell_data, on=["tick", "ue_id"], how="left").dropna(subset=["serving_cell_id"])
    if ue_data_merged.empty: return {}
    ue_counts = ue_data_merged.groupby(["tick", "serving_cell_id"]).size()
    return ue_counts.groupby(level='tick').std().fillna(0).to_dict()

def energyload_metric_generator(serving_cell_data: pd.DataFrame, ue_rxpower_data: pd.DataFrame, site_config_data: pd.DataFrame, rx_power_threshold: float = -90,) -> Dict[int, float]:
    COL_CELL_ID=getattr(c,'CELL_ID','cell_id')
    if serving_cell_data.empty or ue_rxpower_data.empty or site_config_data.empty: return {}
    all_cell_ids = site_config_data[COL_CELL_ID].unique(); total_cells = len(all_cell_ids); cells_off_per_tick = {}
    if total_cells == 0: return {}
    for tick in serving_cell_data["tick"].unique():
        serving_tick = serving_cell_data[serving_cell_data["tick"] == tick]; rx_tick = ue_rxpower_data[ue_rxpower_data["tick"] == tick]
        can_be_off_count = 0
        for cell_id in all_cell_ids:
            ues_served = serving_tick[serving_tick["serving_cell_id"] == cell_id]["ue_id"]
            if ues_served.empty: can_be_off_count += 1; continue
            can_turn_off = True; rx_others = rx_tick[(rx_tick[COL_CELL_ID] != cell_id) & (rx_tick["ue_id"].isin(ues_served))]
            max_power_others = rx_others.groupby("ue_id")["rx_power_dbm"].max()
            # Efficient check if *any* UE served by this cell falls below threshold with other cells
            if max_power_others.loc[ues_served].lt(rx_power_threshold).any(): can_turn_off = False
            if can_turn_off: can_be_off_count += 1
        cells_off_per_tick[tick] = can_be_off_count / total_cells
    return cells_off_per_tick

# --- Main Runner ---
def run_traffic_simulation_and_analysis(site_config_data: pd.DataFrame, num_ues: int, spatial_params_path: str, time_params_path: str,) -> Dict:
    logger.info("--- Starting Traffic Simulation and Analysis ---")
    try:
        with open(spatial_params_path, "r") as f: spatial_params = json.load(f)
        with open(time_params_path, "r") as f: time_params = json.load(f)
    except Exception as e: logger.error(f"Error loading JSON config: {e}"); raise
    # *** Validate time_weights keys against spatial_params types ***
    expected_keys = spatial_params.get("types", [])
    actual_keys = time_params.get("time_weights", {}).keys()
    if set(expected_keys) != set(actual_keys):
         logger.warning(f"Mismatch between spatial types {expected_keys} and time_weights keys {list(actual_keys)}!")
         # Decide how to handle: error out, or use defaults? Using defaults might hide issues.
         # raise ValueError("Mismatch between spatial types and time_weights keys.")

    ue_data = _city_digitaltwin_generator(num_ues, spatial_params, time_params, site_config_data)
    if ue_data.empty: return {"error": "UE generation failed"}
    site_cfg_analysis = site_config_data.copy()
    COL_CELL_TXPWR_DBM = getattr(c, 'CELL_TXPWR_DBM', 'cell_txpwr_dbm')
    if COL_CELL_TXPWR_DBM not in site_cfg_analysis.columns:
         logger.warning(f"'{COL_CELL_TXPWR_DBM}' missing. Adding default for analysis.")
         site_cfg_analysis[COL_CELL_TXPWR_DBM] = 25.0
    rx_power = _radp_model_rftwin(ue_data, site_cfg_analysis)
    serving_cells = _determine_serving_cell(rx_power)
    traffic_metric = _radp_metric_trafficload(ue_data, serving_cells)
    energy_metric = energyload_metric_generator(serving_cells, rx_power, site_cfg_analysis)
    logger.info("--- Traffic Simulation and Analysis Finished ---")
    return {"trafficload_ue_data": ue_data, "ue_rxpower_data": rx_power, "serving_cell_data": serving_cells,
            "trafficload_metric_per_tick": traffic_metric, "energyload_metric_per_tick": energy_metric}

# --- Plotting ---
def plot(tick_to_plot: int, results: Dict, site_config_data: pd.DataFrame, spatial_params: Dict):
    logger.info(f"Generating plot for tick {tick_to_plot}...")
    if Voronoi is None: logger.warning("Scipy missing, cannot plot Voronoi."); return
    if not all(k in results for k in ["trafficload_ue_data", "serving_cell_data"]): logger.warning("Missing data for plot."); return
    ue_data_tick = results["trafficload_ue_data"][results["trafficload_ue_data"]["tick"] == tick_to_plot]
    serving_cell_tick = results["serving_cell_data"][results["serving_cell_data"]["tick"] == tick_to_plot]
    if ue_data_tick.empty: logger.warning(f"No UE data for tick {tick_to_plot}."); return
    ue_data_tick = pd.merge(ue_data_tick, serving_cell_tick[["ue_id", "serving_cell_id"]], on="ue_id", how="left")
    COL_LAT=getattr(c,'LAT','lat'); COL_LON=getattr(c,'LON','lon')
    COL_CELL_LAT=getattr(c,'CELL_LAT','cell_lat'); COL_CELL_LON=getattr(c,'CELL_LON','cell_lon')
    COL_CELL_ID=getattr(c,'CELL_ID','cell_id')

    fig, ax = plt.subplots(figsize=(12, 10))
    if len(site_config_data) >= 4:
        points = site_config_data[[COL_CELL_LON, COL_CELL_LAT]].values
        try: vor = Voronoi(points); voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors="gray", lw=1, line_alpha=0.6, point_size=0)
        except Exception as e: logger.warning(f"Voronoi plot failed: {e}")
    ax.scatter(site_config_data[COL_CELL_LON], site_config_data[COL_CELL_LAT], marker="^", c="red", label="Cell Towers", s=60, zorder=10)
    unique_serving = ue_data_tick["serving_cell_id"].dropna().unique()
    cmap = plt.get_cmap("tab20", max(1, len(unique_serving)))
    handles = [plt.Line2D([0], [0], marker='^', color='w', label='Cell Towers', mfc='red', ms=10)]
    labels = ['Cell Towers']
    for i, cell_id in enumerate(unique_serving):
        cell_ues = ue_data_tick[ue_data_tick["serving_cell_id"] == cell_id]
        if not cell_ues.empty:
            ax.scatter(cell_ues[COL_LON], cell_ues[COL_LAT], color=cmap(i), alpha=0.6, s=15, zorder=5)
            label = f"UEs (Cell {cell_id})"
            if label not in labels: handles.append(plt.Line2D([0], [0], marker='o', color=cmap(i), linestyle='', ms=6)); labels.append(label)
    ues_no_serve = ue_data_tick[ue_data_tick["serving_cell_id"].isna()]
    if not ues_no_serve.empty:
        ax.scatter(ues_no_serve[COL_LON], ues_no_serve[COL_LAT], c='black', marker='x', s=15, zorder=5)
        label = "UEs (No Serving Cell)"
        if label not in labels: handles.append(plt.Line2D([0],[0], marker='x', color='black', linestyle='', ms=6)); labels.append(label)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title(f"UE Distribution and Serving Cells (Tick {tick_to_plot})")
    ax.legend(handles=handles, labels=labels, loc='best', fontsize='small'); ax.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
    plot_dir = "./plots"; os.makedirs(plot_dir, exist_ok=True); filename = os.path.join(plot_dir, f"ue_distribution_tick_{tick_to_plot}.png")
    try: plt.savefig(filename); logger.info(f"Plot saved: {filename}")
    except Exception as e: logger.error(f"Save plot failed {filename}: {e}")
    plt.close(fig)

# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Configuration ---
    GENERATE_TOPOLOGY = True # <<< Set to False to load your file, True to generate dummy data
    SITE_CONFIG_CSV = "./data/topology.csv"

    NUM_SITES_TO_GENERATE = 10; CELLS_PER_SITE_TO_GENERATE = 3
    GENERATION_LAT_RANGE = (40.7, 40.8); GENERATION_LON_RANGE = (-74.05, -73.95)
    DEFAULT_CELL_POWER = 25.0

    SPATIAL_PARAMS_JSON = "./spatial_params.json"
    TIME_PARAMS_JSON = "./time_params.json"
    NUM_UES_TO_GENERATE = 500
    OUTPUT_UE_DATA_DIR = "./ue_data" # <<< Directory for per-tick UE data CSVs

    # --- Setup: Load or Generate Topology ---
    site_config_data = None
    if GENERATE_TOPOLOGY:
        logger.info("Generating dummy topology data...")
        site_config_data = generate_dummy_topology(
            num_sites=NUM_SITES_TO_GENERATE, cells_per_site=CELLS_PER_SITE_TO_GENERATE,
            lat_range=GENERATION_LAT_RANGE, lon_range=GENERATION_LON_RANGE,
            default_power_dbm=DEFAULT_CELL_POWER,
        )
        try:
            os.makedirs(os.path.dirname(SITE_CONFIG_CSV) or '.', exist_ok=True)
            site_config_data.to_csv(SITE_CONFIG_CSV, index=False)
            logger.info(f"Saved generated dummy topology to {SITE_CONFIG_CSV}")
        except Exception as e: logger.error(f"Could not save generated topology: {e}")
    else:
        logger.info(f"Loading topology data from {SITE_CONFIG_CSV}...")
        try:
            site_config_data = pd.read_csv(SITE_CONFIG_CSV)
            COL_CELL_ID=getattr(c,'CELL_ID','cell_id'); COL_CELL_LAT=getattr(c,'CELL_LAT','cell_lat'); COL_CELL_LON=getattr(c,'CELL_LON','cell_lon')
            required_cols = [COL_CELL_ID, COL_CELL_LAT, COL_CELL_LON]
            if not all(col in site_config_data.columns for col in required_cols):
                 missing = [col for col in required_cols if col not in site_config_data.columns]; raise ValueError(f"Topology missing required columns: {missing}")
            COL_CELL_TXPWR_DBM = getattr(c, 'CELL_TXPWR_DBM', 'cell_txpwr_dbm')
            if COL_CELL_TXPWR_DBM not in site_config_data.columns:
                 logger.warning(f"'{COL_CELL_TXPWR_DBM}' missing. Adding default {DEFAULT_CELL_POWER} dBm for RF analysis.")
                 site_config_data[COL_CELL_TXPWR_DBM] = DEFAULT_CELL_POWER
            logger.info(f"Loaded {len(site_config_data)} cells from {SITE_CONFIG_CSV}")
        except FileNotFoundError: logger.error(f"Topology file not found: {SITE_CONFIG_CSV}."); sys.exit(1)
        except Exception as e: logger.error(f"Error loading topology from {SITE_CONFIG_CSV}: {e}"); sys.exit(1)

    if site_config_data is None or site_config_data.empty: logger.error("Site config data unavailable."); sys.exit(1)

    # --- Create Dummy Spatial/Time JSONs ---
    # *** Uses actual type names now if spatial_params exists ***
    if not os.path.exists(SPATIAL_PARAMS_JSON):
        logger.warning(f"{SPATIAL_PARAMS_JSON} not found. Creating dummy data.")
        num_types_generated = 3; default_types = [f"type_{i+1}" for i in range(num_types_generated)]
        dummy_spatial_params = {"types": default_types, "proportions": list(np.random.dirichlet(np.ones(num_types_generated)))}
        try:
            with open(SPATIAL_PARAMS_JSON, "w") as f: json.dump(dummy_spatial_params, f, indent=4)
            spatial_types_for_time = default_types # Use generated types for time weights keys
        except Exception as e: logger.error(f"Could not write dummy {SPATIAL_PARAMS_JSON}: {e}"); spatial_types_for_time = []
    else:
        # Load existing types to use for time weights keys if time file needs generating
        try:
            with open(SPATIAL_PARAMS_JSON, "r") as f: loaded_spatial = json.load(f)
            spatial_types_for_time = loaded_spatial.get("types", [])
        except Exception as e: logger.error(f"Could not read {SPATIAL_PARAMS_JSON} for type names: {e}"); spatial_types_for_time = []

    if not os.path.exists(TIME_PARAMS_JSON):
        logger.warning(f"{TIME_PARAMS_JSON} not found. Creating dummy data.")
        num_ticks = 24
        if not spatial_types_for_time: logger.error("Cannot create dummy time params without spatial types defined."); sys.exit(1)
        dummy_time_params = {
            "total_ticks": num_ticks, "tick_duration": 1,
            "time_weights": { # *** Use actual type names as keys ***
                 stype: list(np.random.rand(num_ticks)) for stype in spatial_types_for_time
             },
         }
        try:
            with open(TIME_PARAMS_JSON, "w") as f: json.dump(dummy_time_params, f, indent=4)
        except Exception as e: logger.error(f"Could not write dummy {TIME_PARAMS_JSON}: {e}")

    # --- Run Simulation ---
    try:
        results = run_traffic_simulation_and_analysis(
            site_config_data=site_config_data, num_ues=NUM_UES_TO_GENERATE,
            spatial_params_path=SPATIAL_PARAMS_JSON, time_params_path=TIME_PARAMS_JSON,
        )
    except Exception as e: logger.exception(f"Traffic simulation failed: {e}"); sys.exit(1)
    if "error" in results or "trafficload_ue_data" not in results or results["trafficload_ue_data"].empty:
         logger.error("Failed to generate valid UE data from traffic simulation."); sys.exit(1)

    # --- Save UE Data Per Tick ---
    logger.info(f"Saving generated UE data per tick to directory: {OUTPUT_UE_DATA_DIR}")
    os.makedirs(OUTPUT_UE_DATA_DIR, exist_ok=True)
    generated_ue_data_all_ticks = results["trafficload_ue_data"]
    saved_files_count = 0; failed_saves = 0
    COL_LAT=getattr(c,'LAT','lat'); COL_LON=getattr(c,'LON','lon') # Get constants/defaults
    try:
        required_output_cols = ["ue_id", COL_LON, COL_LAT, "tick"]
        if not all(col in generated_ue_data_all_ticks.columns for col in required_output_cols):
            missing = [col for col in required_output_cols if col not in generated_ue_data_all_ticks.columns]; raise ValueError(f"Generated UE data missing cols: {missing}")

        for tick in sorted(generated_ue_data_all_ticks['tick'].unique()):
            try:
                ue_data_tick_snapshot = generated_ue_data_all_ticks[generated_ue_data_all_ticks['tick'] == tick].copy()
                final_ue_data_for_cco = ue_data_tick_snapshot[['ue_id', COL_LON, COL_LAT, 'tick']].rename(
                    columns={'ue_id': 'mock_ue_id', COL_LON: 'lon', COL_LAT: 'lat'}
                )
                final_ue_data_for_cco = final_ue_data_for_cco[['mock_ue_id', 'lon', 'lat', 'tick']]
                output_filename = f"generated_ue_data_for_cco_{tick}.csv"
                output_csv_path = os.path.join(OUTPUT_UE_DATA_DIR, output_filename)
                final_ue_data_for_cco.to_csv(output_csv_path, index=False)
                saved_files_count += 1
                logger.debug(f"Saved UE data for tick {tick} to: {output_csv_path}")
            except Exception as tick_e:
                 logger.error(f"Failed saving UE data for tick {tick}: {tick_e}")
                 failed_saves += 1

        logger.info(f"Finished saving per-tick UE data. Successful: {saved_files_count}, Failed: {failed_saves}.")

    except Exception as e: logger.error(f"Failed during per-tick UE data saving: {e}")

    # --- Optional: Print Metrics and Plot Results Per Tick ---
    logger.info("\n--- Analysis Metrics (Per Tick) ---")
    if "trafficload_metric_per_tick" in results:
        print("Traffic Load Metric (Std Dev UE Counts):")
        for tick, metric in sorted(results["trafficload_metric_per_tick"].items()): print(f"  Tick {tick:02d}: {metric:.2f}")
    if "energyload_metric_per_tick" in results:
        print("\nEnergy Load Metric (Prop. Cells Off):")
        for tick, metric in sorted(results["energyload_metric_per_tick"].items()): print(f"  Tick {tick:02d}: {metric:.3f}")

    logger.info("\n--- Generating Plots Per Tick (if matplotlib available) ---")
    try:
        # Reload spatial params for plotting context if needed
        with open(SPATIAL_PARAMS_JSON, "r") as f: spatial_params_plot = json.load(f)
        plotted_ticks_count = 0
        for tick in sorted(generated_ue_data_all_ticks['tick'].unique()):
             logger.debug(f"Generating plot for tick {tick}...")
             plot(tick, results, site_config_data, spatial_params_plot) # Pass original site_config
             plotted_ticks_count += 1
        logger.info(f"Successfully generated {plotted_ticks_count} plots in ./plots/ directory.")
    except NameError: # Catch if matplotlib failed to import
         logger.warning("Matplotlib not found or failed to import. Skipping plot generation.")
    except Exception as e: logger.error(f"Failed during plot generation: {e}")

    logger.info("Script finished.")