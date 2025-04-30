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
RADP_ROOT = os.getenv("MAVERIC_ROOT", "/path/to/your/maveric/project") # SET YOUR PATH HERE OR VIA ENV VAR
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


# --- Helper to generate layout once ---
def generate_spatial_layout(site_config_data: pd.DataFrame, spatial_params: Dict) -> List[Dict]:
    """Generates the Voronoi cells and assigns space types ONCE."""
    logger.info("Generating fixed spatial layout...")
    try:
        space_bound = space_boundary(site_config_data)
        space_list = _space_component_list_generator(spatial_params)
        spatial_cells = _create_spatial_cells(site_config_data, space_list, space_bound)
        if not spatial_cells:
            logger.error("Failed to generate any spatial cells for the layout.")
        return spatial_cells
    except Exception as e:
        logger.error(f"Error generating spatial layout: {e}")
        return [] # Return empty list on error

# In traffic_3.py

# --- (Keep other imports and functions) ---

def _city_digitaltwin_generator(
    total_ue: int,
    #spatial_params_for_city: Dict,
    time_params: Dict,
    #cell_topology_data: pd.DataFrame,
    spatial_cells: List[Dict] # <<< Accept pre-generated spatial cells
) -> pd.DataFrame:
    """
    Generates UE location data over time, distributing a target NUMBER of UEs
    per space type per tick based on time_weights.
    """
    logger.info(f"Generating UE data for approximately {total_ue} UEs per tick...")
    logger.info("Using pre-generated spatial cell layout.")
    

    if not spatial_cells:
         logger.error("No spatial cells generated. Cannot generate UE data.")
         return pd.DataFrame()

    # --- Pre-process: Group spatial cells by their type ---
    # Get unique types present in the provided spatial_cells
    spatial_types = sorted(list(set(cell['type'] for cell in spatial_cells)))
    cells_by_type = {stype: [] for stype in spatial_types}
    for cell_info in spatial_cells:
        cells_by_type[cell_info['type']].append(cell_info) # Group existing cells

    # Log if some defined types have no geographic areas assigned
    for stype in spatial_types:
        if not cells_by_type[stype]:
            logger.warning(f"The generated spatial layout contains no cells assigned the type '{stype}'. No UEs generated for this type.")


    # --- Time Simulation Loop ---
    total_ticks = time_params.get("total_ticks", 1)
    time_weights = time_params.get('time_weights', {})
    ue_data_list = []
    COL_LAT = getattr(c, 'LAT', 'lat'); COL_LON = getattr(c, 'LON', 'lon')

    for tick in range(total_ticks):
        logger.debug(f"--- Generating UEs for Tick {tick} ---")
        # --- Calculate Target UE Count per Space Type for this Tick ---
        tick_type_weights = {stype: time_weights.get(stype, [0]*total_ticks)[tick]
                             for stype in spatial_types if tick < len(time_weights.get(stype, []))}
        
        total_weight_this_tick = sum(tick_type_weights.values())

        if total_weight_this_tick <= 0:
            logger.warning(f"Sum of weights is zero for tick {tick}. No UEs generated for this tick.")
            continue

        # Calculate target counts, handling potential rounding errors
        target_counts = {}
        calculated_total = 0
        for stype, weight in tick_type_weights.items():
            proportion = weight / total_weight_this_tick
            target_counts[stype] = int(round(proportion * total_ue)) # Round to nearest int
            calculated_total += target_counts[stype]

        # Adjust counts slightly if rounding caused mismatch with total_ue
        difference = total_ue - calculated_total
        if difference != 0:
            logger.debug(f"Adjusting rounded UE counts by {difference} for tick {tick}.")
            # Simple adjustment: add/remove from type with largest count
            if target_counts: # Ensure target_counts is not empty
                 adjust_type = max(target_counts, key=target_counts.get)
                 target_counts[adjust_type] += difference
                 # Ensure count doesn't go below zero
                 target_counts[adjust_type] = max(0, target_counts[adjust_type])

        logger.debug(f"Tick {tick} Target UE counts: {target_counts}")

        # --- Place UEs per Space Type ---
        current_ue_id_in_tick = 0 # Unique ID within this tick's generation batch
        for space_type, num_ues_for_type in target_counts.items():
            if num_ues_for_type == 0:
                continue

            available_cells_for_type = cells_by_type.get(space_type, [])
            if not available_cells_for_type:
                logger.warning(f"Cannot place {num_ues_for_type} UEs for type '{space_type}' at tick {tick}: No spatial cells assigned this type.")
                continue # Skip to next type

            logger.debug(f"Placing {num_ues_for_type} UEs in {len(available_cells_for_type)} cells of type '{space_type}'...")

            for _ in range(num_ues_for_type):
                # Randomly choose one of the available cells of the correct type
                chosen_cell = np.random.choice(available_cells_for_type)

                # Generate point within the chosen cell polygon
                try:
                    polygon = Polygon(chosen_cell['bounds'])
                    min_lon, min_lat, max_lon, max_lat = polygon.bounds
                    attempts = 0; max_attempts = 100 # Prevent infinite loops
                    while attempts < max_attempts:
                        ue_lon = np.random.uniform(min_lon, max_lon)
                        ue_lat = np.random.uniform(min_lat, max_lat)
                        point = Point(ue_lon, ue_lat)
                        if polygon.contains(point):
                            # Store UE data
                            ue_data_list.append({
                                "tick": tick,
                                "ue_id": current_ue_id_in_tick,
                                COL_LAT: ue_lat,
                                COL_LON: ue_lon,
                                "space_type": space_type, # Store the target type
                                "voronoi_cell_id": chosen_cell['cell_id'], # Original Voronoi cell ID
                            })
                            current_ue_id_in_tick += 1
                            break # Point generated successfully
                        attempts += 1
                    else: # If loop finishes without break
                         logger.warning(f"Point generation failed after {max_attempts} attempts for cell {chosen_cell['cell_id']} (type '{space_type}'), tick {tick}. Skipping this UE.")
                         # Optionally, just assign current_ue_id_in_tick anyway if total matters more? No, better to skip.
                except Exception as e:
                     logger.error(f"Error placing UE {current_ue_id_in_tick} tick {tick} in cell {chosen_cell['cell_id']}: {e}")
                     # Don't increment ue_id if placement failed

    # --- Final DataFrame ---
    if not ue_data_list:
         logger.warning("No UE data points were generated across all ticks.")
         return pd.DataFrame()

    final_ue_df = pd.DataFrame(ue_data_list)
    logger.info(f"Finished UE generation. Total points generated: {len(final_ue_df)}")
    # Add unique UE ID across all ticks if needed? Currently ue_id is 0..N-1 within each tick's batch.
    # If a persistent UE ID is needed: final_ue_df['persistent_ue_id'] = final_ue_df['tick'] * total_ue + final_ue_df['ue_id']

    return final_ue_df

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


# --- Plotting ---
# In traffic_3.py (replace the existing plot function)

# --- Add Matplotlib Patch for legend ---
from matplotlib.patches import Patch # Add this import near the top with other imports

def plot(tick_to_plot: int, results: Dict, site_config_data: pd.DataFrame, spatial_params: Dict, spatial_cells: List[Dict]):
    """
    Visualizes Voronoi, cells, UEs (colored by serving cell), and shaded
    spatial type areas for a specific tick.
    """
    logger.info(f"Generating plot for tick {tick_to_plot}...")
    # --- Basic Checks ---
    if Voronoi is None:
        logger.warning("Scipy (Voronoi) not found. Cannot plot Voronoi regions.")
        # Decide if you want to proceed without Voronoi lines or return
        # return # Or continue without Voronoi elements

    if not all(k in results for k in ["trafficload_ue_data", "serving_cell_data"]):
        logger.warning(f"Cannot generate plot for tick {tick_to_plot}: Missing required data in results.")
        return

    # --- Get Data for Tick ---
    ue_data_tick = results["trafficload_ue_data"][results["trafficload_ue_data"]["tick"] == tick_to_plot]
    serving_cell_tick = results["serving_cell_data"][results["serving_cell_data"]["tick"] == tick_to_plot]

    if ue_data_tick.empty:
        logger.warning(f"No UE data found for tick {tick_to_plot}. Skipping plot.")
        return

    # Merge serving cell info (use left merge to keep all UEs for the tick)
    ue_data_tick = pd.merge(
        ue_data_tick,
        serving_cell_tick[["ue_id", "serving_cell_id"]], # Only need these columns from serving data
        on="ue_id",
        how="left",
    )

    # --- Constants ---
    COL_LAT=getattr(c,'LAT','lat'); COL_LON=getattr(c,'LON','lon')
    COL_CELL_LAT=getattr(c,'CELL_LAT','cell_lat'); COL_CELL_LON=getattr(c,'CELL_LON','cell_lon')
    COL_CELL_ID=getattr(c,'CELL_ID','cell_id')

    # --- Create Plot Figure ---
    fig, ax = plt.subplots(figsize=(14, 11)) # Adjust size as needed

    # # --- 1. Regenerate Spatial Cells for Shading ---
    # # (Alternatively, generate once outside and pass `spatial_cells` in `results`)
    # spatial_cells = []
    # space_types_defined = spatial_params.get("types", [])
    # if not space_types_defined:
    #     logger.warning("No 'types' defined in spatial_params. Cannot shade areas.")
    # else:
    #     try:
    #         space_bound = space_boundary(site_config_data) # Use same buffer %
    #         space_list = _space_component_list_generator(spatial_params) # Use actual type names
    #         spatial_cells = _create_spatial_cells(site_config_data, space_list, space_bound)
    #     except Exception as e:
    #         logger.error(f"Could not regenerate spatial cells for plotting: {e}")

    # --- 2. Define Colors for Spatial Types ---
    # Use a categorical colormap suitable for distinct areas
    space_types_defined = spatial_params.get("types", [])
    if not space_types_defined: logger.warning("No 'types' in spatial_params for plotting.")
    cmap_spaces = plt.get_cmap('tab10', len(space_types_defined))
    space_colors = {stype: cmap_spaces(i) for i, stype in enumerate(space_types_defined)}


    # --- 3. Plot Shaded Spatial Area Polygons ---
    # --- Plot Shaded Spatial Areas (Use passed spatial_cells) ---
    plotted_space_labels = set()
    spatial_legend_handles = []
    if spatial_cells: # Use the passed list directly
        logger.debug(f"Plotting {len(spatial_cells)} pre-generated spatial cell areas...")
        for cell_info in spatial_cells:
            # ... (Plotting logic using ax.fill remains the same) ...
            # ... (Uses cell_info['bounds'] and cell_info['type']) ...
            try:
                polygon = Polygon(cell_info["bounds"]); space_type = cell_info["type"]
                color = space_colors.get(space_type, 'lightgrey')
                ax.fill(*polygon.exterior.xy, color=color, alpha=0.15, zorder=1)
                if space_type not in plotted_space_labels:
                     label_text = space_type.replace("_"," ").title()
                     spatial_legend_handles.append(Patch(facecolor=color, alpha=0.3, label=label_text))
                     plotted_space_labels.add(space_type)
            except Exception as e: logger.warning(f"Could not plot spatial cell {cell_info.get('cell_id', 'N/A')}: {e}")

    else:
        logger.warning("No spatial cells provided to plot background shading.")
    # --- 4. Plot Voronoi Lines (Optional) ---
    if Voronoi and voronoi_plot_2d and len(site_config_data) >= 4:
        points = site_config_data[[COL_CELL_LON, COL_CELL_LAT]].values
        try:
            vor = Voronoi(points)
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors="black", lw=0.5, line_style=':', alpha=0.4, point_size=0, zorder=2) # Subtler lines
        except Exception as e: logger.warning(f"Voronoi plot failed: {e}")

    # --- 5. Plot Cell Towers ---
    tower_handle = ax.scatter(
        site_config_data[COL_CELL_LON], site_config_data[COL_CELL_LAT],
        marker="^", c="black", edgecolors='white', label="Cell Towers", s=80, zorder=10 # Make towers prominent
    )

    # --- 6. Plot UEs (Colored by Serving Cell) ---
    unique_serving_cells = sorted(ue_data_tick["serving_cell_id"].dropna().unique()) # Sort for consistent color mapping
    cmap_ues = plt.get_cmap("viridis", max(1, len(unique_serving_cells))) # Use 'viridis' or similar for UEs
    ue_legend_handles = []
    plotted_ue_labels = set()

    for i, cell_id in enumerate(unique_serving_cells):
        cell_ues = ue_data_tick[ue_data_tick["serving_cell_id"] == cell_id]
        if not cell_ues.empty:
            color = cmap_ues(i / max(1, len(unique_serving_cells)-1)) # Normalize index
            ax.scatter(cell_ues[COL_LON], cell_ues[COL_LAT], color=color, alpha=0.7, s=10, zorder=5) # Slightly smaller UEs
            label = f"UEs -> {cell_id}" # Label includes serving cell
            if label not in plotted_ue_labels:
                 # Create proxy artist for legend
                 ue_legend_handles.append(plt.Line2D([0], [0], marker='o', color=color, linestyle='', ms=5, label=label))
                 plotted_ue_labels.add(label)

    # Plot UEs with no serving cell
    ues_no_serve = ue_data_tick[ue_data_tick["serving_cell_id"].isna()]
    no_serve_handle = None
    if not ues_no_serve.empty:
        ax.scatter(ues_no_serve[COL_LON], ues_no_serve[COL_LAT], c='red', marker='x', s=15, zorder=6) # Red 'x'
        label = "UEs (No Serving)"
        if label not in plotted_ue_labels:
            no_serve_handle = plt.Line2D([0],[0], marker='x', color='red', linestyle='', ms=6, label=label)
            plotted_ue_labels.add(label)


    # --- 7. Setup Plot Appearance & Combined Legend ---
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"UE Distribution, Serving Cells, and Spatial Types (Tick {tick_to_plot})")

    # Combine legend handles - Towers, then Space Types, then UEs
    all_handles = [tower_handle] + spatial_legend_handles + ue_legend_handles
    if no_serve_handle: all_handles.append(no_serve_handle)

    # Place legend outside plot area to avoid overlap
    ax.legend(handles=all_handles, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', title="Legend", frameon=False)

    ax.grid(True, linestyle='--', alpha=0.4)
    # Adjust layout AFTER legend is placed
    plt.subplots_adjust(right=0.75) # Adjust right margin to fit legend


    # --- 8. Save Plot ---
    plot_dir = "./plots"; os.makedirs(plot_dir, exist_ok=True); filename = os.path.join(plot_dir, f"ue_distribution_tick_{tick_to_plot}.png")
    try:
        plt.savefig(filename, bbox_inches='tight') # Use bbox_inches='tight' to include legend
        logger.info(f"Plot saved: {filename}")
    except Exception as e:
        logger.error(f"Save plot failed {filename}: {e}")
    plt.close(fig) # Close the figure

# --- Config Generation Function ---
def generate_dummy_config(
    topology_df: pd.DataFrame,
    param_name: str = getattr(c, 'CELL_EL_DEG', 'cell_el_deg'), # Use getattr for constant safety
    default_value: float = 12.0 # Default starting value (e.g., for tilt)
) -> pd.DataFrame:
    """
    Generates a dummy config DataFrame with a default starting value
    for a specified parameter for all cells in the topology.

    Args:
        topology_df: The topology DataFrame (must contain c.CELL_ID).
        param_name: The name of the parameter column to create (e.g., 'cell_el_deg').
        default_value: The initial value to assign to the parameter for all cells.

    Returns:
        Pandas DataFrame with 'cell_id' and the parameter column.
    """
    COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id') # Use constant/default

    if topology_df is None or topology_df.empty:
        raise ValueError("Topology DataFrame must be provided to generate config.")
    if COL_CELL_ID not in topology_df.columns:
        raise ValueError(f"Topology DataFrame must contain the column '{COL_CELL_ID}'.")

    logger.info(f"Generating dummy config for parameter '{param_name}' with default value {default_value}.")

    # Get unique cell IDs from topology to ensure config matches
    config_df = topology_df[[COL_CELL_ID]].drop_duplicates().reset_index(drop=True)

    # Add the parameter column with the default value
    config_df[param_name] = default_value

    logger.info(f"Generated dummy config DataFrame with {len(config_df)} rows.")
    # Return only the required columns: cell_id and the parameter
    return config_df[[COL_CELL_ID, param_name]]


# --- Main Runner Function (Modified) ---
def run_traffic_simulation_and_analysis(
    site_config_data: pd.DataFrame, # Topology + TxPower
    initial_config_data: pd.DataFrame, # <<< ADDED: Config with initial cell_el_deg
    spatial_cells: List[Dict], # <<< Accept pre-generated spatial cells
    num_ues: int,
    spatial_params_path: str,
    time_params_path: str,
) -> Dict:
    """
    Loads spatial/time config, runs UE generation, performs basic analysis,
    and generates DUMMY training data CSV format.
    """
    logger.info("--- Starting Traffic Simulation and Analysis ---")
    # Get constant names or defaults
    COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')
    COL_CELL_EL_DEG = getattr(c, 'CELL_EL_DEG', 'cell_el_deg')
    COL_LAT=getattr(c,'LAT','lat'); COL_LON=getattr(c,'LON','lon')

    # --- Load Spatial/Time Params ---
    try:
        with open(spatial_params_path, "r") as f: spatial_params = json.load(f)
        with open(time_params_path, "r") as f: time_params = json.load(f)
        # Validate time_weights keys against spatial_params types
        expected_keys = spatial_params.get("types", [])
        actual_keys = time_params.get("time_weights", {}).keys()
        if set(expected_keys) != set(actual_keys):
             logger.warning(f"Mismatch between spatial types {expected_keys} and time_weights keys {list(actual_keys)}!")
    except Exception as e: logger.error(f"Error loading JSON config: {e}"); raise

    # --- Generate UE Locations ---
    trafficload_ue_data = _city_digitaltwin_generator(
       total_ue=num_ues,
        # spatial_params_for_city is implicitly defined by the types in spatial_cells now
        time_params=time_params,
        spatial_cells=spatial_cells # <<< Pass the fixed layout
    )
    if "error" in trafficload_ue_data or trafficload_ue_data.empty: # Check DataFrame directly
        logger.error("UE generation failed within run function.")
        return {"error": "UE generation failed"}

    # --- Perform Basic RF Simulation and Analysis ---
    site_cfg_analysis = site_config_data.copy()
    COL_CELL_TXPWR_DBM = getattr(c, 'CELL_TXPWR_DBM', 'cell_txpwr_dbm')
    if COL_CELL_TXPWR_DBM not in site_cfg_analysis.columns:
         logger.warning(f"'{COL_CELL_TXPWR_DBM}' missing. Adding default for RF analysis."); site_cfg_analysis[COL_CELL_TXPWR_DBM] = 25.0
    ue_rxpower_data = _radp_model_rftwin(trafficload_ue_data, site_cfg_analysis)
    serving_cell_data = _determine_serving_cell(ue_rxpower_data)
    trafficload_metric = _radp_metric_trafficload(trafficload_ue_data, serving_cell_data)
    energyload_metric = energyload_metric_generator(serving_cell_data, ue_rxpower_data, site_cfg_analysis)

# ... (Calls _radp_model_rftwin, _determine_serving_cell, metrics) ...
    site_cfg_analysis = site_config_data.copy()
    COL_CELL_TXPWR_DBM = getattr(c, 'CELL_TXPWR_DBM', 'cell_txpwr_dbm')
    if COL_CELL_TXPWR_DBM not in site_cfg_analysis.columns:
         logger.warning(f"'{COL_CELL_TXPWR_DBM}' missing. Adding default for RF analysis."); site_cfg_analysis[COL_CELL_TXPWR_DBM] = 25.0
    ue_rxpower_data = _radp_model_rftwin(trafficload_ue_data, site_cfg_analysis)
    serving_cell_data = _determine_serving_cell(ue_rxpower_data)
    trafficload_metric = _radp_metric_trafficload(trafficload_ue_data, serving_cell_data)
    energyload_metric = energyload_metric_generator(serving_cell_data, ue_rxpower_data, site_cfg_analysis)
    dummy_training_data = pd.DataFrame() # Regenerate dummy training data based on this run
    # ... (logic to generate dummy_training_data as before) ...

# Inside run_traffic_simulation_and_analysis function in traffic_3.py

    # --- *** GENERATE DUMMY TRAINING DATA (Limited Size) *** ---
    logger.warning("Generating DUMMY training data - RSRP values DO NOT accurately reflect tilt effects.")
    logger.warning(f"Limiting dummy training data to approximately 12000 rows.")
    dummy_training_data_list = []
    possible_tilts = list(np.arange(0.0, 21.0, 1.0))
    assumed_optimal_tilt = 8.0
    tilt_penalty_factor = 0.5
    TARGET_TRAINING_ROWS = 12000

    try:
        if trafficload_ue_data.empty:
             logger.warning("Cannot generate dummy training data: UE data is empty.")
        else:
            COL_CELL_EL_DEG = getattr(c, 'CELL_EL_DEG', 'cell_el_deg')
            COL_LAT=getattr(c,'LAT','lat'); COL_LON=getattr(c,'LON','lon')
            COL_CELL_LAT=getattr(c,'CELL_LAT','cell_lat'); COL_CELL_LON=getattr(c,'CELL_LON','cell_lon')
            COL_CELL_ID=getattr(c,'CELL_ID','cell_id'); COL_CELL_TXPWR_DBM=getattr(c,'CELL_TXPWR_DBM','cell_txpwr_dbm')
            ref_rx_power = -50
            path_loss_exponent = 3.5

            num_cells = len(site_config_data[COL_CELL_ID].unique())
            if num_cells == 0: raise ValueError("No cells found in site_config_data.")

            # --- Sampling Step ---
            num_ue_tick_samples = max(1, int(round(TARGET_TRAINING_ROWS / num_cells)))
            logger.info(f"Targeting ~{TARGET_TRAINING_ROWS} rows. Will sample {num_ue_tick_samples} UE-tick pairs and process for all {num_cells} cells.")

            # Ensure we don't sample more rows than available
            num_available_ue_ticks = len(trafficload_ue_data)
            if num_ue_tick_samples > num_available_ue_ticks:
                logger.warning(f"Requested {num_ue_tick_samples} samples, but only {num_available_ue_ticks} UE-tick pairs available. Using all available.")
                num_ue_tick_samples = num_available_ue_ticks

            # Sample distinct UE-tick pairs (can adjust replace=True if needed, but sampling unique points is better)
            # If trafficload_ue_data has unique ue_id per tick, sampling rows is fine.
            # If ue_id repeats across ticks, maybe sample based on ('tick', 'ue_id') groups?
            # Let's assume sampling rows is sufficient for now.
            sampled_ue_data = trafficload_ue_data.sample(n=num_ue_tick_samples, replace=False, random_state=42) # Use random_state for reproducibility
            logger.info(f"Processing {len(sampled_ue_data)} sampled UE-tick pairs.")

            # --- Iterate through SAMPLED UE points ---
            for _, ue_row in sampled_ue_data.iterrows():
                ue_lat = ue_row[COL_LAT]
                ue_lon = ue_row[COL_LON]
                # ue_tick = ue_row['tick'] # Not needed for final output format
                ue_id = ue_row['ue_id'] # For logging

                # --- Iterate through ALL cells for each sampled UE point ---
                for _, cell_row in site_config_data.iterrows():
                    cell_id = cell_row[COL_CELL_ID]
                    cell_lat = cell_row[COL_CELL_LAT]
                    cell_lon = cell_row[COL_CELL_LON]
                    cell_txpwr = cell_row[COL_CELL_TXPWR_DBM]

                    # 1. Calculate Simple RSRP
                    try:
                        dist_km = GISTools.dist((ue_lat, ue_lon), (cell_lat, cell_lon))
                        dist_m = dist_km * 1000.0
                        simple_rsrp = ref_rx_power - 10 * path_loss_exponent * np.log10(dist_m) if dist_m > 1e-3 else cell_txpwr
                    except Exception as e:
                        logger.warning(f"RSRP calc error UE {ue_id}, Cell {cell_id}: {e}. Skipping sample.")
                        continue

                    # 2. Assign Random Tilt
                    random_tilt = np.random.choice(possible_tilts)

                    # 3. Apply Crude Tilt Adjustment
                    tilt_deviation = abs(random_tilt - assumed_optimal_tilt)
                    rsrp_penalty = tilt_deviation * tilt_penalty_factor
                    adjusted_rsrp = simple_rsrp - rsrp_penalty

                    # 4. Append Data
                    dummy_training_data_list.append({
                        COL_CELL_ID: cell_id,
                        "avg_rsrp": adjusted_rsrp,
                        "lon": ue_lon,
                        "lat": ue_lat,
                        COL_CELL_EL_DEG: random_tilt
                    })

            # --- Create Final DataFrame ---
            if dummy_training_data_list:
                 dummy_training_data = pd.DataFrame(dummy_training_data_list)
                 final_columns = [COL_CELL_ID, "avg_rsrp", "lon", "lat", COL_CELL_EL_DEG]
                 dummy_training_data = dummy_training_data[final_columns]
                 logger.info(f"Generated limited dummy training data with {len(dummy_training_data)} rows.")
            else:
                 logger.warning("No dummy training data points were generated.")
                 dummy_training_data = pd.DataFrame()

    except Exception as e:
        logger.exception(f"Failed to generate dummy training data: {e}")
        dummy_training_data = pd.DataFrame()

    logger.info("--- Traffic Simulation and Analysis Finished ---")
    # Return results including the dummy data
    return {
        "trafficload_ue_data": trafficload_ue_data,
        "ue_rxpower_data": ue_rxpower_data, # Note: This still contains simple RSRP, not adjusted
        "serving_cell_data": serving_cell_data, # Note: Based on simple RSRP
        "trafficload_metric_per_tick": trafficload_metric,
        "energyload_metric_per_tick": energyload_metric,
        "dummy_training_data": dummy_training_data, # Contains adjusted RSRP and random tilt
        "spatial_cells": spatial_cells
    }

# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Configuration ---
    GENERATE_TOPOLOGY = True
    GENERATE_CONFIG = True # Generate initial config as well
    SITE_CONFIG_CSV = "./data/topology.csv"
    CONFIG_CSV = "./data/config.csv"
    OUTPUT_DUMMY_TRAINING_CSV = "./dummy_ue_training_data.csv" # <<< Path for dummy training data
    OUTPUT_UE_DATA_DIR = "./ue_data" # For per-tick prediction data
    # ... other configs ...
    NUM_SITES_TO_GENERATE = 10; CELLS_PER_SITE_TO_GENERATE = 3
    GENERATION_LAT_RANGE = (40.7, 40.8); GENERATION_LON_RANGE = (-74.05, -73.95)
    DEFAULT_CELL_POWER = 25.0; DEFAULT_CELL_TILT = 12.0
    SPATIAL_PARAMS_JSON = "./spatial_params.json"; TIME_PARAMS_JSON = "./time_params.json"
    NUM_UES_TO_GENERATE = 500

    # --- Load/Generate Topology ---
    # ... (Logic as before - ensures site_config_data is created/loaded) ...
    site_config_data = None # Initialize
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


    # --- Load/Generate Initial Config ---
    initial_config_data = None
    COL_CELL_EL_DEG = getattr(c, 'CELL_EL_DEG', 'cell_el_deg')
    if GENERATE_CONFIG:
         logger.info("Generating dummy initial configuration data...")
         try:
              initial_config_data = generate_dummy_config( # Assuming generate_dummy_config exists
                   topology_df=site_config_data,
                   param_name=COL_CELL_EL_DEG,
                   default_value=DEFAULT_CELL_TILT
              )
              os.makedirs(os.path.dirname(CONFIG_CSV) or '.', exist_ok=True)
              initial_config_data.to_csv(CONFIG_CSV, index=False)
              logger.info(f"Saved generated dummy initial config to {CONFIG_CSV}")
         except Exception as e: logger.error(f"Failed to generate dummy config: {e}"); sys.exit(1)
    else:
         logger.info(f"Loading initial configuration data from {CONFIG_CSV}...")
         try:
            initial_config_data = pd.read_csv(CONFIG_CSV)
            COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')
            required_config_cols = [COL_CELL_ID, COL_CELL_EL_DEG]
            if not all(col in initial_config_data.columns for col in required_config_cols):
                 missing = [col for col in required_config_cols if col not in initial_config_data.columns]; raise ValueError(f"Config CSV missing required columns: {missing}")
            topology_cell_ids = set(site_config_data[COL_CELL_ID])
            config_cell_ids = set(initial_config_data[COL_CELL_ID])
            missing_in_config = topology_cell_ids - config_cell_ids
            if missing_in_config: raise ValueError(f"Config file missing cells from topology: {missing_in_config}")
            extra_in_config = config_cell_ids - topology_cell_ids
            if extra_in_config: logger.warning(f"Config file has extra cells not in topology: {extra_in_config}")
            logger.info(f"Loaded initial config for {len(initial_config_data)} cells from {CONFIG_CSV}")
         except FileNotFoundError: logger.error(f"Initial config file not found: {CONFIG_CSV}."); sys.exit(1)
         except Exception as e: logger.error(f"Error loading initial config from {CONFIG_CSV}: {e}"); sys.exit(1)

    if initial_config_data is None or initial_config_data.empty: logger.error("Initial config data unavailable."); sys.exit(1)


    # --- Create Dummy Spatial/Time JSONs if needed ---
    # ... (Logic remains the same) ...
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

    # --- *** Generate Spatial Layout ONCE *** ---
    spatial_cells_layout = []
    try:
        logger.info("Pre-generating fixed spatial layout...")
        # Need spatial params loaded first for _space_component_list_generator
        with open(SPATIAL_PARAMS_JSON, "r") as f: spatial_params_for_layout = json.load(f)
        spatial_cells_layout = generate_spatial_layout(site_config_data, spatial_params_for_layout)
        if not spatial_cells_layout:
             logger.error("Failed to generate spatial layout. Exiting.")
             sys.exit(1)
        logger.info(f"Generated {len(spatial_cells_layout)} spatial cells for fixed layout.")
        # --- Optional: Save the generated layout ---
        try:
            with open("./generated_spatial_layout.json", "w") as f:
                # Convert numpy arrays in bounds to lists for JSON serialization
                serializable_layout = []
                for cell in spatial_cells_layout:
                     serializable_layout.append({
                         "bounds": [[coord[0], coord[1]] for coord in cell["bounds"]], # Convert tuples/arrays
                         "type": cell["type"],
                         "cell_id": cell["cell_id"],
                         "original_voronoi_region": cell.get("original_voronoi_region")
                     })
                json.dump(serializable_layout, f, indent=2)
            logger.info("Saved generated spatial layout to generated_spatial_layout.json")
        except Exception as e:
            logger.error(f"Could not save spatial layout: {e}")
    except Exception as e:
        logger.error(f"Error during spatial layout generation step: {e}")
        sys.exit(1)

    # --- Run Simulation (Passing Initial Config) ---
    try:
        results = run_traffic_simulation_and_analysis(
            site_config_data=site_config_data,
            initial_config_data=initial_config_data, # <<< Pass config data
            spatial_cells=spatial_cells_layout, # <<< Pass the fixed layout
            num_ues=NUM_UES_TO_GENERATE,
            spatial_params_path=SPATIAL_PARAMS_JSON,
            time_params_path=TIME_PARAMS_JSON,
        )
    except Exception as e:
        logger.exception(f"Traffic simulation failed during run: {e}")
        sys.exit(1)

    if "error" in results or "trafficload_ue_data" not in results or results["trafficload_ue_data"].empty:
         logger.error("Failed to generate valid UE data from traffic simulation."); sys.exit(1)

    # --- Save Per-Tick UE Data (for CCO Prediction/Evaluation) ---
    # ... (Logic remains the same - saves to OUTPUT_UE_DATA_DIR) ...

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


    # --- *** SAVE DUMMY TRAINING DATA *** ---
    logger.info(f"Saving DUMMY training data format to {OUTPUT_DUMMY_TRAINING_CSV}")
    if "dummy_training_data" in results and not results["dummy_training_data"].empty:
        try:
            dummy_training_df = results["dummy_training_data"]
            # Ensure output directory exists (might be different from ./data/)
            os.makedirs(os.path.dirname(OUTPUT_DUMMY_TRAINING_CSV) or '.', exist_ok=True)
            dummy_training_df.to_csv(OUTPUT_DUMMY_TRAINING_CSV, index=False)
            logger.info(f"Successfully saved DUMMY training data format ({len(dummy_training_df)} rows) to: {OUTPUT_DUMMY_TRAINING_CSV}")
            logger.warning("REMINDER: The 'avg_rsrp' in this dummy training data is NOT realistic regarding tilt effects and is based on the INITIAL tilts only.")
        except Exception as e:
            logger.error(f"Failed to save dummy training data CSV to {OUTPUT_DUMMY_TRAINING_CSV}: {e}")
    else:
        logger.warning("No dummy training data was generated (check for simulation/merge errors).")

    # --- Optional: Print Metrics and Plot Results Per Tick ---
    # ... (Logic remains the same) ...
    # --- Optional: Print Metrics and Plot Results Per Tick ---
    logger.info("\n--- Analysis Metrics (Per Tick) ---")
    if "trafficload_metric_per_tick" in results:
        print("Traffic Load Metric (Std Dev UE Counts):")
        for tick, metric in sorted(results["trafficload_metric_per_tick"].items()): print(f"  Tick {tick:02d}: {metric:.2f}")
    if "energyload_metric_per_tick" in results:
        print("\nEnergy Load Metric (Prop. Cells Off):")
        for tick, metric in sorted(results["energyload_metric_per_tick"].items()): print(f"  Tick {tick:02d}: {metric:.3f}")

    logger.info("\n--- Generating Plots Per Tick (Using Fixed Spatial Layout) ---")
    try:
        with open(SPATIAL_PARAMS_JSON, "r") as f: spatial_params_plot = json.load(f)
        generated_ue_data_all_ticks = results["trafficload_ue_data"]
        plotted_ticks_count = 0
        for tick in sorted(generated_ue_data_all_ticks['tick'].unique()):
             logger.debug(f"Generating plot for tick {tick}...")
             # Pass the fixed spatial_cells_layout to plot function
             plot(tick, results, site_config_data, spatial_params_plot, spatial_cells_layout) #<<< Pass layout
             plotted_ticks_count += 1
        logger.info(f"Successfully generated {plotted_ticks_count} plots in ./plots/ directory.")
    except NameError: # Catch if matplotlib failed to import
         logger.warning("Matplotlib not found or failed to import. Skipping plot generation.")
    except Exception as e: logger.error(f"Failed during plot generation: {e}")


    logger.info("Script finished.")