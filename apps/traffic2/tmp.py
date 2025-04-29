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

# --- Modify _city_digitaltwin_generator ---
def _city_digitaltwin_generator(
    total_ue: int,
    # spatial_params_for_city: Dict, # No longer needed directly if space_list passed
    time_params: Dict,
    # cell_topology_data: pd.DataFrame, # No longer needed directly
    spatial_cells: List[Dict] # <<< Accept pre-generated spatial cells
) -> pd.DataFrame:
    """Generates UE locations using a FIXED spatial layout."""
    logger.info(f"Generating UE data for approximately {total_ue} UEs per tick...")
    logger.info("Using pre-generated spatial cell layout.")

    if not spatial_cells:
         logger.error("No spatial cells provided. Cannot generate UE data.")
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
    # ... (Rest of the function remains the same as the PREVIOUS version) ...
    # ... (It uses the passed `spatial_cells` and `cells_by_type` map) ...
    # ... (It calculates target counts per type based on time_weights) ...
    # ... (It places UEs within the appropriate cells from cells_by_type map) ...
    # --- Omitted for brevity, logic is identical to previous version using the passed spatial_cells ---
    total_ticks = time_params.get("total_ticks", 1)
    time_weights = time_params.get('time_weights', {})
    ue_data_list = []
    COL_LAT = getattr(c, 'LAT', 'lat'); COL_LON = getattr(c, 'LON', 'lon')

    for tick in range(total_ticks):
        # Calculate Target Counts (same logic as before)
        tick_type_weights = {}
        for stype in spatial_types:
             weights_list = time_weights.get(stype, [])
             if tick < len(weights_list): tick_type_weights[stype] = max(0, weights_list[tick])
             else: tick_type_weights[stype] = 0.0
        total_weight_this_tick = sum(tick_type_weights.values())
        if total_weight_this_tick <= 1e-9: continue
        target_counts = {}; running_total = 0
        sorted_types = sorted(tick_type_weights.keys())
        for stype in sorted_types:
            proportion = tick_type_weights[stype] / total_weight_this_tick
            target_counts[stype] = int(round(proportion * total_ue))
            running_total += target_counts[stype]
        difference = total_ue - running_total
        if difference != 0 and target_counts:
            adjust_type = max(target_counts, key=target_counts.get)
            target_counts[adjust_type] = max(0, target_counts[adjust_type] + difference)
        # Place UEs (same logic as before, uses pre-grouped cells_by_type)
        ue_id_counter_for_tick = 0
        for space_type, num_ues_for_type in target_counts.items():
            if num_ues_for_type <= 0: continue
            available_cells_for_type = cells_by_type.get(space_type)
            if not available_cells_for_type: continue
            for _ in range(num_ues_for_type):
                 if len(available_cells_for_type) == 1: chosen_cell = available_cells_for_type[0]
                 else: chosen_cell_index = np.random.randint(0, len(available_cells_for_type)); chosen_cell = available_cells_for_type[chosen_cell_index]
                 try:
                     # Generate point (same logic as before)
                     polygon = Polygon(chosen_cell['bounds']); min_lon, min_lat, max_lon, max_lat = polygon.bounds
                     if not (min_lon < max_lon and min_lat < max_lat): continue
                     attempts = 0; max_attempts = 100
                     while attempts < max_attempts:
                         ue_lon = np.random.uniform(min_lon, max_lon); ue_lat = np.random.uniform(min_lat, max_lat)
                         point = Point(ue_lon, ue_lat)
                         if polygon.contains(point):
                             ue_data_list.append({"tick": tick, "ue_id": ue_id_counter_for_tick, COL_LAT: ue_lat, COL_LON: ue_lon, "space_type": space_type, "voronoi_cell_id": chosen_cell['cell_id']})
                             ue_id_counter_for_tick += 1; break
                         attempts += 1
                     # else: logger warning...
                 except Exception as e: logger.error(f"Error placing UE: {e}")
    # Final DataFrame (same as before)
    if not ue_data_list: return pd.DataFrame()
    final_ue_df = pd.DataFrame(ue_data_list)
    logger.info(f"Finished UE generation. Total points generated: {len(final_ue_df)}")
    return final_ue_df


# --- Modify run_traffic_simulation_and_analysis ---
def run_traffic_simulation_and_analysis(
    site_config_data: pd.DataFrame,
    initial_config_data: pd.DataFrame,
    spatial_cells: List[Dict], # <<< Accept pre-generated spatial cells
    num_ues: int,
    spatial_params_path: str, # Still need this for type names potentially
    time_params_path: str,
) -> Dict:
    """Uses pre-generated spatial_cells for UE generation."""
    logger.info("--- Starting Traffic Simulation and Analysis (using fixed spatial layout) ---")
    # Load time params (spatial params only needed for reference/plotting now)
    try:
        with open(spatial_params_path, "r") as f: spatial_params = json.load(f) # Load for reference
        with open(time_params_path, "r") as f: time_params = json.load(f)
    except Exception as e: logger.error(f"Error loading JSON config: {e}"); raise

    # --- Generate UE Locations using the provided spatial_cells ---
    trafficload_ue_data = _city_digitaltwin_generator(
        total_ue=num_ues,
        # spatial_params_for_city is implicitly defined by the types in spatial_cells now
        time_params=time_params,
        spatial_cells=spatial_cells # <<< Pass the fixed layout
    )
    if "error" in trafficload_ue_data or trafficload_ue_data.empty: # Check DataFrame directly
        logger.error("UE generation failed within run function.")
        return {"error": "UE generation failed"}

    # --- Perform Basic RF Simulation and Analysis (Remains the same) ---
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


    logger.info("--- Traffic Simulation and Analysis Finished ---")
    return {
        "trafficload_ue_data": trafficload_ue_data, "ue_rxpower_data": ue_rxpower_data,
        "serving_cell_data": serving_cell_data, "trafficload_metric_per_tick": traffic_metric,
        "energyload_metric_per_tick": energy_metric, "dummy_training_data": dummy_training_data,
        "spatial_cells": spatial_cells # Optional: return the layout used
    }

# --- Modify plot function ---
def plot(tick_to_plot: int, results: Dict, site_config_data: pd.DataFrame,
         spatial_params: Dict, spatial_cells: List[Dict]): # <<< Accept spatial_cells
    """Visualizes using pre-generated spatial_cells."""
    logger.info(f"Generating plot for tick {tick_to_plot} using fixed spatial layout...")
    # ... (Checks remain the same) ...
    # ... (Get UE data, serving cell data for tick) ...
    # ... (Setup plot figure) ...

    # --- Get Colors for Spatial Types ---
    space_types_defined = spatial_params.get("types", [])
    if not space_types_defined: logger.warning("No 'types' in spatial_params for plotting.")
    cmap_spaces = plt.get_cmap('tab10', len(space_types_defined))
    space_colors = {stype: cmap_spaces(i) for i, stype in enumerate(space_types_defined)}

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

    # --- (Rest of plotting: Voronoi, Towers, UEs, Legend, Save) ---
    # --- (No changes needed here) ---
    # ...

# --- Modify Main Execution Block ---
if __name__ == "__main__":
    # ... (Configuration as before) ...
    GENERATE_TOPOLOGY = True; GENERATE_CONFIG = True # Example flags
    SITE_CONFIG_CSV = "./data/topology.csv"; CONFIG_CSV = "./data/config.csv"
    SPATIAL_PARAMS_JSON = "./spatial_params.json"; TIME_PARAMS_JSON = "./time_params.json"
    OUTPUT_DUMMY_TRAINING_CSV = "./dummy_ue_training_data.csv"
    OUTPUT_UE_DATA_DIR = "./ue_data"; NUM_UES_TO_GENERATE = 500
    # ...

    # --- Load/Generate Topology ---
    # ... (site_config_data loading/generation) ...
    # --- Load/Generate Initial Config ---
    # ... (initial_config_data loading/generation) ...
    # --- Create Dummy Spatial/Time JSONs if needed ---
    # ... (Dummy JSON creation) ...

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
        # try:
        #     with open("./generated_spatial_layout.json", "w") as f:
        #         # Convert numpy arrays in bounds to lists for JSON serialization
        #         serializable_layout = []
        #         for cell in spatial_cells_layout:
        #              serializable_layout.append({
        #                  "bounds": [[coord[0], coord[1]] for coord in cell["bounds"]], # Convert tuples/arrays
        #                  "type": cell["type"],
        #                  "cell_id": cell["cell_id"],
        #                  "original_voronoi_region": cell.get("original_voronoi_region")
        #              })
        #         json.dump(serializable_layout, f, indent=2)
        #     logger.info("Saved generated spatial layout to generated_spatial_layout.json")
        # except Exception as e:
        #     logger.error(f"Could not save spatial layout: {e}")
    except Exception as e:
        logger.error(f"Error during spatial layout generation step: {e}")
        sys.exit(1)


    # --- Run Simulation (Passing Fixed Layout) ---
    try:
        results = run_traffic_simulation_and_analysis(
            site_config_data=site_config_data,
            initial_config_data=initial_config_data,
            spatial_cells=spatial_cells_layout, # <<< Pass the fixed layout
            num_ues=NUM_UES_TO_GENERATE,
            spatial_params_path=SPATIAL_PARAMS_JSON, # Still pass for reference/plotting
            time_params_path=TIME_PARAMS_JSON,
        )
    # ... (Error handling) ...

    # --- Save Per-Tick UE Data ---
    # ... (Saving loop remains the same) ...

    # --- Save Dummy Training Data ---
    # ... (Saving logic remains the same) ...

    # --- Plot Results Per Tick (Passing Fixed Layout) ---
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
    # ... (Error handling for plotting) ...

    logger.info("Script finished.")
    
    ----------------------------------
    
    # In traffic_3.py

# --- Add Matplotlib Patch for legend ---
from matplotlib.patches import Patch

# --- Modify plot function ---
def plot(tick_to_plot: int, results: Dict, site_config_data: pd.DataFrame,
         spatial_params: Dict, spatial_cells: List[Dict]): # <<< Accepts spatial_cells
    """
    Visualizes Voronoi, cells, UEs (colored by serving cell), and shaded
    spatial type areas for a specific tick using pre-generated spatial_cells.
    """
    logger.info(f"Generating plot for tick {tick_to_plot} using fixed spatial layout...")
    # --- Basic Checks ---
    if Voronoi is None: logger.warning("Scipy (Voronoi) missing.")
    if not all(k in results for k in ["trafficload_ue_data", "serving_cell_data"]): logger.warning("Missing data for plot."); return

    # --- Get Data for Tick ---
    ue_data_tick = results["trafficload_ue_data"][results["trafficload_ue_data"]["tick"] == tick_to_plot]
    serving_cell_tick = results["serving_cell_data"][results["serving_cell_data"]["tick"] == tick_to_plot]
    if ue_data_tick.empty: logger.warning(f"No UE data for tick {tick_to_plot}."); return
    ue_data_tick = pd.merge(ue_data_tick, serving_cell_tick[["ue_id", "serving_cell_id"]], on="ue_id", how="left")

    # --- Constants ---
    COL_LAT=getattr(c,'LAT','lat'); COL_LON=getattr(c,'LON','lon')
    COL_CELL_LAT=getattr(c,'CELL_LAT','cell_lat'); COL_CELL_LON=getattr(c,'CELL_LON','cell_lon')
    COL_CELL_ID=getattr(c,'CELL_ID','cell_id')

    # --- Create Plot Figure ---
    fig, ax = plt.subplots(figsize=(14, 11))

    # --- *** REMOVED REGENERATION BLOCK *** ---
    # # --- 1. Regenerate Spatial Cells for Shading ---
    # spatial_cells = [] # <<< REMOVE/COMMENT OUT
    # space_types_defined = spatial_params.get("types", [])
    # if not space_types_defined:
    #     # ...
    # else:
    #     try:
    #         # ... calls to space_boundary, _space_component_list_generator, _create_spatial_cells ...
    #     except Exception as e:
    #         # ...
    # --- *** END REMOVED BLOCK *** ---

    # --- Define Colors for Spatial Types ---
    space_types_defined = spatial_params.get("types", []) # Still need types for colors/labels
    if not space_types_defined: logger.warning("No 'types' in spatial_params for plotting legend/colors.")
    cmap_spaces = plt.get_cmap('tab10', len(space_types_defined))
    space_colors = {stype: cmap_spaces(i) for i, stype in enumerate(space_types_defined)}

    # --- Plot Shaded Spatial Areas (Use passed spatial_cells) ---
    plotted_space_labels = set()
    spatial_legend_handles = []
    if spatial_cells: # Use the passed list directly
        logger.debug(f"Plotting {len(spatial_cells)} pre-generated spatial cell areas...")
        for cell_info in spatial_cells: # Iterate through the FIXED layout
            try:
                polygon = Polygon(cell_info["bounds"]); space_type = cell_info["type"]
                color = space_colors.get(space_type, 'lightgrey')
                ax.fill(*polygon.exterior.xy, color=color, alpha=0.15, zorder=1)
                # Create legend handle only once per type found in the fixed layout
                if space_type not in plotted_space_labels:
                     label_text = space_type.replace("_"," ").title()
                     spatial_legend_handles.append(Patch(facecolor=color, alpha=0.3, label=label_text))
                     plotted_space_labels.add(space_type)
            except Exception as e: logger.warning(f"Could not plot spatial cell {cell_info.get('cell_id', 'N/A')}: {e}")
    else:
        logger.warning("No spatial cells provided to plot function for background shading.")

    # --- (Rest of plotting: Voronoi, Towers, UEs, Legend, Save - NO CHANGES NEEDED HERE) ---
    # ...