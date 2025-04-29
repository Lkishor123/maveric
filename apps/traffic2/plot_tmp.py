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