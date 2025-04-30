# Inside run_traffic_simulation_and_analysis function in traffic_3.py

    # --- *** GENERATE DUMMY TRAINING DATA (Corrected Merge/Rename) *** ---
    logger.warning("Generating DUMMY training data - RSRP values DO NOT accurately reflect tilt effects.")
    dummy_training_data = pd.DataFrame() # Initialize empty
    try:
        if serving_cell_data.empty or ue_rxpower_data.empty:
             logger.warning("Cannot generate dummy training data: Missing serving cell or RxPower data.")
        else:
            # 1. Get RSRP from the serving cell (based on simple model)
            # Ensure ue_rxpower_data has the correct CELL_ID column name (e.g., 'cell_id')
            serving_cell_power = pd.merge(
                 serving_cell_data, # cols: tick, ue_id, serving_cell_id
                 ue_rxpower_data,   # cols: tick, ue_id, cell_id, rx_power_dbm
                 # Match the serving_cell_id from left df with cell_id from right df
                 left_on=["tick", "ue_id", "serving_cell_id"],
                 right_on=["tick", "ue_id", COL_CELL_ID],
                 how="left"
            )
            # Select only needed columns and drop the redundant cell_id from ue_rxpower_data
            serving_cell_power = serving_cell_power[["tick", "ue_id", "serving_cell_id", "rx_power_dbm"]]

            # 2. Merge with UE locations to get lat/lon
            ue_locs_with_serving_rsrp = pd.merge(
                trafficload_ue_data[[COL_LAT, COL_LON, "tick", "ue_id"]],
                serving_cell_power,
                on=["tick", "ue_id"],
                how="inner" # Only keep UEs that had a valid serving cell RSRP
            )

            # 3. Merge with initial config tilts
            if COL_CELL_ID not in initial_config_data.columns or COL_CELL_EL_DEG not in initial_config_data.columns:
                raise ValueError(f"Initial config data must have '{COL_CELL_ID}' and '{COL_CELL_EL_DEG}' columns.")

            # *** Merge tilt based on the serving cell ID ***
            training_data_merged = pd.merge(
                 ue_locs_with_serving_rsrp,
                 initial_config_data[[COL_CELL_ID, COL_CELL_EL_DEG]],
                 left_on="serving_cell_id", # Key from the left DataFrame (UE data with serving cell)
                 right_on=COL_CELL_ID,      # Key from the right DataFrame (initial config)
                 how="left",
                 # Suffixes avoid column name collision if keys were same, but we handle manually
                 # suffixes=('', '_config')
            )
            # Drop the redundant key column from the right dataframe after merge
            training_data_merged = training_data_merged.drop(columns=[COL_CELL_ID])


            if training_data_merged[COL_CELL_EL_DEG].isnull().any():
                 missing_tilt_cells = training_data_merged[training_data_merged[COL_CELL_EL_DEG].isnull()]['serving_cell_id'].unique()
                 logger.warning(f"Could not find initial tilt for serving cells: {missing_tilt_cells}. Rows will have NaN tilt.")


            # 4. Select and rename columns to final format
            dummy_training_data = training_data_merged.rename(columns={
                 "serving_cell_id": COL_CELL_ID, # <<< Now rename the correct column to 'cell_id'
                 "rx_power_dbm": "avg_rsrp",
                 COL_LON: "lon",
                 COL_LAT: "lat",
                 # COL_CELL_EL_DEG name is already correct
            })

            # 5. Select final columns in desired order
            final_columns = [COL_CELL_ID, "avg_rsrp", "lon", "lat", COL_CELL_EL_DEG]
            # Ensure all columns exist before selection
            missing_final_cols = [col for col in final_columns if col not in dummy_training_data.columns]
            if missing_final_cols:
                raise ValueError(f"DataFrame missing expected final columns: {missing_final_cols}")

            dummy_training_data = dummy_training_data[final_columns]
            logger.info(f"Generated dummy training data format with {len(dummy_training_data)} rows (includes all ticks).")

    except Exception as e:
        logger.exception(f"Failed to generate dummy training data: {e}")
        dummy_training_data = pd.DataFrame()

    # ... (rest of function, including the return statement) ...
    return {
        # ... other results ...
        "dummy_training_data": dummy_training_data
    }