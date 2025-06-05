# preprocess_ue_for_gym.py
# Renames columns in per-tick UE data CSVs to be compatible with EnergySavingsGym's
# expectation for prediction frame templates (loc_x, loc_y).

import os
import pandas as pd
import logging
import glob # For finding files

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Directory where traffic_demand_simulation.py saved its per-tick UE data
INPUT_UE_DATA_DIR = "./data/ue_data_per_tick" # From traffic_demand_simulation.py
# Directory to save the processed files for the Gym
OUTPUT_GYM_UE_DATA_DIR = "./data/ue_data_gym_ready"

# Expected column names from traffic_demand_simulation.py output
COL_LON_INPUT = "lon"
COL_LAT_INPUT = "lat"
# Target column names for EnergySavingsGym's prediction_frame_template
COL_LOC_X_OUTPUT = "loc_x" # Often used for longitude in RADP
COL_LOC_Y_OUTPUT = "loc_y" # Often used for latitude in RADP


def preprocess_ue_files(input_dir: str, output_dir: str):
    """
    Reads all generated_ue_data_for_cco_{tick}.csv files from input_dir,
    renames longitude and latitude columns, and saves them to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    input_file_pattern = os.path.join(input_dir, "generated_ue_data_for_cco_*.csv")
    ue_csv_files = glob.glob(input_file_pattern)

    if not ue_csv_files:
        logger.error(f"No UE data CSV files found in '{input_dir}' matching pattern.")
        return

    logger.info(f"Found {len(ue_csv_files)} UE data files to process from '{input_dir}'.")
    processed_count = 0

    for filepath in ue_csv_files:
        try:
            filename = os.path.basename(filepath)
            logger.debug(f"Processing file: {filename}")
            
            df = pd.read_csv(filepath)

            # Check for necessary columns
            if COL_LON_INPUT not in df.columns or COL_LAT_INPUT not in df.columns:
                logger.warning(f"Skipping {filename}: Missing '{COL_LON_INPUT}' or '{COL_LAT_INPUT}' column.")
                continue

            # Rename columns
            df.rename(columns={
                COL_LON_INPUT: COL_LOC_X_OUTPUT,
                COL_LAT_INPUT: COL_LOC_Y_OUTPUT
            }, inplace=True)
            
            # Ensure all original columns are kept if needed, or select specific ones
            # For EnergySavingsGym's prediction_frame_template, only loc_x, loc_y are strictly needed
            # from the base template, but keeping others doesn't hurt.
            # The Gym's create_prediction_frames will then add cell-specific features.
            
            output_filepath = os.path.join(output_dir, filename) # Keep original filename
            df.to_csv(output_filepath, index=False)
            logger.info(f"Successfully processed and saved: {output_filepath}")
            processed_count += 1

        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}")
            
    logger.info(f"Finished preprocessing. Processed {processed_count}/{len(ue_csv_files)} files.")

if __name__ == "__main__":
    logger.info("--- Starting UE Data Preprocessing for EnergySavingsGym ---")
    preprocess_ue_files(INPUT_UE_DATA_DIR, OUTPUT_GYM_UE_DATA_DIR)
    logger.info(f"--- Preprocessing Complete. Check '{OUTPUT_GYM_UE_DATA_DIR}' ---")
