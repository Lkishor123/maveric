import os
import pandas as pd
import logging
import glob

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UEDataPreprocessor:
    """
    Preprocesses raw UE data CSVs to be compatible with the EnergySavingGym.
    Specifically, it renames longitude and latitude columns.
    """
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initializes the UEDataPreprocessor.

        Args:
            input_dir (str): Directory containing the raw per-tick UE data files.
            output_dir (str): Directory where the processed files will be saved.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        # Expected column names from traffic_demand_simulation.py output
        self.col_lon_input = "lon"
        self.col_lat_input = "lat"
        # Target column names for EnergySavingsGym's prediction_frame_template
        self.col_loc_x_output = "loc_x"  # Often used for longitude in RADP
        self.col_loc_y_output = "loc_y"  # Often used for latitude in RADP

    def run(self):
        """
        Reads all 'generated_ue_data_for_cco_*.csv' files from the input directory,
        renames longitude and latitude columns, and saves them to the output directory.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        input_file_pattern = os.path.join(self.input_dir, "generated_ue_data_for_cco_*.csv")
        ue_csv_files = glob.glob(input_file_pattern)

        if not ue_csv_files:
            logger.error(f"No UE data CSV files found in '{self.input_dir}' matching the pattern.")
            return

        logger.info(f"Found {len(ue_csv_files)} UE data files to process from '{self.input_dir}'.")
        processed_count = 0

        for filepath in ue_csv_files:
            try:
                filename = os.path.basename(filepath)
                logger.debug(f"Processing file: {filename}")

                df = pd.read_csv(filepath)

                if self.col_lon_input not in df.columns or self.col_lat_input not in df.columns:
                    logger.warning(f"Skipping {filename}: Missing '{self.col_lon_input}' or '{self.col_lat_input}' column.")
                    continue

                df.rename(columns={
                    self.col_lon_input: self.col_loc_x_output,
                    self.col_lat_input: self.col_loc_y_output
                }, inplace=True)

                output_filepath = os.path.join(self.output_dir, filename)
                df.to_csv(output_filepath, index=False)
                logger.info(f"Successfully processed and saved: {output_filepath}")
                processed_count += 1

            except Exception as e:
                logger.error(f"Error processing file {filepath}: {e}")

        logger.info(f"Finished preprocessing. Processed {processed_count}/{len(ue_csv_files)} files.")

