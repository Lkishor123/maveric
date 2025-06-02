# energy_saving_demo_app.py

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Tuple # Ensure Any, Optional, Tuple are imported

import pandas as pd
import numpy as np
import torch # For loading saved model states
import gpytorch # For GPyTorch models

# --- Path Setup (Ensure this points to your Maveric project root) ---
RADP_ROOT = os.getenv("MAVERIC_ROOT", "/path/to/your/maveric/project") # MODIFY IF NEEDED
if not os.path.isdir(RADP_ROOT):
     potential_path = os.path.join(os.path.dirname(__file__), "..", "..")
     if os.path.isdir(os.path.join(potential_path, "radp")): RADP_ROOT = os.path.abspath(potential_path); print(f"Warning: RADP_ROOT assumed: {RADP_ROOT}")
     else: raise FileNotFoundError(f"RADP_ROOT directory not found: {RADP_ROOT}.")
sys.path.insert(0, RADP_ROOT)
sys.path.insert(0, os.path.join(RADP_ROOT, "apps")) # To find energy_savings_gym if it's there

# --- RADP and CCO Imports ---
try:
    from radp.client.client import RADPClient
    from radp.client.helper import RADPHelper, ModelStatus
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.utils.gis_tools import GISTools
    from radp.digital_twin.mobility.ue_tracks import UETracksGenerator # For the Gym
    # Import necessary classes from bayesian_engine
    from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin, ExactGPModel, NormMethod
    from coverage_capacity_optimization.cco_engine import CcoEngine
    # Import the Gym environment
    from energy_savings_gym import EnergySavingsGym # Assuming it's in energy_savings_gym.py
    logger = logging.getLogger(__name__) # Define logger after importing logging
    logger.info("Successfully imported RADP, CCO, and EnergySavingsGym modules.")
except ImportError as e:
    print(f"FATAL: Error importing modules: {e}. Check paths and dependencies.")
    sys.exit(1)
except NameError as e: # Handle logger not defined if logging import failed for some reason
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.error(f"Caught NameError, possibly during logger init, then retried. Original error: {e}")


# --- Configuration ---
# Paths to data files (assuming they are in a 'data' subdirectory)
DATA_DIR = "./data" # Relative to where this script is run
TOPOLOGY_FILE_PATH = os.path.join(DATA_DIR, "topology.csv") # Your 30-cell topology
# **IMPORTANT**: Use REALISTIC training data for this BDT model
TRAINING_DATA_CSV_PATH = os.path.join(DATA_DIR, "realistic_ue_training_data_30cell.csv")
CONFIG_FILE_PATH = os.path.join(DATA_DIR, "config.csv") # Initial config for tilts

# Model ID for the BDT trained on the backend
BDT_MODEL_ID = "bdt_for_energy_savings_v1"
# Path where the backend saves the trained BDT model (state and metadata)
# This path is RELATIVE TO THE BACKEND SERVICE's filesystem (`/srv/radp/models/...`)
BACKEND_MODEL_SAVE_PATH = f"/srv/radp/models/{BDT_MODEL_ID}/model.pk" # .pk or .pth

# Parameters for EnergySavingsGym
TILT_SET = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
WEAK_COVERAGE_THRESHOLD = -95.0  # dBm
OVER_COVERAGE_THRESHOLD = -65.0 # dBm (RSRP for overshooting, SINR usually for interference)
LAMBDA_WEIGHT = 0.5 # Weight for energy vs CCO metric in reward

# Parameters for dummy UE data for the Gym's prediction_frame_template (if not using UETracksGenerator)
NUM_DUMMY_UE_POINTS_PER_CELL_FOR_GYM = 100 # Fewer points for faster demo

# Parameters for UETracksGenerator (if used by Gym)
USE_UE_TRACK_GENERATOR = True # Set to False to use static prediction_frame_template
GYM_HORIZON = 24 # Number of steps in one Gym episode (e.g., 24 hours)


# --- BDT Model Feature Configuration (Must match what the backend model was trained with) ---
# These are the columns the BayesianDigitalTwin's GPyTorch model was trained on (after preprocessing)
# Example from bayesian_engine.py's create_prediction_frames
BDT_X_COLUMNS = [
    getattr(c, 'CELL_LAT', 'cell_lat'),
    getattr(c, 'CELL_LON', 'cell_lon'),
    getattr(c, 'CELL_EL_DEG', 'cell_el_deg'),
    getattr(c, 'LOG_DISTANCE', 'log_distance'),
    getattr(c, 'RELATIVE_BEARING', 'relative_bearing'),
    getattr(c, 'ANTENNA_GAIN', 'antenna_gain'), # Assuming this feature is used
]
# The target variable the BDT predicts
BDT_Y_COLUMNS = [getattr(c, 'RXPOWER_DBM', 'rxpower_dbm')]


class GymBDTWrapper:
    """
    A wrapper to provide the predict_distributed_gpmodel interface
    for a single cell using loaded GPyTorch state and metadata.
    """
    def __init__(self, gp_model_state: Dict, likelihood_state: Dict, metadata: Dict):
        self.gp_model_state = gp_model_state
        self.likelihood_state = likelihood_state
        self.metadata = metadata
        self.epsilon = torch.finfo(torch.float32).eps

        # Reconstruct model and likelihood (once per wrapper instance)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([1]))
        # Use dummy tensors of correct feature size for initialization
        num_features = self.metadata['num_features']
        dummy_train_x = torch.zeros(1, 1, num_features)
        dummy_train_y = torch.zeros(1, 1)
        self.model = ExactGPModel(dummy_train_x, dummy_train_y, self.likelihood)

        self.model.load_state_dict(self.gp_model_state)
        self.likelihood.load_state_dict(self.likelihood_state) # Or model handles this

        self.is_cuda = self.metadata.get('is_cuda', False) and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda else "cpu")
        self.model = self.model.to(self.device)
        # self.likelihood = self.likelihood.to(self.device) # Often moved with model

        self.model.eval()
        self.likelihood.eval()

    def predict_distributed_gpmodel(self, prediction_dfs: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs prediction for a single cell's data.
        prediction_dfs is expected to be a list containing ONE DataFrame for this cell.
        """
        if not prediction_dfs or prediction_dfs[0].empty:
            logger.warning(f"Cell {self.metadata.get('cell_id','N/A')}: Empty prediction_df in predict_distributed_gpmodel.")
            return np.array([]), np.array([])

        ue_prediction_data = prediction_dfs[0] # Get the single DataFrame

        # Normalize input features
        x_cols = self.metadata['x_columns']
        norm_method = NormMethod[self.metadata.get('norm_method', 'MINMAX')]
        # Convert metadata stats to pd.Series for easier broadcasting
        xmin = pd.Series(self.metadata['xmin']); xmax = pd.Series(self.metadata['xmax'])
        xmeans = pd.Series(self.metadata['xmeans']); xstds = pd.Series(self.metadata['xstds'])

        if norm_method == NormMethod.MINMAX:
            range_x = xmax - xmin
            predict_x_normalized = (ue_prediction_data[x_cols] - xmin) / (range_x + self.epsilon)
        elif norm_method == NormMethod.ZSCORE:
            predict_x_normalized = (ue_prediction_data[x_cols] - xmeans) / (xstds + self.epsilon)
        else:
            raise ValueError(f"Unknown norm method: {norm_method}")

        predict_X_tensor = torch.tensor(predict_x_normalized.values, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(predict_X_tensor))
            mean_normalized = observed_pred.mean.squeeze(0).cpu().numpy()
            variance_normalized = observed_pred.variance.squeeze(0).cpu().numpy()

        # Denormalize
        ymeans = pd.Series(self.metadata['ymeans']); ystds = pd.Series(self.metadata['ystds'])
        # Assuming single y_column for simplicity
        y_mean_val = ymeans.iloc[0]; y_std_val = ystds.iloc[0]

        pred_means_denorm = mean_normalized * (y_std_val + self.epsilon) + y_mean_val
        pred_stds_denorm = np.sqrt(variance_normalized) * (y_std_val + self.epsilon)

        # The Gym's _next_observation updates the original prediction_dfs[cell_id]
        # So, we need to make sure this matches what it expects:
        # It does "self.prediction_dfs[cell_id][constants.RXPOWER_DBM] = pred_means"
        # The original BDT.predict_distributed_gpmodel also mutated the input dfs list.
        # This wrapper is for one cell, so we update the input df.
        COL_RXPOWER_DBM = getattr(c, 'RXPOWER_DBM', 'rxpower_dbm')
        COL_RXPOWER_STDDEV_DBM = getattr(c, 'RXPOWER_STDDEV_DBM', 'rxpower_stddev_dbm')
        ue_prediction_data.loc[:, COL_RXPOWER_DBM] = pred_means_denorm
        ue_prediction_data.loc[:, COL_RXPOWER_STDDEV_DBM] = pred_stds_denorm

        return pred_means_denorm, pred_stds_denorm # Return what Gym might expect, though it uses df


def main():
    logger.info("--- Energy Saving Demo App ---")

    # --- 1. Initialize RADP Client ---
    try:
        radp_client = RADPClient()
        radp_helper = RADPHelper(radp_client)
        logger.info("RADP Client and Helper initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize RADP Client/Helper: {e}")
        return

    # --- 2. Train BDT Model (or ensure it's pre-trained) ---
    # For this demo, we assume the BDT is already trained or we trigger training here.
    # If re-training, ensure REALISTIC ue_training_data.csv is used.
    try:
        logger.info(f"Loading topology from: {TOPOLOGY_FILE_PATH}")
        topology_df = pd.read_csv(TOPOLOGY_FILE_PATH)
        logger.info(f"Loading training data from: {TRAINING_DATA_CSV_PATH}")
        ue_training_df = pd.read_csv(TRAINING_DATA_CSV_PATH) # Use realistic data

        logger.info(f"Requesting training for BDT model: {BDT_MODEL_ID}")
        train_response = radp_client.train(
            model_id=BDT_MODEL_ID,
            params={}, # Add specific training params if needed
            ue_training_data=ue_training_df,
            topology=topology_df
        )
        logger.info(f"Train request sent. Response: {train_response}")
        status = radp_helper.resolve_model_status(BDT_MODEL_ID, wait_interval=10, max_attempts=180, verbose=True) # Wait up to 30 mins

        if not status.success:
            logger.error(f"BDT Model training failed for {BDT_MODEL_ID}: {status.error_message}")
            return
        logger.info(f"BDT Model '{BDT_MODEL_ID}' training complete and available.")

    except FileNotFoundError as e:
        logger.error(f"Required data file not found: {e}. Cannot proceed.")
        return
    except Exception as e:
        logger.error(f"Error during BDT training phase: {e}")
        return

    # --- 3. Load the Trained BDT Model (States and Metadata) ---
    logger.info(f"Loading trained BDT model states from backend path: {BACKEND_MODEL_SAVE_PATH}")
    try:
        # This path is where the *backend training service* saved the model.
        # Your local script needs a way to access this, or you manually copy it.
        # For now, let's assume it's accessible at a local equivalent path if using shared volumes,
        # OR the BayesianDigitalTwin.load_models_from_state can fetch it via client.
        # For this demo, we assume load_models_from_state handles it or file is locally available.
        # If load_models_from_state needs a local path, ensure it's correct.
        # Example: LOCAL_MODEL_SAVE_PATH = os.path.join(DATA_DIR, f"{BDT_MODEL_ID}_model_state.pth")
        # Ensure model was saved with BayesianDigitalTwin.save_model_state_and_metadata
        # And that it is locally accessible if load_models_from_state expects a local path.
        # **This is a key potential point of failure if paths/access aren't right.**
        # Let's assume it's downloaded/accessible at `LOCAL_MODEL_SAVE_PATH`
        LOCAL_MODEL_SAVE_PATH = os.path.join(DATA_DIR, BDT_MODEL_ID, "model.pk") # Path convention from backend
        if not os.path.exists(LOCAL_MODEL_SAVE_PATH):
            logger.error(f"Model file not found at {LOCAL_MODEL_SAVE_PATH}. Please ensure it's downloaded from backend or path is correct.")
            logger.error("The backend saves to BACKEND_MODEL_SAVE_PATH. This demo expects it to be accessible locally.")
            return

        loaded_bdt_states_map = BayesianDigitalTwin.load_models_from_state(LOCAL_MODEL_SAVE_PATH)
        if not loaded_bdt_states_map:
            logger.error("Failed to load BDT model states.")
            return
        logger.info(f"Successfully loaded BDT model states for {len(loaded_bdt_states_map)} cells.")

        # Prepare the dictionary of GymBDTWrapper instances for the Gym
        gym_bdt_dict = {}
        for cell_id_str, cell_data in loaded_bdt_states_map.items():
            gym_bdt_dict[cell_id_str] = GymBDTWrapper(
                gp_model_state=cell_data['gp_model_state'],
                likelihood_state=cell_data['likelihood_state'],
                metadata=cell_data['metadata']
            )

    except Exception as e:
        logger.exception(f"Error loading/preparing BDT model for Gym: {e}")
        return

    # --- 4. Prepare Inputs for EnergySavingsGym ---
    logger.info("Preparing inputs for EnergySavingsGym...")
    site_config_df = topology_df.copy() # Gym expects this name
    
    # Create prediction_frame_template OR UETracksGenerator
    prediction_frame_template_gym = {}
    ue_track_generator_gym = None

    if USE_UE_TRACK_GENERATOR:
        logger.info("Using UETracksGenerator for dynamic UE locations in Gym.")
        try:
            # Derive bounds from topology
            min_lat_env = site_config_df[c.CELL_LAT].min() - 0.01 # Small buffer
            max_lat_env = site_config_df[c.CELL_LAT].max() + 0.01
            min_lon_env = site_config_df[c.CELL_LON].min() - 0.01
            max_lon_env = site_config_df[c.CELL_LON].max() + 0.01
            # These are example parameters for UETracksGenerator
            ue_track_generator_gym = UETracksGenerator(
                n_ues=100, # Number of UEs for Gym steps (can be different from traffic script)
                min_lat=min_lat_env, max_lat=max_lat_env,
                min_lon=min_lon_env, max_lon=max_lon_env,
                x_dim=50, y_dim=50, # Grid dimensions for UE generation
                min_wait_time=1, max_wait_time=5, # Ticks
                min_speed=1, max_speed=5, # Grid cells per tick
                seed=42
            )
        except Exception as e:
            logger.error(f"Failed to initialize UETracksGenerator: {e}. Will try static template.")
            USE_UE_TRACK_GENERATOR = False # Fallback

    if not USE_UE_TRACK_GENERATOR:
        logger.info("Using static prediction_frame_template for Gym.")
        # Create a simple grid of UE points for each cell's prediction frame template
        # This is used by the Gym if ue_track_generator is None
        # The Gym's BayesianDigitalTwin.create_prediction_frames will add cell-specific features
        num_pts_side = int(math.sqrt(NUM_DUMMY_UE_POINTS_PER_CELL_FOR_GYM))
        lons = np.linspace(site_config_df[c.CELL_LON].min(), site_config_df[c.CELL_LON].max(), num_pts_side)
        lats = np.linspace(site_config_df[c.CELL_LAT].min(), site_config_df[c.CELL_LAT].max(), num_pts_side)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        base_ue_locs = pd.DataFrame({
            getattr(c, 'LOC_X', 'loc_x'): lon_grid.ravel(),
            getattr(c, 'LOC_Y', 'loc_y'): lat_grid.ravel()
        })
        for cell_id_val in site_config_df[c.CELL_ID].unique():
            prediction_frame_template_gym[cell_id_val] = base_ue_locs.copy()

    # Example static traffic model (if needed by Gym reward and not None)
    # This should ideally come from your traffic data analysis or a more complex model
    traffic_model_df_gym = None
    # if os.path.exists(STATIC_TRAFFIC_MODEL_PATH):
    #     traffic_model_df_gym = pd.read_csv(STATIC_TRAFFIC_MODEL_PATH)
    #     logger.info("Loaded static traffic model for Gym.")

    # --- 5. Instantiate EnergySavingsGym ---
    logger.info("Instantiating EnergySavingsGym...")
    try:
        env = EnergySavingsGym(
            bayesian_digital_twins=gym_bdt_dict, # Pass the dictionary of wrappers
            site_config_df=site_config_df,
            prediction_frame_template=prediction_frame_template_gym, # Used if ue_track_generator is None
            tilt_set=TILT_SET,
            weak_coverage_threshold=WEAK_COVERAGE_THRESHOLD,
            over_coverage_threshold=OVER_COVERAGE_THRESHOLD,
            lambda_=LAMBDA_WEIGHT,
            traffic_model_df=traffic_model_df_gym, # Pass None or loaded DataFrame
            ue_track_generator=ue_track_generator_gym if USE_UE_TRACK_GENERATOR else None,
            horizon=GYM_HORIZON,
            debug=True # Set to False for less verbose output from Gym
        )
        logger.info("EnergySavingsGym instantiated successfully.")
    except Exception as e:
        logger.exception(f"Error instantiating EnergySavingsGym: {e}")
        return

    # --- 6. Demo Interaction with the Gym ---
    logger.info("--- Running Demo Interaction with Gym ---")
    try:
        observation = env.reset() # Get initial observation
        logger.info(f"Initial Observation: {observation}")
        total_reward_accumulated = 0
        for step in range(GYM_HORIZON + 5): # Run for a bit longer than horizon
            action = env.action_space.sample() # Take a random action
            logger.info(f"\nStep {step + 1}/{GYM_HORIZON}")
            logger.info(f"Taking Action: {action}")

            observation, reward, done, info = env.step(action) # Get new state and reward

            logger.info(f"Observation: {observation}")
            logger.info(f"Reward: {reward:.4f}")
            logger.info(f"Info: {info}") # Contains debug info from Gym step
            total_reward_accumulated += reward

            if done:
                logger.info(f"Episode finished after {step + 1} steps.")
                break
        logger.info(f"Demo finished. Total accumulated reward: {total_reward_accumulated:.4f}")

    except Exception as e:
        logger.exception(f"Error during Gym interaction demo: {e}")
    finally:
        env.close()


if __name__ == "__main__":
    # Ensure data directory exists for dummy files if script needs to create them
    os.makedirs(DATA_DIR, exist_ok=True)
    # TODO: Add logic to create dummy topology.csv, realistic_ue_training_data_30cell.csv,
    # config.csv, and potentially a dummy model.pk if they don't exist, for this demo script to run.
    # For now, assuming these files are manually prepared or generated by traffic_3.py first.

    if not os.path.exists(TOPOLOGY_FILE_PATH):
        logger.error(f"Missing topology file: {TOPOLOGY_FILE_PATH}. Please create it.")
        sys.exit(1)
    if not os.path.exists(TRAINING_DATA_CSV_PATH):
        logger.error(f"Missing REALISTIC UE training data file: {TRAINING_DATA_CSV_PATH}. Please create it.")
        sys.exit(1)

    main()