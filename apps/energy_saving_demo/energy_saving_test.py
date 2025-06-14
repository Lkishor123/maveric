# energy_saving_demo_app.py

import os
import sys
import json
import logging
import math
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
import torch
import gpytorch

# --- Path Setup ---
RADP_ROOT = os.getenv("MAVERIC_ROOT", "/home/lk/Projects/accelcq-repos/cloudly/github/maveric/") # MODIFY IF NEEDED
if not os.path.isdir(RADP_ROOT):
    potential_path = os.path.join(os.path.dirname(__file__), "..", "..") # Adjust if script is deeper
    if os.path.isdir(os.path.join(potential_path, "radp")):
        RADP_ROOT = os.path.abspath(potential_path)
        print(f"Warning: RADP_ROOT not explicitly set or found. Assuming relative path: {RADP_ROOT}")
    else:
        raise FileNotFoundError(f"RADP_ROOT directory not found: {RADP_ROOT}. Please set path.")
sys.path.insert(0, RADP_ROOT)
# Add current directory to path to find energy_savings_gym.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# --- RADP and Custom Imports ---
try:
    from radp.client.client import RADPClient
    from radp.client.helper import RADPHelper, ModelStatus
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.utils.gis_tools import GISTools
    from radp.digital_twin.mobility.ue_tracks import UETracksGenerator
    from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin, ExactGPModel, NormMethod
    # Assuming CcoEngine is in apps/coverage_capacity_optimization relative to RADP_ROOT
    sys.path.insert(0, os.path.join(RADP_ROOT, "apps"))
    from coverage_capacity_optimization.cco_engine import CcoEngine
    from apps.energy_savings.energy_savings_gym import EnergySavingsGym # From the file you provided
except ImportError as e:
    print(f"FATAL: Error importing modules: {e}. Check paths and dependencies.")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Configuration ---
DATA_DIR = "./data" # Assumes data is in a subdir relative to this script
TOPOLOGY_FILE_PATH = os.path.join(DATA_DIR, "topology.csv") # Your 30-cell or relevant topology
CONFIG_FILE_PATH = os.path.join(DATA_DIR, "config.csv")     # For initial tilts in Gym

# **IMPORTANT**: Use REALISTIC training data for the BDT model
TRAINING_DATA_CSV_PATH = os.path.join(DATA_DIR, "realistic_ue_training_data_30cell.csv") # REPLACE if needed

BDT_MODEL_ID = "bdt_for_es_demo_v2" # Choose a unique ID for this trained BDT
# Path where the backend training service saves the model (inside Docker)
BACKEND_MODEL_SAVE_PATH = f"/srv/radp/models/{BDT_MODEL_ID}/model.pk" # .pk or .pth (match bayesian_engine save)
# Path where this script will LOOK for the model file (ensure it's accessible)
LOCAL_MODEL_SAVE_PATH = os.path.join(DATA_DIR, BDT_MODEL_ID, "model.pk")

TILT_SET = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
WEAK_COVERAGE_THRESHOLD = -95.0
OVER_COVERAGE_THRESHOLD = -65.0 # Typically for RSRP
LAMBDA_WEIGHT_GYM = 0.5

USE_UE_TRACK_GENERATOR_GYM = True
GYM_HORIZON = 24
NUM_UES_FOR_GYM_STEP = 50 # Number of UEs per step if using UETracksGenerator

# BDT X_COLUMNS for prediction (must match those used/generated by preprocess_ue_prediction_data)
# These are generated by BayesianDigitalTwin.create_prediction_frames called in EnergySavingsGym
BDT_X_COLUMNS_FOR_GYM_PREDICTION = [
    getattr(c, 'CELL_LAT', 'cell_lat'), getattr(c, 'CELL_LON', 'cell_lon'),
    getattr(c, 'CELL_EL_DEG', 'cell_el_deg'), getattr(c, 'LOG_DISTANCE', 'log_distance'),
    getattr(c, 'RELATIVE_BEARING', 'relative_bearing'), getattr(c, 'ANTENNA_GAIN', 'antenna_gain')
]
# Y_COLUMN that the BDT predicts
BDT_Y_COLUMN_FOR_GYM_PREDICTION = [getattr(c, 'RXPOWER_DBM', 'rxpower_dbm')]


class GymBDTCellPredictor:
    """
    Wrapper for a single cell's GPyTorch model to be used by EnergySavingsGym.
    It loads state and metadata, and provides a prediction interface.
    """
    def __init__(self, gp_model_state: Dict, likelihood_state: Dict, metadata: Dict):
        self.metadata = metadata
        self.epsilon = torch.finfo(torch.float32).eps

        self.x_columns = self.metadata['x_columns']
        self.y_columns = self.metadata['y_columns'] # Should be list, e.g. ['rxpower_dbm']
        self.target_col_name = self.y_columns[0] if self.y_columns else getattr(c, 'RXPOWER_DBM', 'rxpower_dbm')

        self.norm_method = NormMethod[self.metadata.get('norm_method', 'MINMAX')]
        self.xmin = pd.Series(self.metadata['xmin'])
        self.xmax = pd.Series(self.metadata['xmax'])
        self.xmeans = pd.Series(self.metadata['xmeans'])
        self.xstds = pd.Series(self.metadata['xstds'])
        self.ymeans = pd.Series(self.metadata['ymeans'])
        self.ystds = pd.Series(self.metadata['ystds'])

        num_features = self.metadata['num_features']
        dummy_train_x = torch.zeros(1, 1, num_features) # batch_size=1, n_train=1 (placeholder)
        dummy_train_y = torch.zeros(1, 1)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([1]))
        self.model = ExactGPModel(dummy_train_x, dummy_train_y, self.likelihood)

        self.model.load_state_dict(gp_model_state)
        # Likelihood state is usually part of model.state_dict(), but loading separately can be safer
        self.likelihood.load_state_dict(likelihood_state)

        self.is_cuda = self.metadata.get('is_cuda', False) and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda else "cpu")
        self.model = self.model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)

        self.model.eval()
        self.likelihood.eval()
        logger.debug(f"GymBDTCellPredictor for cell {self.metadata.get('cell_id')} initialized on {self.device}.")

    def predict_distributed_gpmodel(self, prediction_dfs: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts for a single cell's data. Modifies the input DataFrame in the list."""
        if not prediction_dfs or prediction_dfs[0].empty:
            logger.warning(f"Cell {self.metadata.get('cell_id','N/A')}: Empty prediction_df.")
            return np.array([]), np.array([])

        ue_prediction_data = prediction_dfs[0] # List is expected to have one DataFrame
        predict_x_pd = ue_prediction_data[self.x_columns]

        if self.norm_method == NormMethod.MINMAX:
            range_x = self.xmax - self.xmin
            predict_x_normalized = (predict_x_pd - self.xmin) / (range_x + self.epsilon)
        elif self.norm_method == NormMethod.ZSCORE:
            predict_x_normalized = (predict_x_pd - self.xmeans) / (self.xstds + self.epsilon)
        else:
            raise ValueError(f"Unsupported normalization method: {self.norm_method}")

        predict_X_tensor = torch.tensor(predict_x_normalized.values, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(predict_X_tensor))
            mean_normalized = observed_pred.mean.squeeze(0).cpu().numpy()
            variance_normalized = observed_pred.variance.squeeze(0).cpu().numpy()

        y_mean_val = self.ymeans[self.target_col_name] # Use specific y_col name
        y_std_val = self.ystds[self.target_col_name]

        pred_means_denorm = mean_normalized * (y_std_val + self.epsilon) + y_mean_val
        pred_stds_denorm = np.sqrt(variance_normalized) * (y_std_val + self.epsilon)

        # Modify the DataFrame in the list, as original BDT predict method did
        ue_prediction_data.loc[:, getattr(c, 'RXPOWER_DBM', 'rxpower_dbm')] = pred_means_denorm
        ue_prediction_data.loc[:, getattr(c, 'RXPOWER_STDDEV_DBM', 'rxpower_stddev_dbm')] = pred_stds_denorm
        
        return pred_means_denorm, pred_stds_denorm


def main():
    logger.info("--- Energy Saving Demo Application ---")

    # --- 1. Initialize RADP Client ---
    try:
        radp_client = RADPClient()
        radp_helper = RADPHelper(radp_client)
        logger.info("RADP Client and Helper initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize RADP Client/Helper: {e}"); return

    # --- 2. Train BDT Model (or ensure it's pre-trained and accessible) ---
    try:
        logger.info(f"Loading topology for BDT training from: {TOPOLOGY_FILE_PATH}")
        topology_df = pd.read_csv(TOPOLOGY_FILE_PATH)
        logger.info(f"Loading REALISTIC UE training data from: {TRAINING_DATA_CSV_PATH}")
        ue_training_df_for_bdt = pd.read_csv(TRAINING_DATA_CSV_PATH)

        logger.info(f"Requesting training for BDT model ID: {BDT_MODEL_ID}")
        # Ensure model_update=False for a fresh train or if it's a new ID
        train_response = radp_client.train(
            model_id=BDT_MODEL_ID, params={}, model_update=False,
            ue_training_data=ue_training_df_for_bdt, topology=topology_df
        )
        logger.info(f"Train request sent. Response: {train_response}")
        status = radp_helper.resolve_model_status(BDT_MODEL_ID, wait_interval=15, max_attempts=240, verbose=True)

        if not status.success:
            logger.error(f"BDT Model training failed for '{BDT_MODEL_ID}': {status.error_message}"); return
        logger.info(f"BDT Model '{BDT_MODEL_ID}' training complete.")

    except FileNotFoundError as e: logger.error(f"Data file not found: {e}."); return
    except Exception as e: logger.exception(f"Error during BDT training phase: {e}"); return

    # --- 3. Load the Trained BDT Model (States and Metadata) ---
    logger.info(f"Attempting to load trained BDT model states from local path: {LOCAL_MODEL_SAVE_PATH}")
    # This step assumes the model file saved by the backend at BACKEND_MODEL_SAVE_PATH
    # has been made accessible locally at LOCAL_MODEL_SAVE_PATH.
    # You might need to copy it manually (e.g., docker cp) or use shared volumes.
    if not os.path.exists(LOCAL_MODEL_SAVE_PATH):
        logger.error(f"Model file not found: {LOCAL_MODEL_SAVE_PATH}. Please ensure it's copied from backend.")
        logger.error(f"(Backend expected to save at: {BACKEND_MODEL_SAVE_PATH})")
        return
    try:
        # Use the static method from BayesianDigitalTwin engine to load
        loaded_bdt_states_map = BayesianDigitalTwin.load_models_from_state(LOCAL_MODEL_SAVE_PATH)
        if not loaded_bdt_states_map: logger.error("Failed to load BDT model states."); return
        logger.info(f"Successfully loaded BDT states for {len(loaded_bdt_states_map)} cells.")

        # Prepare the dictionary of GymBDTCellPredictor instances for the Gym
        gym_bdt_predictors = {}
        for cell_id_str, cell_data in loaded_bdt_states_map.items():
            gym_bdt_predictors[cell_id_str] = GymBDTCellPredictor(
                gp_model_state=cell_data['gp_model_state'],
                likelihood_state=cell_data['likelihood_state'],
                metadata=cell_data['metadata']
            )
        logger.info(f"Created {len(gym_bdt_predictors)} cell predictors for Gym.")

    except Exception as e: logger.exception(f"Error loading/preparing BDT model for Gym: {e}"); return

    # --- 4. Prepare Inputs for EnergySavingsGym ---
    logger.info("Preparing inputs for EnergySavingsGym...")
    # The Gym expects site_config_df to have initial tilts. Load from config.csv
    try:
        site_config_df_gym = pd.read_csv(CONFIG_FILE_PATH)
        # Merge with topology if config only has cell_id, cell_el_deg
        if not all(col in site_config_df_gym.columns for col in [c.CELL_LAT, c.CELL_LON, c.CELL_AZ_DEG]):
            logger.info("Merging config with topology for full site_config_df for Gym...")
            site_config_df_gym = pd.merge(topology_df, site_config_df_gym, on=getattr(c,'CELL_ID','cell_id'), how='left')
            # Fill any missing tilts with a default if merge didn't cover all cells
            site_config_df_gym[getattr(c,'CELL_EL_DEG','cell_el_deg')].fillna(TILT_SET[len(TILT_SET)//2], inplace=True)
    except FileNotFoundError:
        logger.error(f"Config file not found: {CONFIG_FILE_PATH}. Using topology as base and default tilts.");
        site_config_df_gym = topology_df.copy()
        site_config_df_gym[getattr(c,'CELL_EL_DEG','cell_el_deg')] = TILT_SET[len(TILT_SET)//2] # Add default tilt if no config
    except Exception as e:
        logger.error(f"Error preparing site_config_df for Gym: {e}. Using basic topology.");
        site_config_df_gym = topology_df.copy()
        site_config_df_gym[getattr(c,'CELL_EL_DEG','cell_el_deg')] = TILT_SET[len(TILT_SET)//2]


    prediction_frame_template_gym = {}
    ue_track_generator_gym = None

    if USE_UE_TRACK_GENERATOR_GYM:
        logger.info("Using UETracksGenerator for dynamic UE locations in Gym.")
        try:
            min_lat_env = topology_df[c.CELL_LAT].min() - 0.02; max_lat_env = topology_df[c.CELL_LAT].max() + 0.02
            min_lon_env = topology_df[c.CELL_LON].min() - 0.02; max_lon_env = topology_df[c.CELL_LON].max() + 0.02
            ue_track_generator_gym = UETracksGenerator(
                n_ues=NUM_UES_FOR_GYM_STEP, min_lat=min_lat_env, max_lat=max_lat_env,
                min_lon=min_lon_env, max_lon=max_lon_env,
                x_dim=50, y_dim=50, min_wait_time=1, max_wait_time=3,
                min_speed=1, max_speed=3, seed=np.random.randint(0,10000) # Different seed per run
            )
        except Exception as e: logger.error(f"Failed to init UETracksGenerator: {e}. Fallback needed."); USE_UE_TRACK_GENERATOR_GYM = False

    if not USE_UE_TRACK_GENERATOR_GYM: # Fallback or initial choice
        logger.info("Using static prediction_frame_template for Gym (grid of points).")
        num_pts_side = int(math.sqrt(NUM_DUMMY_UE_POINTS_PER_CELL_FOR_GYM))
        lons = np.linspace(topology_df[c.CELL_LON].min(), topology_df[c.CELL_LON].max(), num_pts_side)
        lats = np.linspace(topology_df[c.CELL_LAT].min(), topology_df[c.CELL_LAT].max(), num_pts_side)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        base_ue_locs = pd.DataFrame({
            getattr(c, 'LOC_X', 'loc_x'): lon_grid.ravel(),
            getattr(c, 'LOC_Y', 'loc_y'): lat_grid.ravel()
        })
        # The Gym's __init__ will call BayesianDigitalTwin.create_prediction_frames
        # which adds cell-specific features. So, the template can be basic.
        # However, EnergySavingsGym expects a dict for prediction_frame_template
        for cell_id_val in site_config_df_gym[getattr(c,'CELL_ID','cell_id')].unique():
            prediction_frame_template_gym[cell_id_val] = base_ue_locs.copy()

    traffic_model_df_gym = None # Placeholder if you have a static traffic map

    # --- 5. Instantiate EnergySavingsGym ---
    logger.info("Instantiating EnergySavingsGym...")
    try:
        env = EnergySavingsGym(
            bayesian_digital_twins=gym_bdt_predictors, # Pass the dict of predictor wrappers
            site_config_df=site_config_df_gym,
            prediction_frame_template=prediction_frame_template_gym,
            tilt_set=TILT_SET,
            weak_coverage_threshold=WEAK_COVERAGE_THRESHOLD,
            over_coverage_threshold=OVER_COVERAGE_THRESHOLD,
            lambda_=LAMBDA_WEIGHT_GYM,
            traffic_model_df=traffic_model_df_gym,
            ue_track_generator=ue_track_generator_gym,
            horizon=GYM_HORIZON, debug=True
        )
        logger.info("EnergySavingsGym instantiated.")
    except Exception as e: logger.exception(f"Error instantiating EnergySavingsGym: {e}"); return

    # --- 6. Demo Interaction ---
    logger.info("--- Running Demo Interaction with Gym (Random Actions) ---")
    try:
        observation = env.reset()
        logger.info(f"Initial Gym Observation: {observation}")
        total_reward = 0
        for step_num in range(GYM_HORIZON): # Run for one episode
            action = env.action_space.sample() # Random action
            logger.info(f"\nGym Step {step_num + 1}/{GYM_HORIZON} - Action: {action}")
            observation, reward, done, info = env.step(action)
            logger.info(f"Gym Obs: {observation}, Reward: {reward:.3f}, Done: {done}")
            logger.info(f"Gym Info: {info}")
            total_reward += reward
            if done: logger.info(f"Episode finished at step {step_num + 1}."); break
        logger.info(f"Demo finished. Total reward over episode: {total_reward:.3f}")
    except Exception as e: logger.exception(f"Error during Gym interaction: {e}")
    finally: env.close()

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    # User needs to ensure TOPOLOGY_FILE_PATH, TRAINING_DATA_CSV_PATH, CONFIG_FILE_PATH exist.
    if not os.path.exists(TOPOLOGY_FILE_PATH):
        logger.error(f"Missing necessary input: {TOPOLOGY_FILE_PATH}. Please create it or use dummy generation from traffic_3.py."); sys.exit(1)
    if not os.path.exists(TRAINING_DATA_CSV_PATH):
        logger.error(f"Missing REALISTIC training data: {TRAINING_DATA_CSV_PATH}. Backend model needs this."); sys.exit(1)
    if not os.path.exists(CONFIG_FILE_PATH):
        logger.error(f"Missing initial config file: {CONFIG_FILE_PATH}. Please create it or use dummy generation."); sys.exit(1)

    main()