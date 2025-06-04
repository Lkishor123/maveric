# energy_saving_demo_app.py

import os
import sys
import json
import logging
import math
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
# torch and gpytorch are not directly called here if BayesianDigitalTwin objects are fully pickled
# However, they are dependencies of BayesianDigitalTwin and EnergySavingsGym

# --- Path Setup ---
RADP_ROOT = os.getenv("MAVERIC_ROOT", "/home/lk/Projects/accelcq-repos/cloudly/github/maveric/") # MODIFY IF NEEDED
if not os.path.isdir(RADP_ROOT):
    potential_path = os.path.join(os.path.dirname(__file__), "..", "..") # Adjust if script is in a subdir
    if os.path.isdir(os.path.join(potential_path, "radp")):
        RADP_ROOT = os.path.abspath(potential_path)
        print(f"Warning: RADP_ROOT not explicitly set or found. Assuming relative path: {RADP_ROOT}")
    else:
        raise FileNotFoundError(f"RADP_ROOT directory not found: {RADP_ROOT}. Please set path.")
sys.path.insert(0, RADP_ROOT)
sys.path.insert(0, os.path.join(RADP_ROOT, "apps")) # To find energy_savings and CCO modules

# --- RADP and Custom Imports ---
try:
    from radp.client.client import RADPClient
    from radp.client.helper import RADPHelper, ModelStatus
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.utils.gis_tools import GISTools
    from radp.digital_twin.mobility.ue_tracks import UETracksGenerator
    from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin # For loading and type hints
    # from coverage_capacity_optimization.cco_engine import CcoEngine # Imported by EnergySavingsGym
    from energy_savings.energy_savings_gym import EnergySavingsGym # Assuming it's in apps/energy_savings/
except ImportError as e:
    print(f"FATAL: Error importing modules: {e}. Check paths and dependencies.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_DIR = "./" # Relative to where this script is run (e.g., maveric/apps/energy_saving_demo/)
TOPOLOGY_FILE_PATH = os.path.join(DATA_DIR, "topology.csv")
# For BDT training, ideally use realistic data. Using dummy data for pipeline testing.
TRAINING_DATA_CSV_PATH = os.path.join(DATA_DIR, "dummy_ue_training_data.csv")
CONFIG_FILE_PATH = os.path.join(DATA_DIR, "config.csv")     # For initial tilts in Gym

BDT_MODEL_ID = "bdt_for_es_demo_final" # Fresh ID for this training run
# Path where the backend training service *saves* the model (inside its Docker environment)
BACKEND_SAVES_MODEL_TO = f"/srv/radp/models/{BDT_MODEL_ID}/model.pickle"
# Path where this script will *look* for the model file after backend training.
# Ensure this path is accessible from where this script runs.
LOCAL_MODEL_ACCESS_PATH = os.path.join(DATA_DIR, BDT_MODEL_ID, "model.pickle") # Store in a subdir named after model_id

# Gym Parameters
TILT_SET = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
WEAK_COVERAGE_THRESHOLD_GYM = -95.0
OVER_COVERAGE_THRESHOLD_GYM = -65.0
LAMBDA_WEIGHT_GYM = 0.5 # Original lambda from EnergySavingsGym
NUM_DUMMY_UE_POINTS_PER_CELL_FOR_GYM = 50 # For static prediction_frame_template
USE_UE_TRACK_GENERATOR_GYM = False # Set to True to use UETracksGenerator
GYM_HORIZON = 24 # e.g., 24 hours
NUM_UES_FOR_GYM_STEP = 50 # If using UETracksGenerator

def main():
    logger.info("--- Energy Saving Demo Application ---")

    # --- 1. Initialize RADP Client ---
    try:
        radp_client = RADPClient()
        radp_helper = RADPHelper(radp_client)
        logger.info("RADP Client and Helper initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize RADP Client/Helper: {e}"); return

    # --- 2. Train BDT Model via Backend ---
    try:
        logger.info(f"Loading topology for BDT training from: {TOPOLOGY_FILE_PATH}")
        topology_df_for_training = pd.read_csv(TOPOLOGY_FILE_PATH)
        logger.info(f"Loading UE training data from: {TRAINING_DATA_CSV_PATH}")
        ue_training_df_for_bdt = pd.read_csv(TRAINING_DATA_CSV_PATH)

        # Basic validation of training data
        req_train_cols = ['cell_id', 'avg_rsrp', 'lon', 'lat', 'cell_el_deg']
        if not all(col in ue_training_df_for_bdt.columns for col in req_train_cols):
            logger.error(f"Training data {TRAINING_DATA_CSV_PATH} missing required columns: {req_train_cols}")
            return

        logger.info(f"Requesting training for BDT model ID: {BDT_MODEL_ID}")
        train_response = radp_client.train(
            model_id=BDT_MODEL_ID, params={}, model_update=False, # model_update=False for fresh train
            ue_training_data=ue_training_df_for_bdt, topology=topology_df_for_training
        )
        logger.info(f"Train request sent. Response: {train_response}")
        status = radp_helper.resolve_model_status(BDT_MODEL_ID, wait_interval=30, max_attempts=120, verbose=True) # Wait up to 1hr

        if not status.success:
            logger.error(f"BDT Model training failed for '{BDT_MODEL_ID}': {status.error_message}")
            logger.error("Ensure backend 'training' service is running and check its logs.")
            return
        logger.info(f"BDT Model '{BDT_MODEL_ID}' training reported as complete by backend.")
        logger.info(f"Model should be saved by backend at: {BACKEND_SAVES_MODEL_TO}")
        logger.info(f"This script will attempt to load it from: {LOCAL_MODEL_ACCESS_PATH}")
        logger.info("Ensure the model file is accessible at the local path (e.g., via shared volume or manual copy).")

    except FileNotFoundError as e: logger.error(f"Data file not found for training: {e}."); return
    except Exception as e: logger.exception(f"Error during BDT training phase: {e}"); return

    # --- Load topology again, to ensure we have a clean copy for Gym ---
    try:
        topology_df = pd.read_csv(TOPOLOGY_FILE_PATH)
    except FileNotFoundError: logger.error(f"Topology file not found: {TOPOLOGY_FILE_PATH}"); return
    except Exception as e: logger.error(f"Error loading topology: {e}"); return

    # --- 3. Load the Trained BDT Model (as pickled Dict[str, BayesianDigitalTwin]) ---
    logger.info(f"Attempting to load trained BDT model map from local path: {LOCAL_MODEL_ACCESS_PATH}")
    os.makedirs(os.path.dirname(LOCAL_MODEL_ACCESS_PATH), exist_ok=True) # Ensure target directory exists

    if not os.path.exists(LOCAL_MODEL_ACCESS_PATH):
        logger.error(f"Model file not found locally: {LOCAL_MODEL_ACCESS_PATH}.")
        logger.error("Please ensure the model trained by the backend (using save_model_map_to_pickle)")
        logger.error(f"is copied/mounted to this location. Backend default save path: {BACKEND_SAVES_MODEL_TO}")
        return
    try:
        # This uses the original pickling load function from bayesian_engine.py
        loaded_bdt_object_map = BayesianDigitalTwin.load_model_map_from_pickle(LOCAL_MODEL_ACCESS_PATH)

        if not loaded_bdt_object_map:
            logger.error("Failed to load BDT model map (returned empty or None)."); return
        logger.info(f"Successfully loaded BDT model map for {len(loaded_bdt_object_map)} cells.")

        # This map is directly passed to the Gym
        gym_bdt_predictors = loaded_bdt_object_map

    except Exception as e:
        logger.exception(f"Error loading BDT model map for Gym: {e}"); return

    # --- 4. Prepare Inputs for EnergySavingsGym ---
    logger.info("Preparing inputs for EnergySavingsGym...")
    try:
        site_config_df_gym = pd.read_csv(CONFIG_FILE_PATH)
        COL_CELL_ID = getattr(c,'CELL_ID','cell_id')
        COL_CELL_EL_DEG = getattr(c,'CELL_EL_DEG','cell_el_deg')

        if not all(col in site_config_df_gym.columns for col in [COL_CELL_ID, COL_CELL_EL_DEG]):
            raise ValueError(f"Config file {CONFIG_FILE_PATH} missing {COL_CELL_ID} or {COL_CELL_EL_DEG}")

        site_config_df_gym = pd.merge(
            topology_df.copy(),
            site_config_df_gym[[COL_CELL_ID, COL_CELL_EL_DEG]],
            on=COL_CELL_ID,
            how='left'
        )
        site_config_df_gym[COL_CELL_EL_DEG].fillna(TILT_SET[len(TILT_SET)//2], inplace=True)

        # Ensure necessary columns for BayesianDigitalTwin.create_prediction_frames exist
        DEFAULT_HTX = 25.0; DEFAULT_HRX = 1.5; DEFAULT_AZIMUTH = 0.0; DEFAULT_FREQUENCY = 2100.0
        for col_attr, default_val in [
            ('HTX', DEFAULT_HTX), ('HRX', DEFAULT_HRX),
            ('CELL_AZ_DEG', DEFAULT_AZIMUTH), ('CELL_CARRIER_FREQ_MHZ', DEFAULT_FREQUENCY)
        ]:
            col_name = getattr(c, col_attr, col_attr.lower()) # Get constant name or default to lowercase
            if col_name not in site_config_df_gym.columns:
                logger.warning(f"Column '{col_name}' (for {col_attr}) not in site_config_df_gym. Adding default: {default_val}")
                site_config_df_gym[col_name] = default_val

    except FileNotFoundError:
        logger.error(f"Config file not found: {CONFIG_FILE_PATH}. Using topology with default tilts/params.");
        site_config_df_gym = topology_df.copy()
        site_config_df_gym[getattr(c,'CELL_EL_DEG','cell_el_deg')] = TILT_SET[len(TILT_SET)//2]
        site_config_df_gym[getattr(c,'HTX','hTx')] = DEFAULT_HTX
        site_config_df_gym[getattr(c,'HRX','hRx')] = DEFAULT_HRX
        site_config_df_gym[getattr(c,'CELL_AZ_DEG','cell_az_deg')] = DEFAULT_AZIMUTH
        site_config_df_gym[getattr(c,'CELL_CARRIER_FREQ_MHZ','cell_carrier_freq_mhz')] = DEFAULT_FREQUENCY
    except Exception as e:
        logger.exception(f"Error preparing site_config_df_gym: {e}"); return

    prediction_frame_template_gym = {}
    ue_track_generator_gym = None
    
    # Local variable to manage generator usage for this run
    use_generator_this_run = USE_UE_TRACK_GENERATOR_GYM 

    if use_generator_this_run:
        logger.info("Using UETracksGenerator for dynamic UE locations in Gym.")
        try:
            min_lat_env = topology_df[c.CELL_LAT].min() - 0.02; max_lat_env = topology_df[c.CELL_LAT].max() + 0.02
            min_lon_env = topology_df[c.CELL_LON].min() - 0.02; max_lon_env = topology_df[c.CELL_LON].max() + 0.02
            ue_track_generator_gym = UETracksGenerator(
                n_ues=NUM_UES_FOR_GYM_STEP, min_lat=min_lat_env, max_lat=max_lat_env,
                min_lon=min_lon_env, max_lon=max_lon_env,
                x_dim=50, y_dim=50, min_wait_time=1, max_wait_time=3,
                min_speed=1, max_speed=3, seed=np.random.randint(0,10000)
            )
        except Exception as e:
            logger.error(f"Failed to init UETracksGenerator: {e}. Falling back to static template.")
            use_generator_this_run = False
    
    if not use_generator_this_run:
        logger.info("Using static prediction_frame_template for Gym (grid of points).")
        num_pts_side = int(math.sqrt(NUM_DUMMY_UE_POINTS_PER_CELL_FOR_GYM))
        if num_pts_side == 0: num_pts_side = 1
        # Use actual topology bounds for a more relevant grid
        topo_min_lon, topo_max_lon = topology_df[c.CELL_LON].min(), topology_df[c.CELL_LON].max()
        topo_min_lat, topo_max_lat = topology_df[c.CELL_LAT].min(), topology_df[c.CELL_LAT].max()
        lons = np.linspace(topo_min_lon, topo_max_lon, num_pts_side)
        lats = np.linspace(topo_min_lat, topo_max_lat, num_pts_side)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        base_ue_locs = pd.DataFrame({
            getattr(c, 'LOC_X', 'loc_x'): lon_grid.ravel(),
            getattr(c, 'LOC_Y', 'loc_y'): lat_grid.ravel()
        })
        for cell_id_val in site_config_df_gym[getattr(c,'CELL_ID','cell_id')].unique():
            prediction_frame_template_gym[cell_id_val] = base_ue_locs.copy()

    traffic_model_df_gym = None # Placeholder for static traffic model

    # --- 5. Instantiate EnergySavingsGym ---
    logger.info("Instantiating EnergySavingsGym...")
    try:
        env = EnergySavingsGym(
            bayesian_digital_twins=gym_bdt_predictors, # Pass the loaded map of BDT objects
            site_config_df=site_config_df_gym,
            prediction_frame_template=prediction_frame_template_gym, # Used if ue_track_generator is None
            tilt_set=TILT_SET,
            weak_coverage_threshold=WEAK_COVERAGE_THRESHOLD_GYM,
            over_coverage_threshold=OVER_COVERAGE_THRESHOLD_GYM,
            lambda_=LAMBDA_WEIGHT_GYM,
            traffic_model_df=traffic_model_df_gym,
            ue_track_generator=ue_track_generator_gym if use_generator_this_run else None,
            horizon=GYM_HORIZON, debug=True
        )
        logger.info("EnergySavingsGym instantiated successfully.")
    except Exception as e: logger.exception(f"Error instantiating EnergySavingsGym: {e}"); return

    # --- 6. Demo Interaction ---
    logger.info("--- Running Demo Interaction with Gym (Random Actions) ---")
    try:
        # It's good practice to seed the environment for reproducibility if needed
        # obs, info_reset = env.reset(seed=42) 
        obs = env.reset() # Original EnergySavingsGym reset doesn't take seed
        logger.info(f"Initial Gym Observation: {obs}")
        total_reward_accumulated = 0
        for step_num in range(GYM_HORIZON):
            action = env.action_space.sample()
            logger.info(f"\nGym Step {step_num + 1}/{GYM_HORIZON} - Action: {action}")
            # Original EnergySavingsGym step returns obs, reward, done, info
            observation, reward, done, info = env.step(action) 
            logger.info(f"Gym Obs: {observation}, Reward: {reward:.3f}, Done: {done}, Info: {info}")
            total_reward_accumulated += reward
            if done: logger.info(f"Episode finished at step {step_num + 1}."); break
        logger.info(f"Demo finished. Total accumulated reward: {total_reward_accumulated:.4f}")
    except Exception as e: logger.exception(f"Error during Gym interaction: {e}")
    finally: env.close()

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    # This script expects these files to exist. Generate them using traffic_3.py if needed.
    if not os.path.exists(TOPOLOGY_FILE_PATH): logger.error(f"Missing: {TOPOLOGY_FILE_PATH}"); sys.exit(1)
    if not os.path.exists(TRAINING_DATA_CSV_PATH): logger.error(f"Missing Training Data (for BDT): {TRAINING_DATA_CSV_PATH}"); sys.exit(1)
    if not os.path.exists(CONFIG_FILE_PATH): logger.error(f"Missing: {CONFIG_FILE_PATH}"); sys.exit(1)
    main()