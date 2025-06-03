# energy_saving_demo_app.py

import os
import sys
import json
import logging
import math
from typing import Dict, List, Any, Optional, Tuple # Ensure Any, Optional, Tuple are imported

import pandas as pd
import numpy as np
# import torch # Not directly needed here if BDT objects are fully pickled/unpickled
# import gpytorch # Not directly needed here if BDT objects are fully pickled/unpickled

# --- Path Setup ---
RADP_ROOT = os.getenv("MAVERIC_ROOT", "/home/lk/Projects/accelcq-repos/cloudly/github/maveric/") # MODIFY IF NEEDED
# ... (rest of path setup as before) ...
if not os.path.isdir(RADP_ROOT):
    potential_path = os.path.join(os.path.dirname(__file__), "..", "..")
    if os.path.isdir(os.path.join(potential_path, "radp")):
        RADP_ROOT = os.path.abspath(potential_path)
        print(f"Warning: RADP_ROOT not explicitly set or found. Assuming relative path: {RADP_ROOT}")
    else:
        raise FileNotFoundError(f"RADP_ROOT directory not found: {RADP_ROOT}. Please set path.")
sys.path.insert(0, RADP_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RADP_ROOT, "apps"))

# --- RADP and Custom Imports ---
try:
    from radp.client.client import RADPClient
    from radp.client.helper import RADPHelper, ModelStatus
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.utils.gis_tools import GISTools
    from radp.digital_twin.mobility.ue_tracks import UETracksGenerator
    # Import BayesianDigitalTwin directly
    from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin, ExactGPModel, NormMethod
    from coverage_capacity_optimization.cco_engine import CcoEngine
    from apps.energy_savings.energy_savings_gym import EnergySavingsGym
except ImportError as e:
    print(f"FATAL: Error importing modules: {e}. Check paths and dependencies.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (mostly as before) ---
DATA_DIR = "./"
TOPOLOGY_FILE_PATH = os.path.join(DATA_DIR, "topology.csv")
TRAINING_DATA_CSV_PATH = os.path.join(DATA_DIR, "dummy_ue_training_data.csv") # Or your realistic data
CONFIG_FILE_PATH = os.path.join(DATA_DIR, "config.csv")
BDT_MODEL_ID = "bdt_for_es_demo_v4" # Use a fresh ID for clarity
BACKEND_SAVES_MODEL_TO = f"/srv/radp/models/{BDT_MODEL_ID}/model.pickle" # Path inside Docker
LOCAL_MODEL_ACCESS_PATH = os.path.join(DATA_DIR, "model.pickle") # Where this script looks for the model

TILT_SET = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
WEAK_COVERAGE_THRESHOLD_GYM = -95.0
OVER_COVERAGE_THRESHOLD_GYM = -65.0
LAMBDA_WEIGHT_GYM = 0.5
NUM_DUMMY_UE_POINTS_PER_CELL_FOR_GYM = 100
USE_UE_TRACK_GENERATOR_GYM = False # Or False
GYM_HORIZON = 24
NUM_UES_FOR_GYM_STEP = 50

# --- REMOVE GymBDTCellPredictor class definition ---
# class GymBDTCellPredictor: ...

def main():
    logger.info("--- Energy Saving Demo Application ---")

    # --- 1. Initialize RADP Client ---
    try:
        radp_client = RADPClient()
        radp_helper = RADPHelper(radp_client)
        logger.info("RADP Client and Helper initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize RADP Client/Helper: {e}")
        return

    # --- 2. Train BDT Model via Backend (Currently Commented Out by User in provided script) ---
    # If you uncomment this, ensure topology_df is loaded here for use below.
    # For this fix, we assume topology_df will be loaded before Gym input preparation.
    # try:
    #     logger.info(f"Loading topology for BDT training from: {TOPOLOGY_FILE_PATH}")
    #     topology_df_for_training = pd.read_csv(TOPOLOGY_FILE_PATH)
    #     logger.info(f"Loading UE training data from: {TRAINING_DATA_CSV_PATH}")
    #     ue_training_df_for_bdt = pd.read_csv(TRAINING_DATA_CSV_PATH)

    #     logger.info(f"Requesting training for BDT model ID: {BDT_MODEL_ID}")
    #     # ... (rest of training call) ...
    #     if not status.success:
    #         logger.error(f"BDT Model training failed for '{BDT_MODEL_ID}': {status.error_message}"); return
    #     logger.info(f"BDT Model '{BDT_MODEL_ID}' training assumed complete by backend.")
    # except FileNotFoundError as e: logger.error(f"Data file not found for training: {e}."); return
    # except Exception as e: logger.exception(f"Error during BDT training phase: {e}"); return

    # --- Load topology (needed for Gym setup, even if BDT training is skipped) ---
    try:
        if 'topology_df' not in locals(): # Ensure topology_df is loaded if training was skipped
            topology_df = pd.read_csv(TOPOLOGY_FILE_PATH)
            logger.info(f"Loaded topology from {TOPOLOGY_FILE_PATH} for Gym setup.")
    except FileNotFoundError:
        logger.error(f"Topology file not found: {TOPOLOGY_FILE_PATH}. Cannot proceed."); return
    except Exception as e:
        logger.error(f"Error loading topology: {e}"); return


    # --- 3. Load the Trained BDT Model (States and Metadata) ---
    logger.info(f"Attempting to load trained BDT model map from local path: {LOCAL_MODEL_ACCESS_PATH}")
    # Ensure the directory for LOCAL_MODEL_ACCESS_PATH exists if it's nested
    model_dir = os.path.dirname(LOCAL_MODEL_ACCESS_PATH)
    if model_dir and not os.path.exists(model_dir): # Check if model_dir is not empty string
        os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(LOCAL_MODEL_ACCESS_PATH):
        logger.error(f"Model file not found locally: {LOCAL_MODEL_ACCESS_PATH}.")
        logger.error("Please ensure the model trained by the backend (and saved using the correct method)")
        logger.error(f"is copied/mounted to this location. Backend saves to: {BACKEND_SAVES_MODEL_TO}")
        return
    try:
        # Assuming bayesian_engine.py uses load_model_map_from_pickle for old style pickling
        # If it was changed to load_models_from_state (for torch.save files), use that.
        # Based on the error, it seems it loaded something, but the type was wrong for subscripting.
        # Let's assume the user wants to use the original pickling method from bayesian_engine.py for now.
        loaded_bdt_object_map = BayesianDigitalTwin.load_model_map_from_pickle(LOCAL_MODEL_ACCESS_PATH)

        if not loaded_bdt_object_map:
            logger.error("Failed to load BDT model map (returned empty or None)."); return
        logger.info(f"Successfully loaded BDT model map for {len(loaded_bdt_object_map)} cells.")

        # The gym_bdt_predictors will now directly be the loaded_bdt_object_map
        # as EnergySavingsGym expects a Dict[int, BayesianDigitalTwin] or similar interface
        gym_bdt_predictors = loaded_bdt_object_map

    except Exception as e:
        logger.exception(f"Error loading/preparing BDT model map for Gym: {e}"); return
    # --- 4. Prepare Inputs for EnergySavingsGym ---
    logger.info("Preparing inputs for EnergySavingsGym...")
    try:
        site_config_df_gym = pd.read_csv(CONFIG_FILE_PATH)
        COL_CELL_ID = getattr(c,'CELL_ID','cell_id')
        COL_CELL_EL_DEG = getattr(c,'CELL_EL_DEG','cell_el_deg')

        if not all(col in site_config_df_gym.columns for col in [COL_CELL_ID, COL_CELL_EL_DEG]):
            raise ValueError("Config.csv for Gym missing cell_id or cell_el_deg")
        
        # Merge with main topology to get all necessary columns for Gym's site_config_df
        # Ensure topology_df has all columns that create_prediction_frames might need from it.
        site_config_df_gym = pd.merge(
            topology_df.copy(), 
            site_config_df_gym[[COL_CELL_ID, COL_CELL_EL_DEG]],
            on=COL_CELL_ID,
            how='left'
        )
        site_config_df_gym[COL_CELL_EL_DEG].fillna(TILT_SET[len(TILT_SET)//2], inplace=True)

        # --- *** ADD THIS SECTION TO ENSURE REQUIRED COLUMNS EXIST *** ---
        COL_HTX = getattr(c, 'HTX', 'hTx')
        COL_HRX = getattr(c, 'HRX', 'hRx')
        COL_CELL_AZ_DEG = getattr(c, 'CELL_AZ_DEG', 'cell_az_deg')
        COL_CELL_CARRIER_FREQ_MHZ = getattr(c, 'CELL_CARRIER_FREQ_MHZ', 'cell_carrier_freq_mhz')
        
        DEFAULT_HTX = 25.0  # Example default transmitter height (meters)
        DEFAULT_HRX = 1.5   # Example default receiver height (meters)
        DEFAULT_AZIMUTH = 0.0
        DEFAULT_FREQUENCY = 2100

        if COL_HTX not in site_config_df_gym.columns:
            logger.warning(f"Column '{COL_HTX}' not found in site_config_df_gym. Adding default value: {DEFAULT_HTX}")
            site_config_df_gym[COL_HTX] = DEFAULT_HTX
        if COL_HRX not in site_config_df_gym.columns:
            logger.warning(f"Column '{COL_HRX}' not found in site_config_df_gym. Adding default value: {DEFAULT_HRX}")
            site_config_df_gym[COL_HRX] = DEFAULT_HRX
        
        # Add other columns expected by create_prediction_frames if they might be missing from your topology.csv
        if COL_CELL_AZ_DEG not in site_config_df_gym.columns:
            logger.warning(f"Column '{COL_CELL_AZ_DEG}' not found. Adding default: {DEFAULT_AZIMUTH}")
            site_config_df_gym[COL_CELL_AZ_DEG] = DEFAULT_AZIMUTH
        if COL_CELL_CARRIER_FREQ_MHZ not in site_config_df_gym.columns:
            logger.warning(f"Column '{COL_CELL_CARRIER_FREQ_MHZ}' not found. Adding default: {DEFAULT_FREQUENCY}")
            site_config_df_gym[COL_CELL_CARRIER_FREQ_MHZ] = DEFAULT_FREQUENCY
        # --- *** END SECTION TO ADD *** ---

    except FileNotFoundError:
        logger.error(f"Config file not found: {CONFIG_FILE_PATH}. Using topology and default tilts/heights for Gym.")
        site_config_df_gym = topology_df.copy()
        site_config_df_gym[getattr(c,'CELL_EL_DEG','cell_el_deg')] = TILT_SET[len(TILT_SET)//2]
        site_config_df_gym[getattr(c, 'HTX', 'hTx')] = DEFAULT_HTX # Add defaults here too
        site_config_df_gym[getattr(c, 'HRX', 'hRx')] = DEFAULT_HRX
        site_config_df_gym[getattr(c, 'CELL_AZ_DEG', 'cell_az_deg')] = DEFAULT_AZIMUTH
        site_config_df_gym[getattr(c, 'CELL_CARRIER_FREQ_MHZ', 'cell_carrier_freq_mhz')] = DEFAULT_FREQUENCY
    except Exception as e:
        logger.exception(f"Error preparing site_config_df_gym for Gym: {e}. Using basic topology with defaults.");
        site_config_df_gym = topology_df.copy()
        site_config_df_gym[getattr(c,'CELL_EL_DEG','cell_el_deg')] = TILT_SET[len(TILT_SET)//2]
        site_config_df_gym[getattr(c, 'HTX', 'hTx')] = DEFAULT_HTX
        site_config_df_gym[getattr(c, 'HRX', 'hRx')] = DEFAULT_HRX
        site_config_df_gym[getattr(c, 'CELL_AZ_DEG', 'cell_az_deg')] = DEFAULT_AZIMUTH
        site_config_df_gym[getattr(c, 'CELL_CARRIER_FREQ_MHZ', 'cell_carrier_freq_mhz')] = DEFAULT_FREQUENCY

    prediction_frame_template_gym = {}
    ue_track_generator_gym = None
    
    # *** FIX FOR UnboundLocalError STARTS HERE ***
    # Read global config into a local variable for this function's scope
    use_generator_for_this_run = USE_UE_TRACK_GENERATOR_GYM

    if use_generator_for_this_run: # Check the local variable
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
            use_generator_for_this_run = False # Modify the local variable if generator fails
    
    if not use_generator_for_this_run: # Check the (potentially modified) local variable
        logger.info("Using static prediction_frame_template for Gym (grid of points).")
        num_pts_side = int(math.sqrt(NUM_DUMMY_UE_POINTS_PER_CELL_FOR_GYM)) # Corrected variable name
        if num_pts_side == 0 : num_pts_side = 1
        lons = np.linspace(topology_df[c.CELL_LON].min(), topology_df[c.CELL_LON].max(), num_pts_side)
        lats = np.linspace(topology_df[c.CELL_LAT].min(), topology_df[c.CELL_LAT].max(), num_pts_side)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        base_ue_locs = pd.DataFrame({
            getattr(c, 'LOC_X', 'loc_x'): lon_grid.ravel(),
            getattr(c, 'LOC_Y', 'loc_y'): lat_grid.ravel()
        })
        for cell_id_val in site_config_df_gym[getattr(c,'CELL_ID','cell_id')].unique():
            prediction_frame_template_gym[cell_id_val] = base_ue_locs.copy()
    # *** FIX FOR UnboundLocalError ENDS HERE ***

    traffic_model_df_gym = None

    # --- 5. Instantiate EnergySavingsGym ---
    logger.info("Instantiating EnergySavingsGym...")
    try:
        env = EnergySavingsGym(
            bayesian_digital_twins=gym_bdt_predictors, # Pass the loaded map of BDT objects
            site_config_df=site_config_df_gym,
            prediction_frame_template=prediction_frame_template_gym,
            tilt_set=TILT_SET,
            weak_coverage_threshold=WEAK_COVERAGE_THRESHOLD_GYM,
            over_coverage_threshold=OVER_COVERAGE_THRESHOLD_GYM,
            lambda_=LAMBDA_WEIGHT_GYM,
            traffic_model_df=traffic_model_df_gym,
            ue_track_generator=ue_track_generator_gym if use_generator_for_this_run else None, # Use local var
            horizon=GYM_HORIZON, debug=True
        )
        logger.info("EnergySavingsGym instantiated successfully.")
    except Exception as e: logger.exception(f"Error instantiating EnergySavingsGym: {e}"); return

    # --- 6. Demo Interaction ---
    # ... (Demo loop as before) ...
    logger.info("--- Running Demo Interaction with Gym (Random Actions) ---")
    try:
        observation = env.reset()
        logger.info(f"Initial Gym Observation: {observation}")
        total_reward_accumulated = 0
        for step_num in range(GYM_HORIZON): # Run for one episode
            action = env.action_space.sample()
            logger.info(f"\nGym Step {step_num + 1}/{GYM_HORIZON} - Action: {action}")
            observation, reward, done, info = env.step(action)
            logger.info(f"Gym Obs: {observation}, Reward: {reward:.3f}, Done: {done}, Info: {info}")
            total_reward_accumulated += reward
            if done: logger.info(f"Episode finished at step {step_num + 1}."); break
        logger.info(f"Demo finished. Total accumulated reward: {total_reward_accumulated:.4f}")
    except Exception as e: logger.exception(f"Error during Gym interaction: {e}")
    finally: env.close()

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(TOPOLOGY_FILE_PATH): logger.error(f"Missing: {TOPOLOGY_FILE_PATH}"); sys.exit(1)
    # TRAINING_DATA_CSV_PATH only needed if BDT training is active in main()
    # if not os.path.exists(TRAINING_DATA_CSV_PATH): logger.error(f"Missing: {TRAINING_DATA_CSV_PATH}"); sys.exit(1)
    if not os.path.exists(CONFIG_FILE_PATH): logger.error(f"Missing: {CONFIG_FILE_PATH}"); sys.exit(1)
    main()