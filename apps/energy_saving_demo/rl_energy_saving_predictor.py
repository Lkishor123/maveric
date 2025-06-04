# rl_energy_saver_predictor.py

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np

# --- Path Setup ---
RADP_ROOT = os.getenv("MAVERIC_ROOT", "/home/lk/Projects/accelcq-repos/cloudly/github/maveric/") # MODIFY
if not os.path.isdir(RADP_ROOT):
    potential_path = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    if os.path.isdir(os.path.join(potential_path, "radp")): RADP_ROOT = os.path.abspath(potential_path)
    else: raise FileNotFoundError(f"RADP_ROOT not found: {RADP_ROOT}")
sys.path.insert(0, RADP_ROOT)

try:
    from radp.digital_twin.utils import constants as c
except ImportError:
    class c: CELL_ID = "cell_id"; CELL_EL_DEG = "cell_el_deg"
    print("Warning: Using fallback constants for predictor.")

# --- RL Library Import ---
try:
    from stable_baselines3 import PPO
except ImportError:
    print("FATAL: stable-baselines3 not found. pip install stable-baselines3"); sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (Should match trainer's tilt set) ---
TILT_SET_PREDICTOR = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

def run_rl_prediction(model_load_path: str, topology_path: str, target_tick: int):
    logger.info(f"--- Running RL Energy Saver Prediction for Tick {target_tick} ---")
    COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')
    COL_CELL_EL_DEG = getattr(c, 'CELL_EL_DEG', 'cell_el_deg')


    if not (0 <= target_tick <= 23):
        logger.error(f"Target tick {target_tick} out of range (0-23)."); return

    try:
        topology_df = pd.read_csv(topology_path)
        if COL_CELL_ID not in topology_df.columns: raise ValueError("Topology missing cell_id")
        cell_ids_ordered = topology_df[COL_CELL_ID].unique().tolist()
    except Exception as e: logger.error(f"Error loading topology {topology_path}: {e}"); return

    try:
        model_file_to_load = model_load_path if model_load_path.endswith(".zip") else model_load_path + ".zip"
        if not os.path.exists(model_file_to_load):
            raise FileNotFoundError(f"RL Model file not found: {model_file_to_load}")
        rl_model = PPO.load(model_file_to_load)
        logger.info(f"Loaded trained RL model from {model_file_to_load}")
    except Exception as e: logger.error(f"Error loading RL model: {e}"); return

    # Observation for the model is the tick itself
    observation = target_tick
    action_indices, _ = rl_model.predict(observation, deterministic=True)
    logger.info(f"Predicted raw action (indices) for tick {target_tick}: {action_indices}")

    config_list = []
    if len(action_indices) != len(cell_ids_ordered):
        logger.error(f"Action length {len(action_indices)} mismatch with num cells {len(cell_ids_ordered)}")
        return

    for i, cell_action_idx in enumerate(action_indices):
        cell_id = cell_ids_ordered[i]
        if cell_action_idx == len(TILT_SET_PREDICTOR): # Index for "OFF"
            state = "OFF"
            tilt = "N/A" # Or could be the last known ON tilt
        elif 0 <= cell_action_idx < len(TILT_SET_PREDICTOR):
            state = "ON"
            tilt = TILT_SET_PREDICTOR[cell_action_idx]
        else:
            state = "INVALID_ACTION"
            tilt = f"RawIdx:{cell_action_idx}"
            logger.warning(f"Invalid action index {cell_action_idx} for cell {cell_id}")
        config_list.append({COL_CELL_ID: cell_id, "state": state, COL_CELL_EL_DEG: tilt})
    
    predicted_config_df = pd.DataFrame(config_list)
    print("\n--- Predicted Optimal Configuration ---")
    print(f"--- For Tick/Hour: {target_tick} ---")
    print(predicted_config_df.to_string(index=False))

    # To save:
    # output_filename = f"./predicted_config_tick_{target_tick}.csv"
    # predicted_config_df.to_csv(output_filename, index=False)
    # logger.info(f"Saved predicted configuration to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict CCO config using trained RL model.")
    parser.add_argument("-m", "--model", type=str, default="./rl_energy_saver_agent_ppo", help="Path to the saved RL agent model (without .zip).")
    parser.add_argument("-t", "--topology", type=str, default="./topology.csv", help="Path to the topology CSV file.")
    parser.add_argument("--tick", type=int, required=True, help="Target tick/hour (0-23) for prediction.")
    
    args = parser.parse_args()
    run_rl_prediction(args.model, args.topology, args.tick)

