# predict_cco_config.py
# Loads a trained RL agent and predicts CCO config for a given hour.

import gymnasium as gym
import numpy as np
import pandas as pd
import os
import sys
import argparse # For command-line arguments
import logging

# --- Assume RADP constants import (only need CELL_ID) ---
RADP_ROOT = os.getenv("MAVERIC_ROOT", "/path/to/your/maveric/project") # MODIFY IF NEEDED
if not os.path.isdir(RADP_ROOT):
     potential_path = os.path.join(os.path.dirname(__file__), "..", "..")
     if os.path.isdir(os.path.join(potential_path, "radp")): RADP_ROOT = os.path.abspath(potential_path); print(f"Warning: RADP_ROOT assumed: {RADP_ROOT}")
     else: raise FileNotFoundError(f"RADP_ROOT directory not found: {RADP_ROOT}.")
sys.path.insert(0, RADP_ROOT)

try:
    from radp.digital_twin.utils import constants as c
except ImportError:
    class c: CELL_ID = "cell_id"; CELL_EL_DEG = "cell_el_deg" # Fallback needed constants
    print("Warning: Using fallback constants.")

# --- RL Library Import ---
try:
    from stable_baselines3 import PPO # Use the same algorithm used for training
except ImportError:
    print("FATAL: stable-baselines3 not found. Please install it: pip install stable-baselines3")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Function (Similar to Env's mapping) ---
def map_action_to_config_df(action: np.ndarray, cell_ids: List[str], possible_tilts: List[float]) -> pd.DataFrame:
    """Converts action array to a config DataFrame."""
    config_data = []
    if len(action) != len(cell_ids):
        raise ValueError("Action length must match number of cell IDs.")
    num_tilt_options = len(possible_tilts)
    COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')
    COL_CELL_EL_DEG = getattr(c, 'CELL_EL_DEG', 'cell_el_deg')

    for i, cell_id in enumerate(cell_ids):
        tilt_index = np.clip(action[i], 0, num_tilt_options - 1) # Ensure valid
        tilt_value = possible_tilts[tilt_index]
        config_data.append({COL_CELL_ID: cell_id, COL_CELL_EL_DEG: tilt_value})
    return pd.DataFrame(config_data)

# --- Main Prediction Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict CCO configuration for a given hour using a trained RL model.")
    parser.add_argument("-m", "--model", type=str, default="./cco_rl_agent_ppo.zip", help="Path to the saved RL model zip file.")
    parser.add_argument("-t", "--topology", type=str, default="./data/topology.csv", help="Path to the topology CSV file (used for cell ID order).")
    parser.add_argument("--tick", type=int, required=True, help="The hour/tick (0-23) to predict the configuration for.")
    parser.add_argument("--tilts", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20", help="Comma-separated list of possible tilt values (must match training).")

    args = parser.parse_args()

    # Validate tick
    if not (0 <= args.tick <= 23):
        logger.error(f"Invalid tick: {args.tick}. Must be between 0 and 23.")
        sys.exit(1)

    # Load Topology to get cell order
    try:
        topology_df = pd.read_csv(args.topology)
        COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')
        if COL_CELL_ID not in topology_df.columns: raise ValueError("Topology missing cell_id column.")
        cell_ids_ordered = topology_df[COL_CELL_ID].unique().tolist()
        num_cells = len(cell_ids_ordered)
        logger.info(f"Loaded topology with {num_cells} unique cells from {args.topology}")
    except FileNotFoundError:
        logger.error(f"Topology file not found: {args.topology}"); sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading topology: {e}"); sys.exit(1)

    # Parse possible tilts
    try:
        possible_tilts = [float(t) for t in args.tilts.split(',')]
        logger.info(f"Using possible tilts: {possible_tilts}")
    except Exception as e:
        logger.error(f"Error parsing possible tilts string '{args.tilts}': {e}"); sys.exit(1)

    # Load Model
    try:
        if not os.path.exists(args.model): raise FileNotFoundError("Model file not found.")
        logger.info(f"Loading trained model from {args.model}...")
        # Ensure you load the same algorithm type (PPO)
        model = PPO.load(args.model)
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Saved model file not found: {args.model}"); sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading model: {e}"); sys.exit(1)

    # Prepare observation (the tick/hour)
    observation = args.tick

    # Predict Action
    logger.info(f"Predicting configuration for tick={observation}...")
    try:
        action, _states = model.predict(observation, deterministic=True)
        logger.info(f"Predicted action (tilt indices): {action}")
    except Exception as e:
        logger.error(f"Error during prediction: {e}"); sys.exit(1)

    # Map action to configuration DataFrame
    try:
        predicted_config_df = map_action_to_config_df(action, cell_ids_ordered, possible_tilts)
    except Exception as e:
        logger.error(f"Error mapping action to config DataFrame: {e}"); sys.exit(1)

    # Output results
    print("\n--- Predicted Optimal Configuration ---")
    print(f"--- For Tick/Hour: {args.tick} ---")
    # Print DataFrame in a readable format
    print(predicted_config_df.to_string(index=False))

    # Optionally save to CSV
    # output_config_csv = f"./predicted_config_tick_{args.tick}.csv"
    # try:
    #     predicted_config_df.to_csv(output_config_csv, index=False)
    #     logger.info(f"Saved predicted config to {output_config_csv}")
    # except Exception as e:
    #     logger.error(f"Failed to save predicted config CSV: {e}")