import os
import argparse
import logging
import pandas as pd
from stable_baselines3 import PPO

# It's good practice to have a central place for constants
TILT_SET = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_rl_prediction(model_load_path: str, topology_path: str, target_tick: int):
    """
    Loads a trained RL agent and predicts the optimal cell configuration for a given tick.
    """
    logger.info(f"--- Running RL Energy Saver Prediction for Tick {target_tick} ---")

    if not (0 <= target_tick <= 23):
        logger.error(f"Target tick {target_tick} out of range (0-23).")
        return

    try:
        topology_df = pd.read_csv(topology_path)
        cell_ids_ordered = topology_df['cell_id'].unique().tolist()
        
        model_file = model_load_path if model_load_path.endswith(".zip") else model_load_path + ".zip"
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"RL Model file not found: {model_file}")
            
        rl_model = PPO.load(model_file)
        logger.info(f"Loaded trained RL model from {model_file}")

        # The observation for the model is the tick itself
        action_indices, _ = rl_model.predict(target_tick, deterministic=True)
        
        config_list = []
        if len(action_indices) != len(cell_ids_ordered):
            logger.error(f"Action length {len(action_indices)} mismatch with num cells {len(cell_ids_ordered)}")
            return

        for i, cell_action_idx in enumerate(action_indices):
            cell_id = cell_ids_ordered[i]
            if cell_action_idx == len(TILT_SET):
                state, tilt = "OFF", "N/A"
            else:
                state, tilt = "ON", TILT_SET[cell_action_idx]
            config_list.append({"cell_id": cell_id, "state": state, "cell_el_deg": tilt})
        
        predicted_config_df = pd.DataFrame(config_list)
        print("\n--- Predicted Optimal Configuration ---")
        print(f"--- For Tick/Hour: {target_tick} ---")
        print(predicted_config_df.to_string(index=False))

    except Exception as e:
        logger.exception(f"An error occurred during prediction: {e}")

