import os
import pandas as pd
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from apps.energy_saving_app.energy_saving_gym import TickAwareEnergyEnv

# Assuming radp is installed or in the python path
from radp.digital_twin.utils import constants as c


logger = logging.getLogger(__name__)

class RLManager:
    """
    Manages the training and inference of the reinforcement learning model
    for the energy saving application.
    """
    def __init__(self, bdt_manager, ue_data_dir, topology_path, config_path, rl_model_path="energy_saver_model.zip"):
        self.bdt_manager = bdt_manager
        self.ue_data_dir = ue_data_dir
        self.topology_path = topology_path
        self.config_path = config_path
        self.rl_model_path = rl_model_path
        self.model = None
        
        # Define constants for the Gym Environment
        self.TILT_SET = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        self.REWARD_WEIGHTS = {'coverage': 1.0, 'load_balance': 2.0, 'qos': 1.5, 'energy_saving': 3.0}
        self.WEAK_COVERAGE_THRESHOLD = -95.0
        self.OVER_COVERAGE_THRESHOLD = -65.0
        self.QOS_SINR_THRESHOLD = 0.0
        self.MAX_BAD_QOS_RATIO = 0.10
        self.GYM_HORIZON = 24

    def _load_data_for_gym(self):
        """Helper to load all necessary dataframes for the gym environment."""
        bdt_model_map = self.bdt_manager.get_model_map()
        
        topology_df = pd.read_csv(self.topology_path)
        config_df = pd.read_csv(self.config_path)

        # Merge topology and config to create the base site_config
        site_config_df = pd.merge(topology_df, config_df, on='cell_id', how='left')

        # --- FIX: Ensure all required columns for BDT prediction exist ---
        # This mirrors the logic from the original training script to prevent AttributeErrors
        # in the bayesian_engine.py library code.
        
        # Define defaults for potentially missing columns
        DEFAULT_HTX = 25.0
        DEFAULT_HRX = 1.5
        DEFAULT_AZIMUTH = 0.0
        DEFAULT_FREQUENCY = 2100.0

        # Dictionary of (constant_name, default_value)
        required_cols = {
            'HTX': DEFAULT_HTX,
            'HRX': DEFAULT_HRX,
            'CELL_AZ_DEG': DEFAULT_AZIMUTH,
            'CELL_CARRIER_FREQ_MHZ': DEFAULT_FREQUENCY,
        }

        for const_name, default_val in required_cols.items():
            col_name = getattr(c, const_name, const_name.lower())
            if col_name not in site_config_df.columns:
                logger.warning(f"Column '{col_name}' not found in site_config_df. Adding default value: {default_val}")
                site_config_df[col_name] = default_val
        
        # Also ensure tilt exists, filling NaNs from the merge if any
        col_tilt = getattr(c, 'CELL_EL_DEG', 'cell_el_deg')
        if col_tilt in site_config_df.columns:
            default_tilt = self.TILT_SET[len(self.TILT_SET) // 2]
            site_config_df[col_tilt].fillna(default_tilt, inplace=True)
        # --- End of column ensuring logic ---

        ue_data_per_tick = {}
        for tick in range(24):
            fpath = os.path.join(self.ue_data_dir, f"generated_ue_data_for_cco_{tick}.csv")
            if os.path.exists(fpath):
                ue_data_per_tick[tick] = pd.read_csv(fpath)
            else:
                 ue_data_per_tick[tick] = pd.DataFrame()
        
        if not any(not df.empty for df in ue_data_per_tick.values()):
            logger.error(f"No valid per-tick UE data found in '{self.ue_data_dir}'. Aborting RL training prep.")
            return None, None, None

        return bdt_model_map, site_config_df, ue_data_per_tick

    def train(self, total_timesteps=24000, log_dir="./rl_logs/"):
        """
        Trains the PPO reinforcement learning model.
        """
        print("Starting RL model training...")
        os.makedirs(log_dir, exist_ok=True)
        
        bdt_map, site_config, ue_data = self._load_data_for_gym()
        
        if bdt_map is None:
            logger.error("Could not train RL model due to missing data for Gym.")
            return

        env = TickAwareEnergyEnv(
            bayesian_digital_twins=bdt_map,
            site_config_df=site_config,
            ue_data_per_tick=ue_data,
            tilt_set=self.TILT_SET,
            reward_weights=self.REWARD_WEIGHTS,
            weak_coverage_threshold=self.WEAK_COVERAGE_THRESHOLD,
            over_coverage_threshold=self.OVER_COVERAGE_THRESHOLD,
            qos_sinr_threshold=self.QOS_SINR_THRESHOLD,
            max_bad_qos_ratio=self.MAX_BAD_QOS_RATIO,
            horizon=self.GYM_HORIZON
        )
        env = Monitor(env, log_dir)

        self.model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
        
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="rl_model")

        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        self.model.save(self.rl_model_path)
        
        print(f"RL model trained and saved to {self.rl_model_path}")
        env.close()

    def predict(self, target_tick):
        """
        Runs inference for a specific tick using the trained RL model.
        """
        if self.model is None:
            if not os.path.exists(self.rl_model_path):
                raise FileNotFoundError(f"RL model not found at {self.rl_model_path}. Please train first.")
            self.model = PPO.load(self.rl_model_path)

        print(f"Running inference for tick {target_tick}...")
        
        action_indices, _ = self.model.predict(target_tick, deterministic=True)
        
        topology_df = pd.read_csv(self.topology_path)
        cell_ids = topology_df['cell_id'].unique().tolist()

        config_list = []
        for i, cell_action_idx in enumerate(action_indices):
            cell_id = cell_ids[i]
            if cell_action_idx == len(self.TILT_SET): # OFF
                state, tilt = "OFF", "N/A"
            else:
                state, tilt = "ON", self.TILT_SET[cell_action_idx]
            config_list.append({"cell_id": cell_id, "state": state, "cell_el_deg": tilt})
        
        predicted_config_df = pd.DataFrame(config_list)
        print("\n--- Predicted Optimal Configuration ---")
        print(f"--- For Tick/Hour: {target_tick} ---")
        print(predicted_config_df.to_string(index=False))
        
        return predicted_config_df
