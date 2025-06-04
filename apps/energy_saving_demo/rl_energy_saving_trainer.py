# rl_energy_saver_trainer.py

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
import gymnasium as gym # Use Gymnasium
from gymnasium import spaces # Use gymnasium.spaces

# --- Path Setup ---
RADP_ROOT = os.getenv("MAVERIC_ROOT", "/home/lk/Projects/accelcq-repos/cloudly/github/maveric/") # MODIFY
if not os.path.isdir(RADP_ROOT):
    potential_path = os.path.join(os.path.dirname(__file__), "..", "..", "..") # Adjust if script is deeper
    if os.path.isdir(os.path.join(potential_path, "radp")): RADP_ROOT = os.path.abspath(potential_path)
    else: raise FileNotFoundError(f"RADP_ROOT not found: {RADP_ROOT}")
sys.path.insert(0, RADP_ROOT)
sys.path.insert(0, os.path.join(RADP_ROOT, "apps")) # To find CCO Engine

# --- RADP and Custom Imports ---
try:
    from radp.client.client import RADPClient
    from radp.client.helper import RADPHelper, ModelStatus
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.utils.gis_tools import GISTools
    from radp.digital_twin.mobility.ue_tracks import UETracksGenerator
    from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin, ExactGPModel, NormMethod
    from coverage_capacity_optimization.cco_engine import CcoEngine
    from radp.digital_twin.utils.cell_selection import perform_attachment
except ImportError as e:
    print(f"FATAL: Error importing RADP modules: {e}. Check paths and dependencies."); sys.exit(1)

# --- RL Library Import ---
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback
except ImportError:
    print("FATAL: stable-baselines3 not found. pip install stable-baselines3[extra]"); sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Configuration for Training Script ---
DATA_DIR_TRAINER = "./"
TOPOLOGY_FILE_PATH_TRAINER = os.path.join(DATA_DIR_TRAINER, "topology.csv")
CONFIG_FILE_PATH_TRAINER = os.path.join(DATA_DIR_TRAINER, "config.csv")
UE_DATA_GYM_READY_DIR_TRAINER = "./ue_data_gym_ready"
BDT_MODEL_FILE_PATH_TRAINER = os.path.join(DATA_DIR_TRAINER, "model.pickle")

GYM_HORIZON_TRAINER = 24
TILT_SET_TRAINER = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
REWARD_WEIGHTS_TRAINER = {
    'coverage': 1.0, 'load_balance': 2.0, 'qos': 1.5, 'energy_saving': 3.0
}
WEAK_COVERAGE_THRESHOLD_RL = -95.0
OVER_COVERAGE_THRESHOLD_RL = -65.0
QOS_SINR_THRESHOLD_RL = 0.0
MAX_BAD_QOS_RATIO_RL = 0.10

TOTAL_TRAINING_TIMESTEPS_RL = 24 * 1000
RL_ALGORITHM = PPO
RL_POLICY = "MlpPolicy"
RL_LOG_DIR_TRAINER = "./rl_training_logs/"
RL_MODEL_SAVE_PATH_TRAINER = "./rl_energy_saver_agent_ppo"
CHECKPOINT_FREQ_RL_TRAINER = 24 * 50


# === Custom Gym Environment Definition (TickAwareEnergyEnv) ===
class TickAwareEnergyEnv(gym.Env):
    metadata = {'render_modes': [], 'render_fps': 4}
    ENERGY_MAX_PER_CELL = 40.0

    def __init__(
        self,
        bayesian_digital_twins: Dict[str, Any],
        site_config_df: pd.DataFrame,
        ue_data_per_tick: Dict[int, pd.DataFrame],
        tilt_set: List[float],
        reward_weights: Dict[str, float],
        weak_coverage_threshold: float,
        over_coverage_threshold: float,
        qos_sinr_threshold: float,
        max_bad_qos_ratio: float,
        horizon: int = 24,
        debug: bool = False
    ):
        super().__init__()
        self.bayesian_digital_twins = bayesian_digital_twins
        self.site_config_df_initial = site_config_df.copy()
        self.COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')
        self.COL_CELL_EL_DEG = getattr(c, 'CELL_EL_DEG', 'cell_el_deg')
        self.COL_RSRP_DBM = getattr(c, 'RSRP_DBM', 'rsrp_dbm')
        self.COL_SINR_DB = getattr(c, 'SINR_DB', 'sinr_db')
        self.COL_RXPOWER_DBM = getattr(c, 'RXPOWER_DBM', 'rxpower_dbm')

        self.cell_ids = self.site_config_df_initial[self.COL_CELL_ID].unique().tolist()
        self.num_cells = len(self.cell_ids)
        self.ue_data_per_tick = ue_data_per_tick
        self.tilt_set = tilt_set
        self.num_tilt_options = len(self.tilt_set)
        self.reward_weights = reward_weights
        self.weak_coverage_threshold = weak_coverage_threshold
        self.over_coverage_threshold = over_coverage_threshold
        self.qos_sinr_threshold = qos_sinr_threshold
        self.max_bad_qos_ratio = max_bad_qos_ratio
        self.horizon = horizon
        self.debug = debug

        self.current_step_in_episode = 0
        self.current_tick_of_day = 0

        self.site_config_df_state = self.site_config_df_initial.copy()
        self.on_off_state = np.ones(self.num_cells, dtype=int)
        self.tilt_state = self.site_config_df_state[self.COL_CELL_EL_DEG].values.copy()

        self.action_space = spaces.MultiDiscrete([self.num_tilt_options + 1] * self.num_cells)
        self.observation_space = spaces.Discrete(24)
        logger.info(f"TickAwareEnergyEnv initialized for {self.num_cells} cells.")

    def _take_action(self, action: np.ndarray):
        if len(action) != self.num_cells: raise ValueError("Action length mismatch")
        new_on_off_state = np.ones(self.num_cells, dtype=int)
        new_tilt_state = np.zeros(self.num_cells, dtype=float)
        for i in range(self.num_cells):
            action_for_cell = action[i]
            if action_for_cell == self.num_tilt_options:
                new_on_off_state[i] = 0
                new_tilt_state[i] = self.tilt_state[i]
            elif 0 <= action_for_cell < self.num_tilt_options:
                new_on_off_state[i] = 1
                new_tilt_state[i] = self.tilt_set[action_for_cell]
            else:
                logger.warning(f"Invalid action {action_for_cell} for cell {i}. Keeping previous.")
                new_on_off_state[i] = self.on_off_state[i]
                new_tilt_state[i] = self.tilt_state[i]
        self.on_off_state = new_on_off_state
        self.tilt_state = new_tilt_state
        self.site_config_df_state[self.COL_CELL_EL_DEG] = self.tilt_state

    def _get_rf_predictions(self, current_ue_loc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if current_ue_loc_df is None or current_ue_loc_df.empty: return None
        all_preds_list = []
        active_cell_count = 0
        for i, cell_id in enumerate(self.cell_ids):
            if self.on_off_state[i] == 1:
                active_cell_count += 1
                bdt_predictor = self.bayesian_digital_twins.get(cell_id)
                if not bdt_predictor: logger.error(f"BDT for {cell_id} not found!"); continue
                cell_cfg_df = self.site_config_df_state[self.site_config_df_state[self.COL_CELL_ID] == cell_id]
                if cell_cfg_df.empty: logger.error(f"Config for active cell {cell_id} not found."); continue
                
                loc_x_col = getattr(c, 'LOC_X', 'loc_x')
                loc_y_col = getattr(c, 'LOC_Y', 'loc_y')
                if not {loc_x_col, loc_y_col}.issubset(current_ue_loc_df.columns):
                    logger.error(f"UE data for tick {self.current_tick_of_day} missing {loc_x_col} or {loc_y_col}")
                    continue

                pred_frames_for_bdt = BayesianDigitalTwin.create_prediction_frames(
                    site_config_df=cell_cfg_df, prediction_frame_template=current_ue_loc_df
                )
                if cell_id not in pred_frames_for_bdt or pred_frames_for_bdt[cell_id].empty:
                    logger.warning(f"Failed to create prediction frame for cell {cell_id}"); continue
                df_for_prediction = pred_frames_for_bdt[cell_id]
                try:
                    bdt_predictor.predict_distributed_gpmodel(prediction_dfs=[df_for_prediction])
                    all_preds_list.append(df_for_prediction)
                except Exception as e: logger.error(f"BDT prediction error for {cell_id}: {e}")
        if not active_cell_count: return pd.DataFrame()
        if not all_preds_list: return None
        return pd.concat(all_preds_list, ignore_index=True)

    def _calculate_load_balancing_objective(self, rf_dataframe: pd.DataFrame, active_topology_df: pd.DataFrame) -> float:
        """
        Calculates a load balancing objective (-std dev of UEs per active cell).
        Higher is better.
        """
        if rf_dataframe is None or rf_dataframe.empty or active_topology_df.empty:
            return 0.0 # Or a penalty if no UEs/active cells implies bad state

        # Count UEs attached to each cell_id present in the RF results
        ue_counts_per_serving_cell = rf_dataframe.groupby(self.COL_CELL_ID).size()

        # Get all active cell IDs from the passed active_topology_df
        all_active_cell_ids = active_topology_df[self.COL_CELL_ID].unique()

        # Ensure all active cells are represented, filling non-serving (among active) with 0 UEs
        ue_counts_all_active_cells = ue_counts_per_serving_cell.reindex(all_active_cell_ids, fill_value=0)

        if len(ue_counts_all_active_cells) <= 1: # If 0 or 1 active cell, std dev is 0 or undefined (NaN)
            return 0.0 # Perfect balance

        std_dev = ue_counts_all_active_cells.std()
        return -std_dev # Negative standard deviation (higher is better)


    def _calculate_reward(self, cell_selected_rf_df: Optional[pd.DataFrame]) -> Tuple[float, Dict]:
        info = {"coverage_score": 0.0, "load_balance_score": -20.0, "qos_score": 0.0,
                "energy_saving_score": 0.0, "raw_neg_stdev": -20.0, "bad_qos_ratio": 1.0}
        
        num_active_cells = sum(self.on_off_state)

        if cell_selected_rf_df is None or cell_selected_rf_df.empty:
            if num_active_cells == 0: 
                info["energy_saving_score"] = 100.0
                reward = self.reward_weights.get('energy_saving', 1.0) * info["energy_saving_score"]
                info["coverage_score"] = 0.0 
                info["load_balance_score"] = 0.0 
                info["qos_score"] = 0.0 
                return reward, info
            else:
                logger.warning(f"Tick {self.current_tick_of_day}: No RF data for reward. High penalty.")
                return -200.0, info

        try:
            cov_df = CcoEngine.rf_to_coverage_dataframe(cell_selected_rf_df, weak_coverage_threshold=self.weak_coverage_threshold, over_coverage_threshold=self.over_coverage_threshold)
            info["coverage_score"] = (1.0 - cov_df['weakly_covered'].mean()) * 100.0
            
            active_cell_ids = self.site_config_df_state[self.on_off_state == 1][self.COL_CELL_ID].tolist()
            if active_cell_ids:
                active_rf = cell_selected_rf_df[cell_selected_rf_df[self.COL_CELL_ID].isin(active_cell_ids)]
                active_topo = self.site_config_df_state[self.site_config_df_state[self.COL_CELL_ID].isin(active_cell_ids)]
                if not active_rf.empty and not active_topo.empty:
                    # *** USE INTERNAL HELPER FOR LOAD BALANCING ***
                    raw_neg_stdev = self._calculate_load_balancing_objective(active_rf, active_topo)
                    info["raw_neg_stdev"] = raw_neg_stdev; info["load_balance_score"] = max(raw_neg_stdev, -50.0)
                else: info["load_balance_score"] = 0; info["raw_neg_stdev"] = 0
            else: 
                info["load_balance_score"] = 0; info["raw_neg_stdev"] = 0 
            
            if self.COL_SINR_DB in cell_selected_rf_df.columns and not cell_selected_rf_df[self.COL_SINR_DB].empty:
                good_qos_ratio = (cell_selected_rf_df[self.COL_SINR_DB] >= self.qos_sinr_threshold).mean()
                info["bad_qos_ratio"] = 1.0 - good_qos_ratio; info["qos_score"] = good_qos_ratio * 100.0
            else:
                logger.warning(f"SINR column '{self.COL_SINR_DB}' not found or empty. Setting QoS score to 0.")
                info["bad_qos_ratio"] = 1.0; info["qos_score"] = 0.0

            active_ratio = num_active_cells / self.num_cells if self.num_cells > 0 else 0
            info["energy_saving_score"] = (1.0 - active_ratio) * 100.0
            
            reward = (self.reward_weights.get('coverage', 0) * info["coverage_score"] +
                      self.reward_weights.get('load_balance', 0) * info["load_balance_score"] +
                      self.reward_weights.get('qos', 0) * info["qos_score"] +
                      self.reward_weights.get('energy_saving', 0) * info["energy_saving_score"])
            if not np.isfinite(reward): reward = -1000.0
        except Exception as e: logger.exception(f"Reward calc error tick {self.current_tick_of_day}: {e}"); reward = -500.0
        info["reward_total"] = reward
        # Corrected logging for info dictionary
        info_str = ", ".join([f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}" for key, value in info.items()])
        logger.debug(f"Tick {self.current_tick_of_day}: Reward={reward:.2f}, Info={{{info_str}}}")
        return reward, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.current_step_in_episode = 0
        self.current_tick_of_day = self.observation_space.sample()
        self.site_config_df_state = self.site_config_df_initial.copy()
        self.on_off_state = np.ones(self.num_cells, dtype=int)
        self.tilt_state = self.site_config_df_state[self.COL_CELL_EL_DEG].values.copy()
        logger.debug(f"Env Reset. Start Tick: {self.current_tick_of_day}")
        return self.current_tick_of_day, {}

    def step(self, action: np.ndarray):
        current_eval_tick = self.current_tick_of_day
        self._take_action(action)
        ue_df_for_this_tick = self.ue_data_per_tick.get(current_eval_tick)
        all_predictions_df = self._get_rf_predictions(ue_df_for_this_tick)
        cell_selected_rf_overall = None
        if all_predictions_df is not None and not all_predictions_df.empty:
            active_site_config = self.site_config_df_state[self.on_off_state == 1]
            if not active_site_config.empty:
                cell_selected_rf_overall = perform_attachment(all_predictions_df, active_site_config)
            else: cell_selected_rf_overall = pd.DataFrame()
        else: cell_selected_rf_overall = pd.DataFrame()
        
        reward, reward_info = self._calculate_reward(cell_selected_rf_overall)
        self.current_step_in_episode += 1
        self.current_tick_of_day = (current_eval_tick + 1) % 24
        next_observation = self.current_tick_of_day
        done = self.current_step_in_episode >= self.horizon
        truncated = False
        reward_info["tick"] = current_eval_tick; reward_info["action"] = action.tolist()
        return next_observation, reward, done, truncated, reward_info

    def render(self): pass
    def close(self): logger.info("Closing TickAwareEnergyEnv."); pass

# === Main Training Script Logic ===
def run_rl_training():
    logger.info("--- Starting RL Energy Saver Training Script ---")
    os.makedirs(RL_LOG_DIR_TRAINER, exist_ok=True)

    try:
        topology_df = pd.read_csv(TOPOLOGY_FILE_PATH_TRAINER)
        initial_config_df = pd.read_csv(CONFIG_FILE_PATH_TRAINER)
        COL_CELL_ID = getattr(c,'CELL_ID','cell_id'); COL_CELL_EL_DEG = getattr(c,'CELL_EL_DEG','cell_el_deg')
        site_config_df_for_gym = pd.merge(topology_df.copy(), initial_config_df[[COL_CELL_ID, COL_CELL_EL_DEG]], on=COL_CELL_ID, how='left')
        site_config_df_for_gym[COL_CELL_EL_DEG].fillna(TILT_SET_TRAINER[len(TILT_SET_TRAINER)//2], inplace=True)
        DEFAULT_HTX = 25.0; DEFAULT_HRX = 1.5; DEFAULT_AZIMUTH = 0.0; DEFAULT_FREQUENCY = 2100.0
        for col_attr, df_val in [('HTX', DEFAULT_HTX), ('HRX', DEFAULT_HRX), ('CELL_AZ_DEG', DEFAULT_AZIMUTH), ('CELL_CARRIER_FREQ_MHZ', DEFAULT_FREQUENCY)]:
            col_name = getattr(c, col_attr, col_attr.lower())
            if col_name not in site_config_df_for_gym.columns: site_config_df_for_gym[col_name] = df_val
    except Exception as e: logger.error(f"Error loading data for RL training: {e}"); return

    try:
        if not os.path.exists(BDT_MODEL_FILE_PATH_TRAINER):
            raise FileNotFoundError(f"BDT Model file not found: {BDT_MODEL_FILE_PATH_TRAINER}")
        loaded_bdt_map = BayesianDigitalTwin.load_model_map_from_pickle(BDT_MODEL_FILE_PATH_TRAINER)
        if not loaded_bdt_map: raise ValueError("Failed to load BDT map for RL Gym.")
        logger.info(f"Loaded BDT map for {len(loaded_bdt_map)} cells for RL Gym.")
    except Exception as e: logger.exception(f"Error preparing BDT map for RL Gym: {e}"); return

    ue_data_for_env = {}
    COL_LOC_X = getattr(c, 'LOC_X', 'loc_x'); COL_LOC_Y = getattr(c, 'LOC_Y', 'loc_y')
    for tick_idx in range(24):
        fpath = os.path.join(UE_DATA_GYM_READY_DIR_TRAINER, f"generated_ue_data_for_cco_{tick_idx}.csv")
        if os.path.exists(fpath):
            ue_df = pd.read_csv(fpath)
            if COL_LOC_X in ue_df.columns and COL_LOC_Y in ue_df.columns:
                ue_data_for_env[tick_idx] = ue_df[[COL_LOC_X, COL_LOC_Y]].copy()
            else: ue_data_for_env[tick_idx] = pd.DataFrame()
        else: ue_data_for_env[tick_idx] = pd.DataFrame()
    if not any(not df.empty for df in ue_data_for_env.values()):
        logger.error("No valid per-tick UE data loaded. Aborting."); return

    logger.info("Creating TickAwareEnergyEnv for RL training...")
    try:
        env = TickAwareEnergyEnv(
            bayesian_digital_twins=loaded_bdt_map,
            site_config_df=site_config_df_for_gym,
            ue_data_per_tick=ue_data_for_env,
            tilt_set=TILT_SET_TRAINER,
            reward_weights=REWARD_WEIGHTS_TRAINER,
            weak_coverage_threshold=WEAK_COVERAGE_THRESHOLD_RL,
            over_coverage_threshold=OVER_COVERAGE_THRESHOLD_RL,
            qos_sinr_threshold=QOS_SINR_THRESHOLD_RL,
            max_bad_qos_ratio=MAX_BAD_QOS_RATIO_RL,
            horizon=GYM_HORIZON_TRAINER, 
            debug=False
        )
        env = Monitor(env, RL_LOG_DIR_TRAINER)
        logger.info("TickAwareEnergyEnv for RL training created successfully.")
    except Exception as e: logger.exception(f"Failed to create RL environment: {e}"); return

    logger.info(f"Defining PPO agent...")
    model = RL_ALGORITHM(RL_POLICY, env, verbose=1, tensorboard_log=RL_LOG_DIR_TRAINER)
    checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ_RL_TRAINER, save_path=RL_LOG_DIR_TRAINER, name_prefix="rl_energy_agent")
    
    logger.info(f"Starting RL agent training for {TOTAL_TRAINING_TIMESTEPS_RL} timesteps...")
    try:
        model.learn(total_timesteps=TOTAL_TRAINING_TIMESTEPS_RL, callback=checkpoint_callback, tb_log_name="PPOTickAwareEnergyRun")
        logger.info("RL Training finished.")
        model.save(RL_MODEL_SAVE_PATH_TRAINER)
        logger.info(f"Trained RL model saved to {RL_MODEL_SAVE_PATH_TRAINER}.zip")
    except Exception as e: logger.exception(f"Error during RL training: {e}")
    finally: env.close()
    logger.info("--- RL Energy Saver Training Script Finished ---")

if __name__ == "__main__":
    # Ensure necessary directories and files for training
    if not os.path.exists(TOPOLOGY_FILE_PATH_TRAINER): logger.error(f"Missing: {TOPOLOGY_FILE_PATH_TRAINER}"); sys.exit(1)
    if not os.path.exists(CONFIG_FILE_PATH_TRAINER): logger.error(f"Missing: {CONFIG_FILE_PATH_TRAINER}"); sys.exit(1)
    if not os.path.exists(BDT_MODEL_FILE_PATH_TRAINER):
        logger.error(f"Missing BDT model: {BDT_MODEL_FILE_PATH_TRAINER}. Run BDT training first."); sys.exit(1)
    if not os.path.isdir(UE_DATA_GYM_READY_DIR_TRAINER) or not os.listdir(UE_DATA_GYM_READY_DIR_TRAINER):
        logger.error(f"Missing preprocessed UE data: {UE_DATA_GYM_READY_DIR_TRAINER}. Run preprocess_ue_for_gym.py first."); sys.exit(1)
    run_rl_training()

