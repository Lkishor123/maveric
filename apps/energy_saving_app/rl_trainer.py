import os
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add necessary imports for unpickling the BDT model and running the simulation
import torch
import gpytorch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from radp.digital_twin.utils import constants as c
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin, ExactGPModel, NormMethod
from radp.digital_twin.utils.cell_selection import perform_attachment
from apps.coverage_capacity_optimization.cco_engine import CcoEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TickAwareEnergyEnv(gym.Env):
    """
    Custom Gym environment for energy saving, aware of the time of day (tick).
    This environment simulates taking actions (changing cell tilts or turning them off)
    at each tick and calculates a reward based on network KPIs.
    """
    metadata = {'render_modes': [], 'render_fps': 4}

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
        horizon: int = 24
    ):
        super().__init__()
        self.bayesian_digital_twins = bayesian_digital_twins
        self.site_config_df_initial = site_config_df.copy()
        
        self.COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')
        self.COL_CELL_EL_DEG = getattr(c, 'CELL_EL_DEG', 'cell_el_deg')
        self.COL_RXPOWER_DBM = getattr(c, 'RXPOWER_DBM', 'rxpower_dbm')
        self.COL_SINR_DB = getattr(c, 'SINR_DB', 'sinr_db')
        self.COL_LOC_X = getattr(c, 'LOC_X', 'loc_x')
        self.COL_LOC_Y = getattr(c, 'LOC_Y', 'loc_y')
        self.COL_RSRP_DBM = getattr(c, 'RSRP_DBM', 'rsrp_dbm')


        self.cell_ids = self.site_config_df_initial[self.COL_CELL_ID].unique().tolist()
        self.num_cells = len(self.cell_ids)
        self.ue_data_per_tick = ue_data_per_tick
        self.tilt_set = tilt_set
        self.num_tilt_options = len(self.tilt_set)
        self.reward_weights = reward_weights
        self.weak_coverage_threshold = weak_coverage_threshold
        self.over_coverage_threshold = over_coverage_threshold
        self.qos_sinr_threshold = qos_sinr_threshold
        self.horizon = horizon

        self.current_step_in_episode = 0
        self.current_tick_of_day = 0
        self.site_config_df_state = self.site_config_df_initial.copy()
        self.on_off_state = np.ones(self.num_cells, dtype=int)
        self.tilt_state = self.site_config_df_state[self.COL_CELL_EL_DEG].values.copy()

        self.action_space = spaces.MultiDiscrete([self.num_tilt_options + 1] * self.num_cells)
        self.observation_space = spaces.Discrete(24)
        logger.info(f"TickAwareEnergyEnv initialized for {self.num_cells} cells.")

    def _take_action(self, action: np.ndarray):
        new_on_off_state = np.ones(self.num_cells, dtype=int)
        new_tilt_state = np.zeros(self.num_cells, dtype=float)
        for i, action_for_cell in enumerate(action):
            if action_for_cell == self.num_tilt_options:
                new_on_off_state[i] = 0
                new_tilt_state[i] = self.tilt_state[i]
            else:
                new_on_off_state[i] = 1
                new_tilt_state[i] = self.tilt_set[action_for_cell]
        self.on_off_state = new_on_off_state
        self.tilt_state = new_tilt_state
        self.site_config_df_state[self.COL_CELL_EL_DEG] = self.tilt_state

    def _get_rf_predictions(self, current_ue_loc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if current_ue_loc_df is None or current_ue_loc_df.empty: return None
        all_preds_list = []
        for i, cell_id in enumerate(self.cell_ids):
            if self.on_off_state[i] == 1:
                bdt_predictor = self.bayesian_digital_twins.get(cell_id)
                if not bdt_predictor: continue
                cell_cfg_df = self.site_config_df_state[self.site_config_df_state[self.COL_CELL_ID] == cell_id]
                pred_frames = BayesianDigitalTwin.create_prediction_frames(
                    site_config_df=cell_cfg_df, prediction_frame_template=current_ue_loc_df
                )
                if cell_id in pred_frames and not pred_frames[cell_id].empty:
                    df_for_prediction = pred_frames[cell_id]
                    try:
                        bdt_predictor.predict_distributed_gpmodel(prediction_dfs=[df_for_prediction])
                        all_preds_list.append(df_for_prediction)
                    except Exception as e:
                        logger.error(f"BDT prediction error for cell {cell_id}: {e}")
        return pd.concat(all_preds_list, ignore_index=True) if all_preds_list else pd.DataFrame()

    def _calculate_reward(self, cell_selected_rf_df: Optional[pd.DataFrame]) -> Tuple[float, Dict]:
        info = {}
        num_active_cells = np.sum(self.on_off_state)
        if cell_selected_rf_df is None or cell_selected_rf_df.empty:
            info.update({"coverage_score": -100, "load_balance_score": -100, "qos_score": -100})
        else:
            # The 'perform_attachment' function returns a dataframe with 'rsrp_dbm'.
            # The CcoEngine expects this dataframe and infers the column names.
            # We must check for the existence of the RSRP column before calling.
            if self.COL_RSRP_DBM not in cell_selected_rf_df.columns:
                logger.warning(f"'{self.COL_RSRP_DBM}' not found in prediction results after attachment. Cannot calculate coverage.")
                info["coverage_score"] = -100.0
            else:
                # FIX: Remove the unexpected keyword arguments and call the function
                # as it is called in the original working script.
                cov_df = CcoEngine.rf_to_coverage_dataframe(
                    cell_selected_rf_df,
                    weak_coverage_threshold=self.weak_coverage_threshold,
                    over_coverage_threshold=self.over_coverage_threshold
                )
                info["coverage_score"] = (1.0 - cov_df['weakly_covered'].mean()) * 100
            
            active_topo = self.site_config_df_state[self.on_off_state == 1]
            ue_counts = cell_selected_rf_df[self.COL_CELL_ID].value_counts().reindex(active_topo[self.COL_CELL_ID], fill_value=0)
            info["load_balance_score"] = -ue_counts.std()
            
            info["qos_score"] = (cell_selected_rf_df[self.COL_SINR_DB] >= self.qos_sinr_threshold).mean() * 100 if self.COL_SINR_DB in cell_selected_rf_df.columns and not cell_selected_rf_df[self.COL_SINR_DB].isnull().all() else 0.0

        info["energy_saving_score"] = (1.0 - (num_active_cells / self.num_cells)) * 100 if self.num_cells > 0 else 100
        reward = sum(self.reward_weights.get(k, 1.0) * info.get(k, -100) for k in self.reward_weights)
        info["reward_total"] = reward
        return reward, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.current_step_in_episode = 0
        self.current_tick_of_day = self.observation_space.sample()
        self.site_config_df_state = self.site_config_df_initial.copy()
        self.on_off_state = np.ones(self.num_cells, dtype=int)
        self.tilt_state = self.site_config_df_state[self.COL_CELL_EL_DEG].values.copy()
        return self.current_tick_of_day, {}

    def step(self, action: np.ndarray):
        current_eval_tick = self.current_tick_of_day
        self._take_action(action)
        ue_df = self.ue_data_per_tick.get(current_eval_tick)
        all_preds_df = self._get_rf_predictions(ue_df)
        
        cell_selected_df = perform_attachment(all_preds_df, self.site_config_df_state[self.on_off_state == 1]) if all_preds_df is not None and not all_preds_df.empty else pd.DataFrame()
        
        reward, reward_info = self._calculate_reward(cell_selected_df)
        self.current_step_in_episode += 1
        self.current_tick_of_day = (self.current_tick_of_day + 1) % 24
        return self.current_tick_of_day, reward, self.current_step_in_episode >= self.horizon, False, reward_info

def run_rl_training(bdt_model_path, ue_data_dir, topology_path, config_path, rl_model_path, log_dir, total_timesteps=24000):
    """
    Main function to load data, initialize the environment, and train the RL agent.
    """
    logger.info("--- Starting RL Energy Saver Training Script ---")
    os.makedirs(log_dir, exist_ok=True)

    try:
        if not os.path.exists(bdt_model_path):
            raise FileNotFoundError(f"BDT Model file not found: {bdt_model_path}")
        
        bdt_model_map = BayesianDigitalTwin.load_model_map_from_pickle(bdt_model_path)
        logger.info(f"Loaded BDT map for {len(bdt_model_map)} cells.")

        topology_df = pd.read_csv(topology_path)
        config_df = pd.read_csv(config_path)
        site_config_df = pd.merge(topology_df, config_df, on='cell_id', how='left')

        TILT_SET = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        site_config_df[getattr(c, 'CELL_EL_DEG', 'cell_el_deg')].fillna(TILT_SET[len(TILT_SET)//2], inplace=True)
        required_cols = {'HTX': 25.0, 'HRX': 1.5, 'CELL_AZ_DEG': 0.0, 'CELL_CARRIER_FREQ_MHZ': 2100.0}
        for const, val in required_cols.items():
            col = getattr(c, const, const.lower())
            if col not in site_config_df.columns:
                site_config_df[col] = val

        ue_data_per_tick = {tick: pd.read_csv(os.path.join(ue_data_dir, f"generated_ue_data_for_cco_{tick}.csv")) for tick in range(24)}
        
        env = TickAwareEnergyEnv(
            bayesian_digital_twins=bdt_model_map, site_config_df=site_config_df, ue_data_per_tick=ue_data_per_tick,
            tilt_set=TILT_SET, reward_weights={'coverage': 1.0, 'load_balance': 2.0, 'qos': 1.5, 'energy_saving': 3.0},
            weak_coverage_threshold=-95.0, over_coverage_threshold=-65.0, qos_sinr_threshold=0.0, horizon=24
        )
        env = Monitor(env, log_dir)

        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="rl_model")
        logger.info(f"Starting RL training for {total_timesteps} timesteps...")
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        model.save(rl_model_path)
        logger.info(f"Training complete. Model saved to {rl_model_path}")
        env.close()

    except Exception as e:
        logger.exception(f"An error occurred during RL training setup or execution: {e}")
