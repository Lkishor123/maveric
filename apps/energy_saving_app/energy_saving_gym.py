import os
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Assuming radp is installed or in the python path
from radp.digital_twin.utils import constants as c
# FIX: Add imports for all classes contained within the pickled BDT object
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
        max_bad_qos_ratio: float,
        horizon: int = 24
    ):
        super().__init__()
        self.bayesian_digital_twins = bayesian_digital_twins
        self.site_config_df_initial = site_config_df.copy()
        
        # Using getattr for safe access to constants
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
        self.max_bad_qos_ratio = max_bad_qos_ratio
        self.horizon = horizon

        self.current_step_in_episode = 0
        self.current_tick_of_day = 0

        self.site_config_df_state = self.site_config_df_initial.copy()
        self.on_off_state = np.ones(self.num_cells, dtype=int)
        self.tilt_state = self.site_config_df_state[self.COL_CELL_EL_DEG].values.copy()

        # Action: For each cell, choose a tilt index or "OFF"
        self.action_space = spaces.MultiDiscrete([self.num_tilt_options + 1] * self.num_cells)
        # Observation: The current hour/tick of the day
        self.observation_space = spaces.Discrete(24)
        logger.info(f"TickAwareEnergyEnv initialized for {self.num_cells} cells.")

    def _take_action(self, action: np.ndarray):
        new_on_off_state = np.ones(self.num_cells, dtype=int)
        new_tilt_state = np.zeros(self.num_cells, dtype=float)

        for i in range(self.num_cells):
            action_for_cell = action[i]
            if action_for_cell == self.num_tilt_options:  # "OFF" state
                new_on_off_state[i] = 0
                new_tilt_state[i] = self.tilt_state[i] # Keep tilt but cell is off
            else: # "ON" state with a specific tilt
                new_on_off_state[i] = 1
                new_tilt_state[i] = self.tilt_set[action_for_cell]

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
                    # REVERT to the correct, in-place modification method from the working script
                    bdt_predictor.predict_distributed_gpmodel(prediction_dfs=[df_for_prediction])
                    all_preds_list.append(df_for_prediction)
                except Exception as e: logger.error(f"BDT prediction error for {cell_id}: {e}")
        if not active_cell_count: return pd.DataFrame()
        if not all_preds_list: return None
        return pd.concat(all_preds_list, ignore_index=True)

    def _calculate_reward(self, cell_selected_rf_df: Optional[pd.DataFrame]) -> Tuple[float, Dict]:
        info = {}
        num_active_cells = np.sum(self.on_off_state)

        if cell_selected_rf_df is None or cell_selected_rf_df.empty:
            info["energy_saving_score"] = (1.0 - (num_active_cells / self.num_cells)) * 100 if self.num_cells > 0 else 100
            info["coverage_score"] = -100 # Penalize no coverage heavily
            info["load_balance_score"] = -100
            info["qos_score"] = -100
        else:
            if self.COL_RXPOWER_DBM not in cell_selected_rf_df.columns:
                 logger.warning(f"'{self.COL_RXPOWER_DBM}' not found in prediction results. Cannot calculate coverage.")
                 info["coverage_score"] = -100.0
            else:
                cov_df = CcoEngine.rf_to_coverage_dataframe(
                    rf_dataframe=cell_selected_rf_df,
                    loc_x_field=self.COL_LOC_X,
                    loc_y_field=self.COL_LOC_Y,
                    serving_cell_field=self.COL_CELL_ID,
                    rsrp_field=self.COL_RXPOWER_DBM,
                    weak_coverage_threshold=self.weak_coverage_threshold,
                    over_coverage_threshold=self.over_coverage_threshold,
                )
                info["coverage_score"] = (1.0 - cov_df['weakly_covered'].mean()) * 100
            
            active_topo = self.site_config_df_state[self.on_off_state == 1]
            ue_counts = cell_selected_rf_df[self.COL_CELL_ID].value_counts().reindex(active_topo[self.COL_CELL_ID], fill_value=0)
            info["load_balance_score"] = -ue_counts.std() # Higher is better (less negative)
            
            if self.COL_SINR_DB in cell_selected_rf_df.columns and not cell_selected_rf_df[self.COL_SINR_DB].isnull().all():
                good_qos_ratio = (cell_selected_rf_df[self.COL_SINR_DB] >= self.qos_sinr_threshold).mean()
                info["qos_score"] = good_qos_ratio * 100
            else:
                logger.warning(f"SINR column '{self.COL_SINR_DB}' not found or is all NaN. Setting QoS score to 0.")
                info["qos_score"] = 0.0
            
            active_ratio = num_active_cells / self.num_cells if self.num_cells > 0 else 0
            info["energy_saving_score"] = (1.0 - active_ratio) * 100

        reward = (self.reward_weights.get('coverage', 1.0) * info.get('coverage_score', -100) +
                  self.reward_weights.get('load_balance', 1.0) * info.get('load_balance_score', -100) +
                  self.reward_weights.get('qos', 1.0) * info.get('qos_score', -100) +
                  self.reward_weights.get('energy_saving', 1.0) * info.get('energy_saving_score', 0))
        
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
        
        cell_selected_df = None
        if all_preds_df is not None and not all_preds_df.empty:
            active_site_config = self.site_config_df_state[self.on_off_state == 1]
            if not active_site_config.empty:
                cell_selected_df = perform_attachment(all_preds_df, active_site_config)

        reward, reward_info = self._calculate_reward(cell_selected_df)
        
        self.current_step_in_episode += 1
        self.current_tick_of_day = (current_eval_tick + 1) % 24
        next_observation = self.current_tick_of_day
        done = self.current_step_in_episode >= self.horizon
        truncated = False
        
        return next_observation, reward, done, truncated, reward_info

    def render(self):
        pass
        
    def close(self):
        pass
