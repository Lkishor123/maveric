# energy_savings_gym_adapted.py (Save this as energy_savings_gym.py or integrate)

import os
import sys
import logging
import math
from typing import Dict, List, Any, Optional, Tuple

import gymnasium as gym # Use gymnasium
from gymnasium import spaces # Use gymnasium.spaces
import numpy as np
import pandas as pd

# --- Assume RADP imports are set up correctly via sys.path / RADP_ROOT ---
try:
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.utils.gis_tools import GISTools
    from radp.digital_twin.mobility.ue_tracks import UETracksGenerator
    from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin # For type hinting and static methods
    from radp.digital_twin.utils.cell_selection import perform_attachment
    # Assuming CcoEngine is accessible, e.g. from apps.coverage_capacity_optimization
    from coverage_capacity_optimization.cco_engine import CcoEngine
except ImportError as e:
    print(f"CRITICAL ERROR importing RADP modules for EnergySavingsGym: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

class EnergySavingsGym(gym.Env):
    """
    Adapted Energy Savings Gym Environment for RL.
    Observation: Current hour of the day (0-23).
    Action: Tilt and On/Off state for each cell.
    Reward: Balances energy savings, coverage, QoS, and load balance.
    """
    metadata = {'render_modes': [], 'render_fps': 4}

    # Constants for observation space normalization (can be tuned)
    ENERGY_MAX_PER_CELL = 40.0 # Example, arbitrary units
    # MAX_SUM_PEAK_RATE_CLUSTER = 10000.0 # Original, might not be used if obs changes
    # MAX_CLUSTER_CCO = 0.0 # Original, might not be used if obs changes
    # For a simpler observation space (just the tick), these might not be needed.

    def __init__(
        self,
        bayesian_digital_twins: Dict[str, Any], # Dict of BDT predictor objects (e.g., GymBDTCellPredictor)
        site_config_df: pd.DataFrame,
        ue_data_dir: str, # Directory with per-tick UE data (loc_x, loc_y)
        tilt_set: List[float], # Changed to float for actual tilt values
        reward_weights: Dict[str, float],
        weak_coverage_threshold: float = -95.0,
        over_coverage_threshold: float = -65.0, # For CCO metric part of reward
        qos_sinr_threshold: float = 0.0, # dB, for QoS penalty
        max_bad_qos_ratio: float = 0.05, # Penalize if more than 5% UEs have bad QoS
        lambda_reward: float = 0.5, # Original lambda for energy vs CCO
        horizon: int = 24, # Typically one day cycle
        seed: int = 0,
        debug: bool = False,
        # prediction_frame_template: Dict[str, pd.DataFrame] = None, # Replaced by ue_data_dir
        # traffic_model_df: pd.DataFrame = None, # Can be added if needed for reward
        # ue_track_generator: UETracksGenerator = None # Replaced by ue_data_dir
    ):
        super().__init__()
        np.random.seed(seed) # Seed numpy for reproducibility in UE selection from areas

        self.bayesian_digital_twins = bayesian_digital_twins # Dict: {cell_id: BDT_predictor_object}
        self.site_config_df_initial = site_config_df.copy() # Keep original
        self.cell_ids = self.site_config_df_initial[getattr(c, 'CELL_ID', 'cell_id')].unique().tolist()
        self.num_cells = len(self.cell_ids)

        self.ue_data_dir = ue_data_dir
        self.tilt_set = tilt_set
        self.num_tilt_options = len(self.tilt_set)
        self.reward_weights = reward_weights
        self.weak_coverage_threshold = weak_coverage_threshold
        self.over_coverage_threshold = over_coverage_threshold
        self.qos_sinr_threshold = qos_sinr_threshold
        self.max_bad_qos_ratio = max_bad_qos_ratio
        self.lambda_reward = lambda_reward # For original reward component if kept
        self.horizon = horizon
        self.debug = debug

        # --- Load per-tick UE data ---
        self.ue_data_per_tick: Dict[int, Optional[pd.DataFrame]] = self._load_all_ue_data()
        if not any(self.ue_data_per_tick.values()):
            raise FileNotFoundError(f"No valid UE data files loaded from {ue_data_dir} for Gym.")

        # --- State Variables ---
        self.current_step_in_episode = 0 # For tracking episode length vs horizon
        self.current_tick_of_day = 0     # The actual observation (0-23)
        
        # site_config_df_state holds the current configuration being evaluated
        self.site_config_df_state = self.site_config_df_initial.copy()
        
        # on_off_state: 1 for ON, 0 for OFF. Action index len(tilt_set) means OFF.
        self.on_off_state = np.ones(self.num_cells, dtype=int)
        # tilt_state: stores the actual tilt value for each cell
        self.tilt_state = self.site_config_df_state[getattr(c, 'CELL_EL_DEG', 'cell_el_deg')].values.copy()


        # --- Action Space ---
        # For each cell: choose a tilt from tilt_set OR turn it off.
        # Action `len(self.tilt_set)` for a cell means that cell is OFF.
        # Action `0` to `len(self.tilt_set)-1` corresponds to an index in `self.tilt_set`.
        self.action_space = spaces.MultiDiscrete([self.num_tilt_options + 1] * self.num_cells)
        self.action_space.seed(seed) # Seed the action space for reproducibility

        # --- Observation Space ---
        # The observation is the current hour of the day (tick).
        self.observation_space = spaces.Discrete(24) # Hours 0-23

        # For reward normalization (from original EnergySavingsGym, can be adjusted)
        self.r_norm = (1 - self.lambda_reward) * (
            -10 * np.log10(self.num_cells if self.num_cells > 0 else 1) # Avoid log(0)
            - self.over_coverage_threshold # This seems like it should be a positive value to subtract
            # + self.min_rsrp # min_rsrp was in original, but not well-defined here
            - self.weak_coverage_threshold
        )
        if self.num_cells == 0: self.r_norm = 0 # Avoid issues if no cells

        logger.info(f"EnergySavingsGym initialized for {self.num_cells} cells. Action space: {self.action_space}. Obs space: {self.observation_space}.")

    def _load_all_ue_data(self) -> Dict[int, Optional[pd.DataFrame]]:
        """Loads per-tick UE data (loc_x, loc_y) into memory."""
        data = {}
        logger.info(f"Gym: Loading per-tick UE data from {self.ue_data_dir}...")
        COL_LOC_X = getattr(c, 'LOC_X', 'loc_x'); COL_LOC_Y = getattr(c, 'LOC_Y', 'loc_y')
        found_count = 0
        for tick in range(24): # Expecting files for ticks 0-23
            # Filename should match output of preprocess_ue_for_gym.py
            filename = f"generated_ue_data_for_cco_{tick}.csv"
            filepath = os.path.join(self.ue_data_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    if COL_LOC_X in df.columns and COL_LOC_Y in df.columns:
                         data[tick] = df[[COL_LOC_X, COL_LOC_Y]].copy() # Only need these for prediction template
                         found_count +=1
                    else: logger.warning(f"Gym: Skipping {filename}: Missing '{COL_LOC_X}' or '{COL_LOC_Y}'."); data[tick]=None
                except Exception as e: logger.error(f"Gym: Error loading {filepath}: {e}"); data[tick]=None
            else: logger.warning(f"Gym: UE data file not found for tick {tick}: {filepath}"); data[tick]=None
        logger.info(f"Gym: Loaded UE data for {found_count}/24 ticks.")
        return data

    def _take_action(self, action: np.ndarray):
        """Applies the agent's action to update cell on/off states and tilts."""
        if len(action) != self.num_cells:
            raise ValueError(f"Action length ({len(action)}) != num_cells ({self.num_cells})")

        new_on_off_state = np.ones(self.num_cells, dtype=int)
        new_tilt_state = np.zeros(self.num_cells, dtype=float)
        COL_CELL_EL_DEG = getattr(c,'CELL_EL_DEG','cell_el_deg')

        for i in range(self.num_cells):
            action_for_cell = action[i]
            if action_for_cell == self.num_tilt_options: # Special action index for "OFF"
                new_on_off_state[i] = 0
                # Keep current tilt or set to a default when off? Let's keep current for now.
                new_tilt_state[i] = self.site_config_df_state.iloc[i][COL_CELL_EL_DEG]
            elif 0 <= action_for_cell < self.num_tilt_options:
                new_on_off_state[i] = 1 # Cell is ON
                new_tilt_state[i] = self.tilt_set[action_for_cell]
            else:
                logger.warning(f"Invalid action {action_for_cell} for cell {i}. Keeping previous state.")
                new_on_off_state[i] = self.on_off_state[i]
                new_tilt_state[i] = self.tilt_state[i]
        
        self.on_off_state = new_on_off_state
        self.tilt_state = new_tilt_state
        self.site_config_df_state[COL_CELL_EL_DEG] = self.tilt_state
        # Note: site_config_df_state is now updated with new tilts.
        # The on_off_state will be used to filter which cells participate in RF prediction.

    def _get_rf_predictions_for_active_cells(self, current_ue_df_template: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Gets RF predictions from BDTs for all UEs against all ACTIVE cells.
        Returns a single DataFrame with all predictions, or None on failure.
        """
        if current_ue_df_template is None or current_ue_df_template.empty:
            logger.warning("Gym: No UE data for current tick to run predictions.")
            return None

        all_predictions_list = []
        active_cell_count = 0

        for i, cell_id in enumerate(self.cell_ids):
            if self.on_off_state[i] == 1: # If cell is ON
                active_cell_count += 1
                bdt_predictor = self.bayesian_digital_twins.get(cell_id)
                if not bdt_predictor:
                    logger.error(f"Gym: BDT predictor not found for active cell {cell_id}. Skipping.")
                    continue

                # Create prediction frame for this specific cell using its current config
                # BayesianDigitalTwin.create_prediction_frames expects a list of site_config_df rows
                # Here, we pass the config for the single cell being processed.
                # It also expects a prediction_frame_template (just loc_x, loc_y for UEs)
                # which is current_ue_df_template.
                cell_specific_config_df = self.site_config_df_state[
                    self.site_config_df_state[getattr(c,'CELL_ID','cell_id')] == cell_id
                ]
                if cell_specific_config_df.empty:
                    logger.error(f"Gym: Could not find config for active cell {cell_id}. Skipping.")
                    continue

                # create_prediction_frames returns a dict: {cell_id: prediction_df_with_features}
                # We need to ensure the input template has loc_x, loc_y
                # The output of create_prediction_frames will have the X_COLUMNS for the BDT
                prediction_df_with_features_map = BayesianDigitalTwin.create_prediction_frames(
                    site_config_df=cell_specific_config_df,
                    prediction_frame_template=current_ue_df_template # Has loc_x, loc_y
                )
                
                # Check if the map was created and has the cell_id
                if cell_id not in prediction_df_with_features_map or prediction_df_with_features_map[cell_id].empty:
                    logger.warning(f"Gym: Failed to create prediction frame for cell {cell_id}.")
                    continue

                # The GymBDTCellPredictor's predict_distributed_gpmodel expects a list of DFs
                # and it modifies the DF in place.
                # It returns (pred_means, pred_stds), but we primarily care about the modified DF.
                df_for_prediction = prediction_df_with_features_map[cell_id]
                try:
                    bdt_predictor.predict_distributed_gpmodel(prediction_dfs=[df_for_prediction])
                    # df_for_prediction now has RXPOWER_DBM and RXPOWER_STDDEV_DBM columns
                    all_predictions_list.append(df_for_prediction)
                except Exception as e:
                    logger.error(f"Gym: Error during BDT prediction for cell {cell_id}: {e}")
        
        if not active_cell_count:
            logger.warning("Gym: All cells are off. No RF predictions to make.")
            return pd.DataFrame() # Return empty DataFrame
            
        if not all_predictions_list:
            logger.warning("Gym: No successful RF predictions from active cells.")
            return None

        return pd.concat(all_predictions_list, ignore_index=True)


    def _calculate_combined_reward(self, cell_selected_rf: Optional[pd.DataFrame], energy_consumption_metric: float) -> Tuple[float, Dict]:
        """Calculates the combined reward based on multiple objectives."""
        # Initialize info dict with default penalty values
        info = {
            "coverage_score": 0.0, # Range 0-100 (higher is better)
            "load_balance_score": -10.0, # Range e.g. -50 to 0 (-stdev, higher is better)
            "qos_score": 0.0, # Range 0-100 (higher is better, based on 1 - penalty)
            "raw_neg_stdev": -10.0,
            "bad_qos_ratio": 1.0,
            "energy_metric": energy_consumption_metric # Already calculated
        }
        # Default high penalty if RF simulation failed or no UEs attached
        if cell_selected_rf is None or cell_selected_rf.empty:
            logger.warning(f"Gym Tick {self.current_tick_of_day}: No RF data for reward. High penalty.")
            return -200.0, info # Large penalty

        try:
            # 1. Coverage Score (e.g., percentage of UEs with RSRP > threshold)
            # Using CcoEngine.rf_to_coverage_dataframe to get 'covered' status
            # This assumes CcoEngine is adapted or its output is suitable
            cov_df = CcoEngine.rf_to_coverage_dataframe(
                rf_dataframe=cell_selected_rf,
                weak_coverage_threshold=self.weak_coverage_threshold,
                over_coverage_threshold=self.over_coverage_threshold # Used by CcoEngine
            )
            # Example: % UEs that are 'covered' (not weak, not over)
            # coverage_score = cov_df['covered'].mean() * 100.0
            # Simpler: % UEs NOT weakly covered
            coverage_score = (1.0 - cov_df['weakly_covered'].mean()) * 100.0
            info["coverage_score"] = coverage_score

            # 2. Load Balancing Score (-stdev of UEs per *active* cell)
            active_cell_ids = self.site_config_df_state[self.on_off_state == 1][getattr(c,'CELL_ID','cell_id')].tolist()
            if active_cell_ids:
                # Filter cell_selected_rf for UEs served by active cells only for load balancing calc
                active_rf_for_load = cell_selected_rf[cell_selected_rf[getattr(c,'CELL_ID','cell_id')].isin(active_cell_ids)]
                # Create a temporary topology_df with only active cells for get_load_balancing_objective
                active_topology_df = self.site_config_df_state[self.site_config_df_state[getattr(c,'CELL_ID','cell_id')].isin(active_cell_ids)]

                if not active_rf_for_load.empty and not active_topology_df.empty:
                    raw_neg_stdev = CcoEngine.get_load_balancing_objective(
                        rf_dataframe=active_rf_for_load,
                        topology_df=active_topology_df # Pass only active cells
                    )
                    info["raw_neg_stdev"] = raw_neg_stdev
                    # Clip to prevent extreme penalties if stdev is huge
                    info["load_balance_score"] = max(raw_neg_stdev, -50.0) # Maximize this (closer to 0 is better)
                else: # No UEs attached to active cells, or no active cells
                    info["load_balance_score"] = 0 # Perfect balance if no load/no active cells
                    info["raw_neg_stdev"] = 0
            else: # All cells are off
                info["load_balance_score"] = 0 # Perfect balance if no active cells
                info["raw_neg_stdev"] = 0


            # 3. QoS Score (based on SINR, higher is better)
            # Example: 100 * (1 - (penalty for % UEs below SINR threshold))
            sinr_ok = cell_selected_rf['sinr_db'] >= self.qos_sinr_threshold
            bad_qos_ratio = 1.0 - sinr_ok.mean()
            info["bad_qos_ratio"] = bad_qos_ratio
            # QoS score: 100 if all UEs meet threshold, 0 if all UEs fail (linearly scaled)
            # Or, more simply, 100 * (percentage of UEs with good QoS)
            qos_score = sinr_ok.mean() * 100.0
            info["qos_score"] = qos_score

            # 4. Combine Rewards
            w_cov = self.reward_weights.get('coverage', 1.0)
            w_load = self.reward_weights.get('load_balance', 1.0)
            w_qos = self.reward_weights.get('qos', 1.0)
            w_energy = self.reward_weights.get('energy', 1.0) # Weight for energy saving

            # Energy saving metric: (1 - proportion of ON cells) * 100
            # energy_consumption_metric is already (proportion_ON_cells * MAX_ENERGY)
            # So, a reward for energy saving would be:
            # (1 - (energy_consumption_metric / (self.num_cells * self.ENERGY_MAX_PER_CELL))) * 100
            # Or simpler: -energy_consumption_metric (to minimize it)
            # The Gym's original reward used: self.lambda_ * -1.0 * energy_consumption
            # Let's use a positive reward for saving energy:
            # Max energy saving is when all cells are off (energy_consumption_metric = 0)
            # Min energy saving is when all cells are on (energy_consumption_metric = self.num_cells * MAX_ENERGY_PER_CELL)
            # Let's use a normalized energy saving score (0-100)
            max_possible_consumption = self.num_cells * self.ENERGY_MAX_PER_CELL if self.num_cells > 0 else self.ENERGY_MAX_PER_CELL
            if max_possible_consumption == 0: max_possible_consumption = 1 # Avoid div by zero
            
            energy_saving_score = (1 - (energy_consumption_metric / max_possible_consumption)) * 100
            info["energy_saving_score"] = energy_saving_score


            reward = (w_cov * info["coverage_score"] +
                      w_load * info["load_balance_score"] + # load_balance_score is already -stdev
                      w_qos * info["qos_score"] +
                      w_energy * energy_saving_score)
            
            if not np.isfinite(reward):
                logger.warning(f"Non-finite reward: {reward}. Components: {info}. Clipping.")
                reward = -1000.0

        except Exception as e:
            logger.exception(f"Gym Tick {self.current_tick_of_day}: Error calculating reward components: {e}")
            reward = -500.0 # Penalize calculation errors

        info["reward_total"] = reward
        logger.debug(f"Gym Tick {self.current_tick_of_day}: Reward={reward:.3f}, Info={{{k: (f'{v:.3f}' if isinstance(v, float) else v) for k,v in info.items()}}}")
        return reward, info


    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.current_step_in_episode = 0
        self.current_tick_of_day = self.observation_space.sample() # Start at a random hour
        # self.current_tick_of_day = 0 # Or always start at tick 0

        # Reset cell states to initial from site_config_df_initial
        self.site_config_df_state = self.site_config_df_initial.copy()
        self.on_off_state = np.ones(self.num_cells, dtype=int)
        self.tilt_state = self.site_config_df_state[getattr(c,'CELL_EL_DEG','cell_el_deg')].values.copy()

        logger.debug(f"Gym Env Reset. Start Tick: {self.current_tick_of_day}")
        return self.current_tick_of_day, {} # Return observation and empty info

    def step(self, action: np.ndarray):
        current_eval_tick = self.current_tick_of_day
        logger.debug(f"Gym Step: Current Tick {current_eval_tick}, Action: {action}")

        # 1. Apply action (update self.on_off_state, self.tilt_state, self.site_config_df_state)
        self._take_action(action)

        # 2. Get UE data for the current_eval_tick
        ue_df_for_this_tick = self.ue_data_per_tick.get(current_eval_tick)

        # 3. Get RF predictions for active cells using current UE data
        #    _get_rf_predictions_for_active_cells uses self.site_config_df_state (which has new tilts)
        #    and current_ue_df_template (UEs for this tick)
        cell_selected_rf_overall = None
        if ue_df_for_this_tick is not None and not ue_df_for_this_tick.empty:
            # The _get_rf_predictions_for_active_cells needs a "prediction_frame_template"
            # which is just the UE loc_x, loc_y.
            # It then calls BayesianDigitalTwin.create_prediction_frames internally for each active cell.
            all_predictions_df = self._get_rf_predictions_for_active_cells(ue_df_for_this_tick)

            if all_predictions_df is not None and not all_predictions_df.empty:
                # Perform attachment across predictions from all active cells
                active_site_config = self.site_config_df_state[self.on_off_state == 1]
                if not active_site_config.empty:
                    cell_selected_rf_overall = perform_attachment(all_predictions_df, active_site_config)
                else: # All cells were off
                    cell_selected_rf_overall = pd.DataFrame() # Empty df
            else: # No predictions or all cells off
                 cell_selected_rf_overall = pd.DataFrame()
        else:
            logger.warning(f"Gym Tick {current_eval_tick}: No UE data available for predictions.")
            cell_selected_rf_overall = pd.DataFrame()


        # 4. Calculate energy consumption metric for the observation & reward
        # This is based on the self.on_off_state updated by _take_action
        energy_consumption_metric = (
            self.ENERGY_MAX_PER_CELL * sum(self.on_off_state) # Number of ON cells * max energy
        )

        # 5. Calculate reward
        reward, reward_info = self._calculate_combined_reward(cell_selected_rf_overall, energy_consumption_metric)

        # 6. Prepare next observation (next tick) and episode termination
        self.current_step_in_episode += 1
        self.current_tick_of_day = (current_eval_tick + 1) % 24 # Cycle through 24 hours
        next_observation = self.current_tick_of_day

        done = self.current_step_in_episode >= self.horizon
        truncated = False # Not using truncation based on step limit within an episode for now

        # Populate info dictionary
        info = reward_info
        info["tick"] = current_eval_tick # Log the tick for which action was taken
        info["action"] = action.tolist() # Log the action taken
        info["energy_consumption_metric_obs"] = energy_consumption_metric # For observation if needed

        return next_observation, reward, done, truncated, info # Gymnasium expects truncated

    def render(self): pass
    def close(self): logger.info("Closing EnergySavingsGym."); pass

# === RL Trainer Script (`rl_energy_saver_trainer.py`) ===
def run_rl_training():
    logger.info("--- Starting RL Energy Saver Training Script ---")

    # --- Configuration for Training ---
    # Paths (reuse from global or define specifically)
    # TOPOLOGY_FILE_PATH, CONFIG_FILE_PATH are already defined globally
    UE_DATA_GYM_READY_DIR = "./ue_data_gym_ready" # Output of preprocess_ue_for_gym.py
    
    # BDT_MODEL_ID is global. LOCAL_MODEL_ACCESS_PATH is where the trained BDT is.
    if not os.path.exists(LOCAL_MODEL_ACCESS_PATH):
        logger.error(f"TRAINING ERROR: BDT Model file not found at {LOCAL_MODEL_ACCESS_PATH}. Train BDT first or check path.")
        return

    # Reward Weights for RL (CRITICAL - TUNE THESE)
    REWARD_WEIGHTS_RL = {
        'coverage': 1.0,       # e.g., for % good coverage
        'load_balance': 2.0,   # e.g., for -stdev (higher weight means more emphasis)
        'qos': 1.5,            # e.g., for QoS score (100 - penalty)
        'energy': 3.0          # e.g., for (1 - normalized_consumption) * 100
    }
    # RL Training Hyperparameters
    TOTAL_TRAINING_TIMESTEPS_RL = 50000 # Start small for testing, e.g., 24 * 1000
    RL_ALGORITHM = PPO
    RL_POLICY = "MlpPolicy"
    RL_LOG_DIR = "./rl_training_logs/"
    RL_MODEL_SAVE_PATH = "./rl_energy_saver_agent" # Agent will be saved here
    CHECKPOINT_FREQ_RL = 10000 # Save model checkpoint every N steps

    os.makedirs(RL_LOG_DIR, exist_ok=True)

    # --- Load Topology and Initial Config ---
    try:
        topology_df = pd.read_csv(TOPOLOGY_FILE_PATH)
        initial_config_df = pd.read_csv(CONFIG_FILE_PATH) # For Gym's site_config_df
        # Merge initial_config_df with topology_df to create site_config_df for Gym
        site_config_df_for_gym = pd.merge(
            topology_df.copy(),
            initial_config_df[[getattr(c,'CELL_ID','cell_id'), getattr(c,'CELL_EL_DEG','cell_el_deg')]],
            on=getattr(c,'CELL_ID','cell_id'),
            how='left'
        )
        site_config_df_for_gym[getattr(c,'CELL_EL_DEG','cell_el_deg')].fillna(TILT_SET[len(TILT_SET)//2], inplace=True)
        # Add hTx, hRx etc. if not present, as Gym's create_prediction_frames needs them
        DEFAULT_HTX = 25.0; DEFAULT_HRX = 1.5; DEFAULT_AZIMUTH = 0.0; DEFAULT_FREQUENCY = 2100
        for col, default_val in [
            (getattr(c,'HTX','hTx'), DEFAULT_HTX), (getattr(c,'HRX','hRx'), DEFAULT_HRX),
            (getattr(c,'CELL_AZ_DEG','cell_az_deg'), DEFAULT_AZIMUTH),
            (getattr(c,'CELL_CARRIER_FREQ_MHZ','cell_carrier_freq_mhz'), DEFAULT_FREQUENCY)
        ]:
            if col not in site_config_df_for_gym.columns: site_config_df_for_gym[col] = default_val

    except Exception as e: logger.error(f"Error loading data for RL training: {e}"); return

    # --- Load BDT Model States for Gym ---
    try:
        loaded_bdt_map = BayesianDigitalTwin.load_models_from_state(LOCAL_MODEL_ACCESS_PATH)
        if not loaded_bdt_map: logger.error("Failed to load BDT states for RL Gym."); return
        
        gym_bdt_predictors = {}
        for cell_id, data in loaded_bdt_map.items():
            gym_bdt_predictors[cell_id] = GymBDTCellPredictor(
                data['gp_model_state'], data['likelihood_state'], data['metadata']
            )
        logger.info(f"Prepared {len(gym_bdt_predictors)} BDT predictors for RL Gym.")
    except Exception as e: logger.exception(f"Error preparing BDT predictors for RL Gym: {e}"); return

    # --- Create RL Environment ---
    logger.info("Creating EnergySavingsGym for RL training...")
    try:
        # For RL training, we will use the per-tick files directly
        env = EnergySavingsGym(
            bayesian_digital_twins=gym_bdt_predictors,
            site_config_df=site_config_df_for_gym,
            ue_data_dir=UE_DATA_GYM_READY_DIR, # Gym will load these per tick
            tilt_set=TILT_SET,
            reward_weights=REWARD_WEIGHTS_RL,
            weak_coverage_threshold=WEAK_COVERAGE_THRESHOLD_GYM,
            over_coverage_threshold=OVER_COVERAGE_THRESHOLD_GYM,
            qos_sinr_threshold=0.0, # Example
            max_bad_qos_ratio=0.1, # Allow up to 10% bad QoS before penalty ramps up
            lambda_reward=LAMBDA_WEIGHT_GYM, # Original lambda, might be less relevant with new reward
            horizon=GYM_HORIZON, # Episode length = 1 day
            debug=False # Set True for very verbose Gym step logs
        )
        env = Monitor(env, RL_LOG_DIR) # Wrap for SB3 logging
        # check_env(env) # Optional: Sanity check
        logger.info("EnergySavingsGym for RL training created successfully.")
    except Exception as e: logger.exception(f"Failed to create RL environment: {e}"); return

    # --- Define and Train RL Agent ---
    logger.info(f"Defining RL agent ({RL_ALGORITHM.__name__}) with policy {RL_POLICY}...")
    model = RL_ALGORITHM(
        RL_POLICY,
        env,
        verbose=1,
        tensorboard_log=RL_LOG_DIR
        # Add other hyperparameters for PPO if needed
    )
    checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ_RL, save_path=RL_LOG_DIR, name_prefix="rl_energy_agent")
    logger.info(f"Starting RL agent training for {TOTAL_TRAINING_TIMESTEPS_RL} timesteps...")
    try:
        model.learn(total_timesteps=TOTAL_TRAINING_TIMESTEPS_RL, callback=checkpoint_callback, tb_log_name="EnergySaverRun")
        logger.info("RL Training finished.")
        model.save(RL_MODEL_SAVE_PATH)
        logger.info(f"Trained RL model saved to {RL_MODEL_SAVE_PATH}.zip")
    except Exception as e: logger.exception(f"Error during RL training: {e}")
    finally: env.close()
    logger.info("--- RL Energy Saver Training Script Finished ---")


# === RL Predictor Script (`rl_energy_saver_predictor.py`) ===
def run_rl_prediction(model_path: str, topology_path: str, target_tick: int):
    logger.info(f"--- Running RL Energy Saver Prediction for Tick {target_tick} ---")
    COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')

    if not (0 <= target_tick <= 23):
        logger.error("Target tick must be between 0 and 23."); return

    # Load Topology to get cell order and number of cells
    try:
        topology_df = pd.read_csv(topology_path)
        if COL_CELL_ID not in topology_df.columns: raise ValueError("Topology missing cell_id")
        cell_ids_ordered = topology_df[COL_CELL_ID].unique().tolist()
    except Exception as e: logger.error(f"Error loading topology {topology_path}: {e}"); return

    # Load Trained RL Model
    try:
        if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(f"RL Model not found at {model_path} or {model_path}.zip")
        # If path doesn't include .zip, SB3 often appends it
        model_load_path = model_path if model_path.endswith(".zip") else model_path + ".zip"
        rl_model = PPO.load(model_load_path) # Use the same algorithm (PPO)
        logger.info(f"Loaded trained RL model from {model_load_path}")
    except Exception as e: logger.error(f"Error loading RL model: {e}"); return

    # Predict action for the target tick
    # The observation for EnergySavingsGym is simply the tick number
    observation = target_tick
    action, _ = rl_model.predict(observation, deterministic=True)
    logger.info(f"Predicted raw action for tick {target_tick}: {action}")

    # Map action to human-readable configuration
    # Action: array of integers. Index len(TILT_SET) means OFF.
    # Others are indices into TILT_SET.
    config_list = []
    for i, cell_action_idx in enumerate(action):
        cell_id = cell_ids_ordered[i]
        if cell_action_idx == len(TILT_SET): # Special index for OFF
            state = "OFF"
            tilt = "N/A" # Or previous tilt
        else:
            state = "ON"
            tilt = TILT_SET[cell_action_idx] if 0 <= cell_action_idx < len(TILT_SET) else "InvalidAction"
        config_list.append({"cell_id": cell_id, "state": state, "cell_el_deg": tilt})
    
    predicted_config_df = pd.DataFrame(config_list)
    print("\n--- Predicted Optimal Configuration ---")
    print(f"--- For Tick/Hour: {target_tick} ---")
    print(predicted_config_df.to_string(index=False))

    # Here, you could optionally run a simulation with this predicted_config_df
    # using the BDT model to see the expected RF performance, load, QoS, etc.
    # This would involve:
    # 1. Loading the BDT model map (like in the trainer or demo app)
    # 2. Preparing UE data for the target_tick
    # 3. Calling _run_backend_simulation or similar logic from CCO_RL_Env/EnergySavingsGym
    # 4. Calculating and printing the metrics.


if __name__ == "__main__":
    # This main block can be used to switch between training and prediction
    # For simplicity, we'll call training. Prediction would typically be separate.

    # --- Ensure necessary files/dirs exist for training ---
    if not os.path.exists(TOPOLOGY_FILE_PATH): logger.error(f"Missing for training: {TOPOLOGY_FILE_PATH}"); sys.exit(1)
    if not os.path.exists(LOCAL_MODEL_ACCESS_PATH): # BDT model file
        logger.error(f"Missing BDT model for training: {LOCAL_MODEL_ACCESS_PATH}. Run BDT training first or ensure path is correct.");
        # sys.exit(1) # Allow to proceed if user only wants to generate data/predict with dummy model
    if not os.path.isdir(UE_DATA_GYM_READY_DIR):
        logger.error(f"Missing preprocessed UE data for Gym: {UE_DATA_GYM_READY_DIR}. Run preprocess_ue_for_gym.py first.");
        # sys.exit(1)

    # --- Run Training ---
    run_rl_training()

    # --- Example of how to run prediction afterwards ---
    # logger.info("\n\n--- Example Prediction Run (after training) ---")
    # PREDICTION_TICK = 10 # Example tick to predict for
    # run_rl_prediction(
    #     model_path=RL_MODEL_SAVE_PATH, # Path where trained RL agent was saved
    #     topology_path=TOPOLOGY_FILE_PATH,
    #     target_tick=PREDICTION_TICK
    # )
