# train_rl_cco.py
# Script to train a Reinforcement Learning agent for CCO
# MODIFIED to use a locally loaded BDT model for RF simulation

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import sys
import logging
import math
from typing import Dict, List, Tuple, Optional, Any

# --- Path Setup ---
RADP_ROOT = os.getenv("MAVERIC_ROOT", "/home/lk/Projects/accelcq-repos/cloudly/github/maveric") # MODIFY IF NEEDED

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not os.path.isdir(RADP_ROOT):
     potential_path = os.path.join(os.path.dirname(__file__), "..", "..")
     if os.path.isdir(os.path.join(potential_path, "radp")): RADP_ROOT = os.path.abspath(potential_path); print(f"Warning: RADP_ROOT assumed: {RADP_ROOT}")
     else: raise FileNotFoundError(f"RADP_ROOT directory not found: {RADP_ROOT}.")
sys.path.insert(0, RADP_ROOT)

try:
    from radp.client.client import RADPClient
    from radp.client.helper import RADPHelper, SimulationStatus
    from radp.digital_twin.utils.cell_selection import perform_attachment
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin, ExactGPModel, NormMethod
    sys.path.insert(0, os.path.join(RADP_ROOT, "apps"))
    from coverage_capacity_optimization.cco_engine import CcoEngine
    # logger.info("Successfully imported RADP and CCO modules.") # Moved to main
except ImportError as e:
    print(f"FATAL: Error importing RADP/CCO modules: {e}. Check RADP_ROOT and ensure modules exist.")
    sys.exit(1)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback
except ImportError:
    print("FATAL: stable-baselines3 not found. Please install it: pip install stable-baselines3")
    sys.exit(1)

# --- Global Configuration for Training Script ---
DATA_DIR_TRAINER = "./"
TOPOLOGY_FILE_TRAINER = os.path.join(DATA_DIR_TRAINER, "data", "topology.csv")
CONFIG_FILE_PATH_TRAINER = os.path.join(DATA_DIR_TRAINER, "data", "config.csv")
UE_DATA_DIR_TRAINER = os.path.join(DATA_DIR_TRAINER, "data", "ue_data_gym_ready")
BDT_MODEL_FILE_PATH_TRAINER = os.path.join(DATA_DIR_TRAINER, "model.pickle")

REWARD_WEIGHTS_TRAINER = {'coverage': 1.0, 'load': 5.0, 'qos': 10.0}
TOTAL_TRAINING_TIMESTEPS_TRAINER = 2000
PPO_POLICY_TRAINER = "MlpPolicy"
LOG_DIR_TRAINER = "./rl_logs/"
MODEL_SAVE_PATH_TRAINER = "./cco_rl_agent_ppo_local_sim"
CHECKPOINT_FREQ_TRAINER = 100
POSSIBLE_TILTS_TRAINER = list(np.arange(0.0, 21.0, 1.0))
QOS_SINR_THRESHOLD_TRAINER = 0.0
MAX_BAD_QOS_RATIO_TRAINER = 0.10
WEAK_COVERAGE_THRESHOLD_REWARD_TRAINER = -95.0
OVER_COVERAGE_THRESHOLD_REWARD_TRAINER = -65.0
GYM_HORIZON_TRAINER = 24

# Defaults for missing topology columns
DEFAULT_HTX = 25.0
DEFAULT_HRX = 1.5
DEFAULT_CELL_AZ_DEG = 0.0
DEFAULT_CELL_CARRIER_FREQ_MHZ = 2100.0


# --- Topology Generation Function ---
def generate_dummy_topology(
    num_sites: int,
    cells_per_site: int = 3,
    lat_range: Tuple[float, float] = (40.7, 40.8),
    lon_range: Tuple[float, float] = (-74.05, -73.95),
    start_ecgi: int = 1001,
    start_enodeb_id: int = 1,
    default_tac: int = 1,
    default_freq: int = DEFAULT_CELL_CARRIER_FREQ_MHZ, # Use global default
    default_power_dbm: float = 25.0,
    azimuth_step: int = 120,
    default_htx: float = DEFAULT_HTX, # Use global default
    default_hrx: float = DEFAULT_HRX  # Use global default
) -> pd.DataFrame:
    logger.info(f"Generating dummy topology for {num_sites} sites with {cells_per_site} cells each.")
    topology_data = []
    current_ecgi = start_ecgi
    current_enodeb_id = start_enodeb_id

    COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')
    COL_CELL_LAT = getattr(c, 'CELL_LAT', 'cell_lat')
    COL_CELL_LON = getattr(c, 'CELL_LON', 'cell_lon')
    COL_CELL_AZ_DEG = getattr(c, 'CELL_AZ_DEG', 'cell_az_deg')
    COL_CELL_TXPWR_DBM = getattr(c, 'CELL_TXPWR_DBM', 'cell_txpwr_dbm')
    COL_ECGI = getattr(c, 'ECGI', 'ecgi')
    COL_SITE_ID = getattr(c, 'SITE_ID', 'site_id')
    COL_CELL_NAME = getattr(c, 'CELL_NAME', 'cell_name')
    COL_ENODEB_ID = getattr(c, 'ENODEB_ID', 'enodeb_id')
    COL_TAC = getattr(c, 'TAC', 'tac')
    COL_CELL_CARRIER_FREQ_MHZ = getattr(c, 'CELL_CARRIER_FREQ_MHZ', 'cell_carrier_freq_mhz')
    COL_HTX = getattr(c, 'HTX', 'hTx') # Get constant name
    COL_HRX = getattr(c, 'HRX', 'hRx') # Get constant name


    for i in range(num_sites):
        site_lat = np.random.uniform(lat_range[0], lat_range[1])
        site_lon = np.random.uniform(lon_range[0], lon_range[1])
        site_id_str = f"Site{i+1}"

        for j in range(cells_per_site):
            cell_az = (j * azimuth_step) % 360
            cell_id_str = f"cell_{current_enodeb_id}_{cell_az}"
            cell_name_str = f"Cell{j+1}"

            row = {
                COL_ECGI: current_ecgi,
                COL_SITE_ID: site_id_str,
                COL_CELL_NAME: cell_name_str,
                COL_ENODEB_ID: current_enodeb_id,
                COL_CELL_AZ_DEG: cell_az,
                COL_TAC: default_tac,
                COL_CELL_LAT: site_lat,
                COL_CELL_LON: site_lon,
                COL_CELL_ID: cell_id_str,
                COL_CELL_CARRIER_FREQ_MHZ: default_freq,
                COL_CELL_TXPWR_DBM: default_power_dbm + np.random.uniform(-2, 2),
                COL_HTX: default_htx, # Add hTx
                COL_HRX: default_hrx   # Add hRx
            }
            topology_data.append(row)
        current_ecgi += 1
        current_enodeb_id += 1

    df = pd.DataFrame(topology_data)
    column_order = [
        COL_ECGI, COL_SITE_ID, COL_CELL_NAME, COL_ENODEB_ID, COL_CELL_AZ_DEG, COL_TAC,
        COL_CELL_LAT, COL_CELL_LON, COL_CELL_ID, COL_CELL_CARRIER_FREQ_MHZ,
        COL_CELL_TXPWR_DBM, COL_HTX, COL_HRX # Add to column order
    ]
    df = df.reindex(columns=column_order)
    logger.info(f"Generated dummy topology DataFrame with {len(df)} cells.")
    return df


# === Custom Gym Environment Definition ===
class CCO_RL_Env(gym.Env):
    # ... (CCO_RL_Env class definition as in your provided script A) ...
    # (Make sure the imports like `perform_attachment` are within this class or globally available to it)
    metadata = {'render_modes': [], 'render_fps': 4}

    def __init__(self, topology_df: pd.DataFrame, ue_data_dir: str,
                 bdt_model_file_path: str, # Path to the pickled BDT model map
                 reward_weights: Dict[str, float],
                 possible_tilts: List[float] = list(np.arange(0.0, 21.0, 1.0)),
                 qos_sinr_threshold: float = 0.0,
                 max_bad_qos_ratio: float = 0.05,
                 weak_coverage_threshold_reward: float = -95.0,
                 over_coverage_threshold_reward: float = -65.0,
                 horizon: int = 24
                ):
        super().__init__()
        self.COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')
        self.COL_CELL_EL_DEG = getattr(c, 'CELL_EL_DEG', 'cell_el_deg')
        self.COL_LAT = getattr(c, 'LAT', 'lat')
        self.COL_LON = getattr(c, 'LON', 'lon')
        self.COL_RSRP_DBM = getattr(c, 'RSRP_DBM', 'rsrp_dbm')
        self.COL_SINR_DB = getattr(c, 'SINR_DB', 'sinr_db')
        self.COL_LOC_X = getattr(c, 'LOC_X', 'loc_x')
        self.COL_LOC_Y = getattr(c, 'LOC_Y', 'loc_y')


        self.topology_df = topology_df.copy() # This topology_df MUST have hTx, hRx etc.
        self.cell_ids = self.topology_df[self.COL_CELL_ID].unique().tolist()
        self.num_cells = len(self.cell_ids)
        self.ue_data_dir = ue_data_dir
        
        logger.info(f"Loading BDT model map from: {bdt_model_file_path}")
        if not os.path.exists(bdt_model_file_path):
            raise FileNotFoundError(f"BDT Model file not found at: {bdt_model_file_path}")
        self.bdt_predictors: Dict[str, BayesianDigitalTwin] = BayesianDigitalTwin.load_model_map_from_pickle(bdt_model_file_path)
        if not self.bdt_predictors or not isinstance(self.bdt_predictors, dict):
            raise ValueError("Failed to load a valid BDT model map (dictionary of BayesianDigitalTwin objects).")
        logger.info(f"Loaded {len(self.bdt_predictors)} BDT cell models.")
        missing_bdt_cells = [cid for cid in self.cell_ids if cid not in self.bdt_predictors]
        if missing_bdt_cells:
            raise ValueError(f"Missing BDT models for cells in topology: {missing_bdt_cells}")

        self.reward_weights = reward_weights
        self.possible_tilts = possible_tilts
        self.num_tilt_options = len(possible_tilts)
        self.qos_sinr_threshold = qos_sinr_threshold
        self.max_bad_qos_ratio = max_bad_qos_ratio
        self.weak_coverage_threshold_reward = weak_coverage_threshold_reward
        self.over_coverage_threshold_reward = over_coverage_threshold_reward
        self.horizon = horizon

        self.action_space = spaces.MultiDiscrete([self.num_tilt_options] * self.num_cells)
        self.observation_space = spaces.Discrete(24)

        self.ue_data_per_tick: Dict[int, Optional[pd.DataFrame]] = self._load_all_ue_data()
        at_least_one_df_loaded = any(df is not None and not df.empty for df in self.ue_data_per_tick.values())
        if not at_least_one_df_loaded:
            raise FileNotFoundError(f"No valid UE data files successfully loaded from {self.ue_data_dir}")

        self.current_tick = 0
        logger.info(f"CCO RL Env initialized. Num Cells: {self.num_cells}")

    def _load_all_ue_data(self) -> Dict[int, Optional[pd.DataFrame]]:
        data = {}
        logger.info(f"Loading per-tick UE data from {self.ue_data_dir}...")
        found_count = 0
        for tick_val in range(24): 
            filename = f"generated_ue_data_for_cco_{tick_val}.csv" 
            filepath = os.path.join(self.ue_data_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    input_lon_col = 'lon' if 'lon' in df.columns else getattr(c, 'LOC_X', 'loc_x')
                    input_lat_col = 'lat' if 'lat' in df.columns else getattr(c, 'LOC_Y', 'loc_y')
                    if input_lon_col not in df.columns and self.COL_LON in df.columns: input_lon_col = self.COL_LON
                    if input_lat_col not in df.columns and self.COL_LAT in df.columns: input_lat_col = self.COL_LAT
                    if input_lon_col in df.columns and input_lat_col in df.columns:
                         df_gym = df[[input_lon_col, input_lat_col, 'mock_ue_id']].copy()
                         df_gym.rename(columns={input_lon_col: 'loc_x', input_lat_col: 'loc_y'}, inplace=True)
                         if 'lon' not in df_gym.columns and self.COL_LON in df.columns : df_gym['lon'] = df[self.COL_LON]
                         if 'lat' not in df_gym.columns and self.COL_LAT in df.columns : df_gym['lat'] = df[self.COL_LAT]
                         data[tick_val] = df_gym
                         found_count += 1
                    else: logger.warning(f"Skipping {filename}: Missing location columns.")
                except Exception as e: logger.error(f"Error loading {filepath}: {e}"); data[tick_val] = None
            else: logger.warning(f"UE data file not found for tick {tick_val}: {filepath}"); data[tick_val] = None
        logger.info(f"Loaded UE data for {found_count}/24 ticks.")
        return data

    def _map_action_to_config(self, action: np.ndarray) -> pd.DataFrame:
        if len(action) != self.num_cells: raise ValueError("Action length mismatch")
        config_data = []
        for i, cell_id in enumerate(self.cell_ids):
            tilt_index = np.clip(action[i], 0, self.num_tilt_options - 1)
            tilt_value = self.possible_tilts[tilt_index]
            config_data.append({self.COL_CELL_ID: cell_id, self.COL_CELL_EL_DEG: tilt_value})
        return pd.DataFrame(config_data)

    def _run_local_simulation_with_bdt(self, config_df: pd.DataFrame, ue_data_df_template: pd.DataFrame) -> Optional[pd.DataFrame]:
        if ue_data_df_template is None or ue_data_df_template.empty:
            logger.error("Local sim: Input UE data template is empty.")
            return None
        all_cell_predictions = []
        logger.debug(f"Local sim for tick {self.current_tick}: Processing {self.num_cells} cells.")
        for i, cell_id in enumerate(self.cell_ids):
            bdt_predictor = self.bdt_predictors.get(cell_id)
            if not bdt_predictor:
                logger.error(f"Local sim: BDT predictor for cell {cell_id} not found. Skipping."); continue
            current_cell_config_series = config_df[config_df[self.COL_CELL_ID] == cell_id].iloc[0]
            single_cell_site_config_for_bdt = self.topology_df[self.topology_df[self.COL_CELL_ID] == cell_id].copy()
            if single_cell_site_config_for_bdt.empty:
                 logger.error(f"Local sim: Topology data for cell {cell_id} not found. Skipping."); continue
            single_cell_site_config_for_bdt[self.COL_CELL_EL_DEG] = current_cell_config_series[self.COL_CELL_EL_DEG]
            try:
                prediction_frames_dict = BayesianDigitalTwin.create_prediction_frames(
                    site_config_df=single_cell_site_config_for_bdt,
                    prediction_frame_template=ue_data_df_template
                )
            except Exception as e: logger.error(f"Local sim: Error in create_prediction_frames for cell {cell_id}: {e}"); continue
            if cell_id not in prediction_frames_dict or prediction_frames_dict[cell_id].empty:
                logger.warning(f"Local sim: create_prediction_frames no data for cell {cell_id}."); continue
            df_to_predict_on = prediction_frames_dict[cell_id]
            try:
                bdt_predictor.predict_distributed_gpmodel(prediction_dfs=[df_to_predict_on])
                all_cell_predictions.append(df_to_predict_on)
            except Exception as e: logger.error(f"Local sim: BDT prediction error for cell {cell_id}: {e}")
        if not all_cell_predictions: logger.warning("Local sim: No successful BDT predictions."); return None
        combined_rf_predictions = pd.concat(all_cell_predictions, ignore_index=True)
        if combined_rf_predictions.empty: logger.warning("Local sim: Combined RF predictions empty."); return None
        try:
            cell_selected_rf = perform_attachment(combined_rf_predictions, self.topology_df)
            logger.debug(f"Local sim tick {self.current_tick} attachment complete.")
        except Exception as e: logger.exception(f"Local sim: Attachment error tick {self.current_tick}: {e}"); return None
        if cell_selected_rf is None or cell_selected_rf.empty:
            logger.warning(f"Local sim: Attachment tick {self.current_tick} empty result."); return None
        if self.COL_RSRP_DBM not in cell_selected_rf.columns or self.COL_SINR_DB not in cell_selected_rf.columns:
            logger.error(f"Local sim: Attached RF columns: {cell_selected_rf.columns.tolist()}")
            raise ValueError("Attached RF missing RSRP/SINR after perform_attachment.")
        return cell_selected_rf

    def _calculate_load_balancing_objective(self, rf_dataframe: pd.DataFrame, topology_df_for_load: pd.DataFrame) -> float:
        if rf_dataframe is None or rf_dataframe.empty or topology_df_for_load.empty: return 0.0
        ue_counts = rf_dataframe.groupby(self.COL_CELL_ID).size()
        all_cell_ids = topology_df_for_load[self.COL_CELL_ID].unique()
        ue_counts_all = ue_counts.reindex(all_cell_ids, fill_value=0)
        if len(ue_counts_all) <= 1: return 0.0
        return -ue_counts_all.std()

    def _calculate_reward(self, cell_selected_rf: Optional[pd.DataFrame]) -> Tuple[float, Dict]:
        info = {"coverage_score": 0.0, "load_balance_score": -20.0, "qos_score": 0.0,
                "raw_neg_stdev": -20.0, "bad_qos_ratio": 1.0}
        if cell_selected_rf is None or cell_selected_rf.empty:
            logger.warning(f"Tick {self.current_tick}: Reward calc: empty/failed RF data. High penalty.")
            return -100.0, info
        try:
            cov_df = CcoEngine.rf_to_coverage_dataframe(
                rf_dataframe=cell_selected_rf,
                weak_coverage_threshold=self.weak_coverage_threshold_reward,
                over_coverage_threshold=self.over_coverage_threshold_reward
            )
            info["coverage_score"] = (1.0 - cov_df['weakly_covered'].mean()) * 100.0
            raw_neg_stdev = self._calculate_load_balancing_objective(cell_selected_rf, self.topology_df)
            info["raw_neg_stdev"] = raw_neg_stdev; info["load_balance_score"] = max(raw_neg_stdev, -50.0)
            if self.COL_SINR_DB in cell_selected_rf.columns and not cell_selected_rf[self.COL_SINR_DB].empty:
                good_qos_ratio = (cell_selected_rf[self.COL_SINR_DB] >= self.qos_sinr_threshold).mean()
                info["bad_qos_ratio"] = 1.0 - good_qos_ratio; info["qos_score"] = good_qos_ratio * 100.0
            else: info["bad_qos_ratio"] = 1.0; info["qos_score"] = 0.0
            reward = (self.reward_weights.get('coverage', 0) * info["coverage_score"] +
                      self.reward_weights.get('load', 0) * info["load_balance_score"] +
                      self.reward_weights.get('qos', 0) * info["qos_score"])
            if not np.isfinite(reward): reward = -1000.0
        except Exception as e: logger.exception(f"Reward calc error tick {self.current_tick}: {e}"); reward = -500.0
        info["reward_total"] = reward
        info_log_str = ", ".join([f"{k_}: {v_:.3f}" if isinstance(v_, float) else f"{k_}: {v_}" for k_, v_ in info.items()])
        logger.debug(f"Tick {self.current_tick}: Reward={reward:.3f}, Info={{{info_log_str}}}")
        return reward, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_tick = self.np_random.integers(0, 24)
        observation = self.current_tick
        info = {}
        logger.debug(f"Env Reset. Start Tick: {self.current_tick}")
        return observation, info

    def step(self, action: np.ndarray):
        current_eval_tick = self.current_tick
        config_df = self._map_action_to_config(action)
        ue_data_df_for_tick = self.ue_data_per_tick.get(current_eval_tick)
        terminated = False; truncated = False
        if ue_data_df_for_tick is None or ue_data_df_for_tick.empty:
            reward = -1000.0; info = {"error": f"Missing UE data tick {current_eval_tick}"}
        else:
            cell_selected_rf_overall = self._run_local_simulation_with_bdt(config_df, ue_data_df_for_tick)
            reward, info = self._calculate_reward(cell_selected_rf_overall)
        info["tick"] = current_eval_tick; info["config"] = config_df[self.COL_CELL_EL_DEG].tolist()
        self.current_tick = (current_eval_tick + 1) % 24
        observation = self.current_tick
        # An episode is one full day (24 ticks)
        done = (self.current_tick == 0 and current_eval_tick == 23) # Done after completing tick 23
        return observation, reward, done, truncated, info

    def render(self): pass
    def close(self): logger.info("Closing CCO RL Env."); pass

# === Main Training Script Logic ===
if __name__ == "__main__":
    logger.info("Successfully imported RADP and CCO modules.") # Moved here
    logger.info("--- Starting CCO RL Agent Training Script (with Local BDT Simulation) ---")
    os.makedirs(LOG_DIR_TRAINER, exist_ok=True)
    if MODEL_SAVE_PATH_TRAINER and os.path.dirname(MODEL_SAVE_PATH_TRAINER) and not os.path.exists(os.path.dirname(MODEL_SAVE_PATH_TRAINER)):
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH_TRAINER))

    logger.info("Initializing components...")
    try:
        topology = pd.read_csv(TOPOLOGY_FILE_TRAINER)
        if getattr(c, 'CELL_ID', 'cell_id') not in topology.columns: raise ValueError("Topology missing cell_id")
        
        # Add missing columns to topology if they don't exist (hTx, hRx, cell_az_deg, cell_carrier_freq_mhz)
        # These are needed by BayesianDigitalTwin.create_prediction_frames
        cols_to_check_in_topo = {
            getattr(c, 'HTX', 'hTx'): DEFAULT_HTX,
            getattr(c, 'HRX', 'hRx'): DEFAULT_HRX,
            getattr(c, 'CELL_AZ_DEG', 'cell_az_deg'): DEFAULT_CELL_AZ_DEG,
            getattr(c, 'CELL_CARRIER_FREQ_MHZ', 'cell_carrier_freq_mhz'): DEFAULT_CELL_CARRIER_FREQ_MHZ
        }
        for col, default_val in cols_to_check_in_topo.items():
            if col not in topology.columns:
                logger.warning(f"Topology missing '{col}'. Adding default value: {default_val}")
                topology[col] = default_val
        
        # RADPClient and RADPHelper are NOT used by the local BDT simulation environment
        # radp_client = RADPClient()
        # radp_helper = RADPHelper(radp_client)
        # logger.info("RADP Client/Helper (if used by BDT internals) initialized.") # Not needed now
    except FileNotFoundError: logger.error(f"Topology file not found: {TOPOLOGY_FILE_TRAINER}"); sys.exit(1)
    except Exception as e: logger.error(f"Initialization error: {e}"); sys.exit(1)

    logger.info("Creating CCO RL Environment...")
    try:
        env = CCO_RL_Env(
            topology_df=topology,
            ue_data_dir=UE_DATA_DIR_TRAINER,
            bdt_model_file_path=BDT_MODEL_FILE_PATH_TRAINER,
            reward_weights=REWARD_WEIGHTS_TRAINER,
            # radp_client and radp_helper are removed as _run_local_simulation_with_bdt doesn't use them
            possible_tilts=POSSIBLE_TILTS_TRAINER,
            qos_sinr_threshold=QOS_SINR_THRESHOLD_TRAINER,
            max_bad_qos_ratio=MAX_BAD_QOS_RATIO_TRAINER,
            weak_coverage_threshold_reward=WEAK_COVERAGE_THRESHOLD_REWARD_TRAINER,
            over_coverage_threshold_reward=OVER_COVERAGE_THRESHOLD_REWARD_TRAINER,
            horizon=GYM_HORIZON_TRAINER
        )
        env = Monitor(env, LOG_DIR_TRAINER)
        logger.info("Environment created successfully.")
    except FileNotFoundError as e:
        logger.error(f"Failed to create RL environment: {e}. Ensure BDT model file exists at '{BDT_MODEL_FILE_PATH_TRAINER}'.")
        sys.exit(1)
    except Exception as e: logger.error(f"Failed to create RL environment: {e}"); sys.exit(1)

    logger.info(f"Defining PPO agent with policy {PPO_POLICY_TRAINER}...")
    model = PPO(PPO_POLICY_TRAINER, env, verbose=1, tensorboard_log=LOG_DIR_TRAINER)
    checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ_TRAINER, save_path=LOG_DIR_TRAINER, name_prefix="cco_rl_model_local_sim")
    
    logger.info(f"Starting training for {TOTAL_TRAINING_TIMESTEPS_TRAINER} timesteps...")
    try:
        model.learn(total_timesteps=TOTAL_TRAINING_TIMESTEPS_TRAINER, callback=checkpoint_callback, tb_log_name="PPO_CCO_LocalSim_Run")
        logger.info("Training finished.")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e: logger.exception(f"Error during training: {e}")
    finally:
        try:
            model.save(MODEL_SAVE_PATH_TRAINER)
            logger.info(f"Successfully saved trained model to {MODEL_SAVE_PATH_TRAINER}.zip")
        except Exception as e: logger.error(f"Failed to save final model: {e}")
        env.close()
    logger.info("--- Training Script Finished ---")

