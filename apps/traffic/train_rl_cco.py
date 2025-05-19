# train_rl_cco.py
# Script to train a Reinforcement Learning agent for CCO

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import sys
import logging
import math
from typing import Dict, List, Tuple, Optional, Any


# --- Assume RADP imports & setup ---
# Ensure RADP_ROOT is set correctly in environment or script
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
    # Assuming CcoEngine is in the CCO app directory relative to this script's potential location
    sys.path.insert(0, os.path.join(RADP_ROOT, "apps")) # Add apps dir to path
    from coverage_capacity_optimization.cco_engine import CcoEngine
    logger.info("Successfully imported RADP and CCO modules.")
except ImportError as e:
    print(f"FATAL: Error importing RADP/CCO modules: {e}. Check RADP_ROOT and ensure modules exist.")
    sys.exit(1)

# --- RL Library Import (Example: Stable Baselines3) ---
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback
except ImportError:
    print("FATAL: stable-baselines3 not found. Please install it: pip install stable-baselines3")
    sys.exit(1)


# === Custom Gym Environment Definition ===
# (Ideally, this class should be in its own file, e.g., cco_rl_env.py and imported)
class CCO_RL_Env(gym.Env):
    """ Custom Gym Environment for Reinforcement Learning based CCO. """
    metadata = {'render_modes': [], 'render_fps': 4}

    def __init__(self, topology_df: pd.DataFrame, ue_data_dir: str,
                 bayesian_digital_twin_id: str, reward_weights: Dict[str, float],
                 radp_client: RADPClient, radp_helper: RADPHelper,
                 possible_tilts: List[float] = list(np.arange(0.0, 21.0, 1.0)),
                 qos_sinr_threshold: float = 0.0, # dB
                 max_bad_qos_ratio: float = 0.05 # Penalize above 5%
                ):
        super().__init__()
        # --- Essential Column Constants ---
        self.COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')
        self.COL_CELL_EL_DEG = getattr(c, 'CELL_EL_DEG', 'cell_el_deg')
        self.COL_LAT = getattr(c, 'LAT', 'lat')
        self.COL_LON = getattr(c, 'LON', 'lon')

        self.topology_df = topology_df.copy()
        self.cell_ids = self.topology_df[self.COL_CELL_ID].unique().tolist()
        self.num_cells = len(self.cell_ids)
        self.ue_data_dir = ue_data_dir
        self.bayesian_digital_twin_id = bayesian_digital_twin_id
        self.radp_client = radp_client
        self.radp_helper = radp_helper
        self.reward_weights = reward_weights
        self.possible_tilts = possible_tilts
        self.num_tilt_options = len(possible_tilts)
        self.qos_sinr_threshold = qos_sinr_threshold
        self.max_bad_qos_ratio = max_bad_qos_ratio

        # --- Define Action and Observation Spaces ---
        self.action_space = spaces.MultiDiscrete([self.num_tilt_options] * self.num_cells)
        self.observation_space = spaces.Discrete(24) # Hour 0-23

        self.ue_data_per_tick: Dict[int, Optional[pd.DataFrame]] = self._load_all_ue_data()
        at_least_one_df_loaded = any(df is not None for df in self.ue_data_per_tick.values())

        if not at_least_one_df_loaded: # If the flag is False (meaning all values were None or dict empty)
            raise FileNotFoundError(f"No valid UE data files were successfully loaded from {self.ue_data_dir}")

        self.current_tick = 0
        self._current_config_id_counter = 0
        logger.info(f"CCO RL Env initialized. Num Cells: {self.num_cells}")

    def _load_all_ue_data(self) -> Dict[int, Optional[pd.DataFrame]]:
        data = {}
        logger.info(f"Loading per-tick UE data from {self.ue_data_dir}...")
        found_count = 0
        for tick in range(24):
            filename = f"generated_ue_data_for_cco_{tick}.csv"
            filepath = os.path.join(self.ue_data_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    if self.COL_LAT in df.columns and self.COL_LON in df.columns:
                         # Keep only necessary columns + mock_ue_id for potential debugging
                         data[tick] = df[[self.COL_LAT, self.COL_LON, 'mock_ue_id']].copy()
                         found_count += 1
                    else: logger.warning(f"Skipping {filename}: Missing lat/lon columns.")
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {e}")
                    data[tick] = None # Mark as unloadable
            else: logger.warning(f"UE data file not found for tick {tick}: {filepath}"); data[tick] = None
        logger.info(f"Loaded UE data for {found_count}/24 ticks.")
        return data

    def _get_next_config_id(self) -> int:
        self._current_config_id_counter += 1; return self._current_config_id_counter

    def _map_action_to_config(self, action: np.ndarray) -> pd.DataFrame:
        if len(action) != self.num_cells: raise ValueError("Action length mismatch")
        config_data = []
        for i, cell_id in enumerate(self.cell_ids):
            tilt_index = np.clip(action[i], 0, self.num_tilt_options - 1) # Ensure valid index
            tilt_value = self.possible_tilts[tilt_index]
            config_data.append({self.COL_CELL_ID: cell_id, self.COL_CELL_EL_DEG: tilt_value})
        return pd.DataFrame(config_data)

    def _run_backend_simulation(self, config_df: pd.DataFrame, ue_data_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Runs simulation, performs attachment, returns attached RF df or None."""
        if ue_data_df is None or ue_data_df.empty: logger.error("Sim input UE data empty."); return None
        config_id = self._get_next_config_id()
        sim_event: Dict[str, Any] = {
            "simulation_time_interval_seconds": 1,
            "ue_tracks": {"ue_data_id": f"rl_eval_{self.current_tick}_{config_id}"},
            "rf_prediction": {"model_id": self.bayesian_digital_twin_id, "config_id": config_id},
        }
        logger.debug(f"Running sim: Tick {self.current_tick}, ConfigID {config_id}")
        try:
            sim_resp = self.radp_client.simulation(sim_event, ue_data_df, config_df)
            sim_id = sim_resp.get("simulation_id"); assert sim_id
            status = self.radp_helper.resolve_simulation_status(sim_id, wait_interval=1, max_attempts=120, verbose=False) # Increased attempts
            if not status.success: raise Exception(f"Sim {sim_id} failed: {status.error_message}")
            rf_full = self.radp_client.consume_simulation_output(sim_id)
            if rf_full is None or rf_full.empty: logger.warning(f"Sim {sim_id} output empty."); return None
            if 'rsrp_dbm' not in rf_full.columns or 'sinr_db' not in rf_full.columns: raise ValueError("Sim output missing RSRP/SINR.")
            cell_selected_rf = perform_attachment(rf_full, self.topology_df)
            logger.debug(f"Sim {sim_id} processed.")
            return cell_selected_rf
        except Exception as e: logger.exception(f"Backend sim error tick {self.current_tick}: {e}"); return None

    def _calculate_reward(self, cell_selected_rf: Optional[pd.DataFrame]) -> Tuple[float, Dict]:
        """Calculates combined reward. Higher is better."""
        info = {"coverage_score": 0.0, "load_score": -10.0, "qos_penalty": 10.0, "raw_neg_stdev": -10.0, "bad_qos_ratio": 1.0}
        if cell_selected_rf is None or cell_selected_rf.empty:
            logger.warning(f"Tick {self.current_tick}: Calculating reward based on empty RF data (sim failure?). Assigning high penalty.")
            # Return large negative reward if simulation failed
            return -100.0, info # Ensure penalty is significant

        try:
            # 1. Coverage Score (Example: Mean positive utility, penalize negatives?)
            # Assumes CcoEngine.rf_to_coverage_dataframe exists and works
            # We might simplify this for RL - e.g., % UEs above RSRP threshold?
            # Let's use % UEs NOT weakly covered for simplicity now.
            cov_df = CcoEngine.rf_to_coverage_dataframe(cell_selected_rf, weak_coverage_threshold=-95) # Use only weak threshold
            # Scale score from 0 to 100
            coverage_score = (1.0 - cov_df['weakly_covered'].mean()) * 100.0
            info["coverage_score"] = coverage_score

            # 2. Load Balancing Score (-stdev, higher is better)
            # Assumes CcoEngine.get_load_balancing_objective exists
            load_score = CcoEngine.get_load_balancing_objective(cell_selected_rf, self.topology_df)
            info["raw_neg_stdev"] = load_score # Store the raw -stdev
            # Normalize or scale? Std dev depends on num UEs/cells. Let's scale it relative to max possible stdev?
            # Simpler: just use the raw negative value, weighted. Max value is 0.
            # Let's bound it slightly to avoid huge negative rewards if stdev is large
            load_score_clipped = max(load_score, -50.0) # Cap penalty if stdev > 50
            info["load_score"] = load_score_clipped

            # 3. QoS Penalty (Penalize % UEs below SINR threshold)
            sinr_ok = cell_selected_rf['sinr_db'] >= self.qos_sinr_threshold
            bad_qos_ratio = 1.0 - sinr_ok.mean()
            info["bad_qos_ratio"] = bad_qos_ratio
            # Penalize linearly for exceeding the acceptable bad QoS ratio
            qos_penalty = max(0, bad_qos_ratio - self.max_bad_qos_ratio)
             # Scale penalty: e.g., if ratio is 10% over limit (0.15), penalty = 10*0.1 = 1
            qos_penalty_scaled = qos_penalty * 100 # Make penalty larger relative to scores
            info["qos_penalty"] = qos_penalty_scaled

            # 4. Combine: Maximize Coverage, Maximize Load Score (-stdev), Minimize QoS Penalty
            w_cov = self.reward_weights.get('coverage', 1.0)
            w_load = self.reward_weights.get('load', 1.0) # Applied to load_score_clipped
            w_qos = self.reward_weights.get('qos', 1.0) # Applied to qos_penalty_scaled

            reward = (w_cov * coverage_score) + (w_load * load_score_clipped) - (w_qos * qos_penalty_scaled)

            # Ensure reward is finite
            if not np.isfinite(reward):
                logger.warning(f"Non-finite reward calculated: {reward}. Components: {info}. Clipping.")
                reward = -1000.0 # Assign large penalty

        except Exception as e:
            logger.exception(f"Error calculating reward for tick {self.current_tick}: {e}")
            reward = -500.0 # Penalize calculation errors

        info["reward_total"] = reward
        logger.debug(f"Tick {self.current_tick}: Reward={reward:.3f}, Info={{{k: f'{v:.3f}' for k,v in info.items() if isinstance(v, (int, float))}}}")
        return reward, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_tick = self.np_random.integers(0, 24) # Start at random hour? Or always 0? Let's use 0 for now.
        # self.current_tick = 0
        observation = self.current_tick
        info = {}
        logger.debug(f"Env Reset. Start Tick: {self.current_tick}")
        return observation, info

    def step(self, action: np.ndarray):
        current_eval_tick = self.current_tick
        logger.debug(f"Step: Current Tick {current_eval_tick}, Action: {action}")

        config_df = self._map_action_to_config(action)
        ue_data_df = self.ue_data_per_tick.get(current_eval_tick)

        terminated = False # Episode runs over days/weeks during training
        truncated = False # No artificial truncation for now

        if ue_data_df is None:
            logger.error(f"Missing UE data for tick {current_eval_tick}. Returning high penalty.")
            reward = -1000.0
            info = {"error": f"Missing UE data tick {current_eval_tick}"}
            cell_selected_rf = None # Ensure info dict is populated below if needed
        else:
            cell_selected_rf = self._run_backend_simulation(config_df, ue_data_df)
            reward, info = self._calculate_reward(cell_selected_rf)

        # Add simulation results summary to info if needed for callbacks
        info["tick"] = current_eval_tick
        info["config"] = config_df[self.COL_CELL_EL_DEG].tolist() # Log applied tilts

        # Transition state
        self.current_tick = (current_eval_tick + 1) % 24
        observation = self.current_tick

        return observation, reward, terminated, truncated, info

    def render(self): pass
    def close(self): logger.info("Closing CCO RL Env."); pass

# === Main Training Script Logic ===
if __name__ == "__main__":
    logger.info("--- Starting CCO RL Agent Training Script ---")

    # --- Configuration ---
    TOPOLOGY_FILE = "./data/topology.csv" # Assumes topology exists
    UE_DATA_DIR = "./ue_data"             # Directory with per-tick UE CSVs
    # IMPORTANT: Get this ID after training the base RF model using cco_example_app or similar
    BAYESIAN_DIGITAL_TWIN_MODEL_ID = "cco_test_model" # REPLACE with your actual trained model ID

    # Reward Weights (NEEDS CAREFUL TUNING!)
    REWARD_WEIGHTS = {
        'coverage': 1.0, # Weight for (% Good Coverage * 100)
        'load': 5.0,     # Weight for (max(-stdev(load), -50)) - higher weight means care more about balancing
        'qos': 10.0      # Weight for penalty if bad_qos_ratio > 5%
    }

    # RL Training Hyperparameters
    TOTAL_TRAINING_TIMESTEPS = 200000 # Adjust based on complexity/convergence (e.g., 1e5, 1e6)
    PPO_POLICY = "MlpPolicy"          # Standard MLP policy for discrete obs, multidiscrete action
    LOG_DIR = "./rl_logs/"            # Directory for TensorBoard logs
    MODEL_SAVE_PATH = "./cco_rl_agent_ppo" # Path to save trained agent
    CHECKPOINT_FREQ = 10000          # Save model checkpoint every N steps

    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Initialization ---
    logger.info("Initializing components...")
    try:
        topology = pd.read_csv(TOPOLOGY_FILE)
        # Validate topology
        if getattr(c, 'CELL_ID', 'cell_id') not in topology.columns: raise ValueError("Topology missing cell_id")
        # Instantiate RADP Client and Helper (ensure .env or config is loaded for client)
        radp_client = RADPClient()
        radp_helper = RADPHelper(radp_client)
        logger.info("RADP Client/Helper initialized.")
    except FileNotFoundError:
        logger.error(f"Topology file not found: {TOPOLOGY_FILE}"); sys.exit(1)
    except Exception as e:
        logger.error(f"Initialization error: {e}"); sys.exit(1)

    # --- Create and Check Environment ---
    logger.info("Creating CCO RL Environment...")
    try:
        env = CCO_RL_Env(
            topology_df=topology,
            ue_data_dir=UE_DATA_DIR,
            bayesian_digital_twin_id=BAYESIAN_DIGITAL_TWIN_MODEL_ID,
            reward_weights=REWARD_WEIGHTS,
            radp_client=radp_client,
            radp_helper=radp_helper
            # Add optional params like possible_tilts if needed
        )
        # Wrap with Monitor for SB3 logging
        env = Monitor(env, LOG_DIR)
        # Check environment compatibility (optional but recommended)
        # check_env(env)
        logger.info("Environment created successfully.")
    except Exception as e:
        logger.error(f"Failed to create RL environment: {e}"); sys.exit(1)

    # --- Define Agent ---
    # PPO is a good default choice supporting MultiDiscrete actions
    # Hyperparameters might need tuning (learning_rate, n_steps, batch_size, etc.)
    logger.info(f"Defining PPO agent with policy {PPO_POLICY}...")
    model = PPO(
        PPO_POLICY,
        env,
        verbose=1, # Log training progress
        tensorboard_log=LOG_DIR,
        # Example hyperparameter adjustments (defaults are often reasonable starting points)
        # learning_rate=3e-4,
        # n_steps=2048,
        # batch_size=64,
        # n_epochs=10,
        # gamma=0.99,
        # gae_lambda=0.95,
        # clip_range=0.2,
        # ent_coef=0.0,
        # vf_coef=0.5,
        # max_grad_norm=0.5,
    )

    # --- Define Callbacks ---
    checkpoint_callback = CheckpointCallback(
      save_freq=CHECKPOINT_FREQ,
      save_path=LOG_DIR,
      name_prefix="cco_rl_model",
      save_replay_buffer=False, # Not applicable for PPO usually
      save_vecnormalize=True # Save VecNormalize stats if used
    )

    # --- Train Agent ---
    logger.info(f"Starting training for {TOTAL_TRAINING_TIMESTEPS} timesteps...")
    try:
        model.learn(
            total_timesteps=TOTAL_TRAINING_TIMESTEPS,
            callback=checkpoint_callback,
            log_interval=1, # Log stats every episode
            tb_log_name="PPO_CCO_Run" # Name for TensorBoard run
        )
        logger.info("Training finished.")
    except Exception as e:
        logger.exception(f"Error during training: {e}") # Log full traceback
        # Decide whether to save partially trained model?
        # model.save(f"{MODEL_SAVE_PATH}_interrupted")
        # logger.info(f"Saved partially trained model to {MODEL_SAVE_PATH}_interrupted")
    finally:
        # --- Save Final Model ---
        try:
            model.save(MODEL_SAVE_PATH)
            logger.info(f"Successfully saved trained model to {MODEL_SAVE_PATH}.zip")
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
        # --- Close Environment ---
        env.close()

    logger.info("--- Training Script Finished ---")