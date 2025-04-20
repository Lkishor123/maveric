import gymnasium as gym # Use gymnasium (successor to gym)
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import logging
import math

# --- Assume previous helper imports & setup ---
# Need RADPClient, RADPHelper, constants (c), GISTools, perform_attachment
# Need CcoEngine (potentially modified or with new methods)
# Need to configure logger

# Example placeholder imports (replace with actual)
from radp.client.client import RADPClient
from radp.client.helper import RADPHelper, SimulationStatus
from radp.digital_twin.utils.cell_selection import perform_attachment
from radp.digital_twin.utils import constants as c
# Assuming CcoEngine has the necessary methods, including get_load_balancing_objective
from apps.coverage_capacity_optimization.cco_engine import CcoEngine

# --- Environment Definition ---

class CCO_RL_Env(gym.Env):
    """
    Custom Gym Environment for Reinforcement Learning based CCO.

    Observation: Current hour of the day (0-23).
    Action: Electrical tilt (0-20) for each cell.
    Reward: Combination of coverage, load balance, and QoS metrics.
    """
    metadata = {'render_modes': [], 'render_fps': 4} # Required by Gym API

    def __init__(self,
                 topology_df: pd.DataFrame,
                 ue_data_dir: str, # Directory containing generated_ue_data_for_cco_{tick}.csv
                 bayesian_digital_twin_id: str,
                 reward_weights: Dict[str, float], # e.g., {'coverage': 1.0, 'load': 0.5, 'qos': 0.2}
                 radp_client: RADPClient, # Use Dependency Injection
                 radp_helper: RADPHelper, # Use Dependency Injection
                 possible_tilts: List[float] = list(np.arange(0.0, 21.0, 1.0)) # 0-20 degrees
                ):
        super().__init__()

        self.topology_df = topology_df.copy()
        self.cell_ids = self.topology_df[c.CELL_ID].unique().tolist()
        self.num_cells = len(self.cell_ids)
        self.ue_data_dir = ue_data_dir
        self.bayesian_digital_twin_id = bayesian_digital_twin_id
        self.radp_client = radp_client
        self.radp_helper = radp_helper
        self.reward_weights = reward_weights
        self.possible_tilts = possible_tilts
        self.num_tilt_options = len(possible_tilts)

        # --- Define Action and Observation Spaces ---
        # Action: Set tilt (index 0 to 20) for each cell
        self.action_space = spaces.MultiDiscrete([self.num_tilt_options] * self.num_cells)

        # Observation: Hour of the day (0 to 23)
        self.observation_space = spaces.Discrete(24)

        # --- Pre-load UE data (if feasible, otherwise load in step) ---
        self.ue_data_per_tick: Dict[int, pd.DataFrame] = self._load_all_ue_data()
        if not self.ue_data_per_tick:
            raise FileNotFoundError(f"No UE data files found or loaded from {ue_data_dir}")

        # Environment state
        self.current_tick = 0
        self._current_config_id_counter = 0 # For backend simulation tracking

        logger.info(f"CCO RL Env initialized. Num Cells: {self.num_cells}, Action Space: {self.action_space}, Obs Space: {self.observation_space}")

    def _load_all_ue_data(self) -> Dict[int, pd.DataFrame]:
        """Loads per-tick UE data into memory."""
        data = {}
        logger.info(f"Loading per-tick UE data from {self.ue_data_dir}...")
        try:
            for tick in range(24): # Expecting files for ticks 0-23
                filename = f"generated_ue_data_for_cco_{tick}.csv"
                filepath = os.path.join(self.ue_data_dir, filename)
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    # Basic validation
                    if 'lat' in df.columns and 'lon' in df.columns:
                         # Keep only necessary columns to save memory?
                         data[tick] = df[['lat', 'lon']].copy()
                    else:
                         logger.warning(f"Skipping {filename}: Missing lat/lon columns.")
                else:
                    logger.warning(f"UE data file not found for tick {tick}: {filepath}")
            logger.info(f"Loaded UE data for {len(data)} ticks.")
            return data
        except Exception as e:
            logger.exception(f"Error loading UE data: {e}")
            return {} # Return empty dict on error

    def _get_next_config_id(self) -> int:
        """Increments and returns the config ID for simulation tracking."""
        self._current_config_id_counter += 1
        return self._current_config_id_counter

    def _map_action_to_config(self, action: np.ndarray) -> pd.DataFrame:
        """Converts the agent's action (array of tilt indices) to a config DataFrame."""
        if len(action) != self.num_cells:
            raise ValueError(f"Action length ({len(action)}) does not match num_cells ({self.num_cells})")

        config_data = []
        for i, cell_id in enumerate(self.cell_ids):
            tilt_index = action[i]
            if not (0 <= tilt_index < self.num_tilt_options):
                 logger.warning(f"Action contained invalid tilt index {tilt_index} for cell {i}. Clamping to valid range [0, {self.num_tilt_options-1}]")
                 tilt_index = np.clip(tilt_index, 0, self.num_tilt_options - 1)

            tilt_value = self.possible_tilts[tilt_index]
            config_data.append({c.CELL_ID: cell_id, c.CELL_EL_DEG: tilt_value})

        return pd.DataFrame(config_data)

    def _run_backend_simulation(self, config_df: pd.DataFrame, ue_data_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Runs simulation on backend, performs attachment, returns attached RF dataframe or None on failure."""
        if ue_data_df is None or ue_data_df.empty:
            logger.error("Cannot run simulation: UE data is empty.")
            return None

        config_id_for_sim = self._get_next_config_id()
        sim_event_template: Dict[str, Any] = { # Define template locally or access from self if needed
            "simulation_time_interval_seconds": 1,
            "ue_tracks": {"ue_data_id": f"eval_ues_tick_{self.current_tick}"},
            "rf_prediction": {"model_id": self.bayesian_digital_twin_id, "config_id": config_id_for_sim},
        }
        logger.debug(f"Running simulation for tick {self.current_tick}, config_id: {config_id_for_sim}")

        try:
            sim_response = self.radp_client.simulation(
                simulation_event=sim_event_template,
                ue_data=ue_data_df,
                config=config_df,
            )
            simulation_id = sim_response.get("simulation_id")
            if not simulation_id: raise ValueError("Backend did not return simulation_id.")
            logger.info(f"Sim triggered (Tick {self.current_tick}): ID {simulation_id}")

            # TODO: Externalize polling parameters
            status: SimulationStatus = self.radp_helper.resolve_simulation_status(simulation_id, wait_interval=1, max_attempts=100, verbose=False)
            if not status.success: raise Exception(f"Sim {simulation_id} failed: {status.error_message}")
            logger.info(f"Sim {simulation_id} completed.")

            rf_full = self.radp_client.consume_simulation_output(simulation_id)
            if rf_full is None or rf_full.empty: logger.warning(f"Sim output {simulation_id} empty."); return None

            # Cell attachment needs RSRP/SINR - ensure backend provides them
            # Assuming backend provides columns needed by perform_attachment
            if 'rsrp_dbm' not in rf_full.columns or 'sinr_db' not in rf_full.columns:
                 logger.error(f"Simulation output missing required RSRP/SINR columns.")
                 return None

            cell_selected_rf = perform_attachment(rf_full, self.topology_df)
            logger.debug(f"Cell attachment complete for Sim {simulation_id}.")
            return cell_selected_rf

        except Exception as e:
            logger.exception(f"Error during backend simulation/processing for tick {self.current_tick}: {e}")
            return None # Return None to indicate failure

    def _calculate_reward(self, cell_selected_rf: Optional[pd.DataFrame]) -> Tuple[float, Dict]:
        """Calculates the combined reward based on simulation results."""
        if cell_selected_rf is None or cell_selected_rf.empty:
            # Penalize heavily for simulation failure or no attached UEs
            return -100.0, {"coverage_score": 0, "load_score": -10, "qos_penalty": 10} # Example penalty values

        # --- 1. Coverage Score ---
        # Example: Use average network utility from CcoEngine
        try:
            # TODO: Make thresholds configurable
            coverage_df = CcoEngine.rf_to_coverage_dataframe(
                rf_dataframe=cell_selected_rf,
                weak_coverage_threshold=-95, # Example thresholds
                over_coverage_threshold=-65
            )
            # Aggregate utility (higher is better, max ~1000 based on tanh scaling?)
            # Normalize maybe? Let's use mean utility for now.
            coverage_score = coverage_df["network_coverage_utility"].mean()
            # Alternative: % Good Coverage = coverage_df["covered"].mean()
        except Exception as e:
            logger.error(f"Error calculating coverage score: {e}")
            coverage_score = 0 # Default penalty

        # --- 2. Load Balancing Score ---
        # Use the negative standard deviation (higher is better)
        try:
            load_balancing_score = CcoEngine.get_load_balancing_objective(
                rf_dataframe=cell_selected_rf, # Already attached
                topology_df=self.topology_df
            ) # Returns -stdev
        except Exception as e:
             logger.error(f"Error calculating load balancing score: {e}")
             load_balancing_score = -10 # Default penalty (high stdev)

        # --- 3. QoS Penalty ---
        # Example: Penalize if > X% of UEs have SINR < threshold
        qos_penalty = 0
        try:
            # TODO: Make threshold configurable
            sinr_threshold = 0 # dB
            bad_qos_ues = cell_selected_rf[cell_selected_rf['sinr_db'] < sinr_threshold]
            bad_qos_ratio = len(bad_qos_ues) / len(cell_selected_rf) if len(cell_selected_rf) > 0 else 0
            # Simple penalty: linearly increase penalty above 5% bad QoS
            qos_penalty_factor = 10 # Example scaling factor
            qos_penalty = max(0, bad_qos_ratio - 0.05) * qos_penalty_factor
        except Exception as e:
            logger.error(f"Error calculating QoS penalty: {e}")
            qos_penalty = 1 # Default penalty


        # --- Combine Rewards ---
        # Reward = w1 * Coverage + w2 * LoadBalance - w3 * QoSPenalty
        # Note: load_balancing_score is already negative (-stdev)
        # Need careful tuning of weights!
        w_cov = self.reward_weights.get('coverage', 1.0)
        w_load = self.reward_weights.get('load', 1.0) # Weight for -stdev term
        w_qos = self.reward_weights.get('qos', 1.0)

        reward = (w_cov * coverage_score) + (w_load * load_balancing_score) - (w_qos * qos_penalty)

        # Store individual components for logging/debugging
        info = {
            "reward_total": reward,
            "coverage_score": coverage_score,
            "load_score": load_balancing_score, # (-stdev)
            "qos_penalty": qos_penalty,
            "bad_qos_ratio": bad_qos_ratio,
        }

        logger.debug(f"Tick {self.current_tick}: Reward={reward:.3f}, Info={info}")
        return reward, info


    def reset(self, seed=None, options=None):
        """Resets the environment to the start of a day (tick 0)."""
        super().reset(seed=seed) # Important for reproducibility

        self.current_tick = 0
        observation = self.current_tick
        info = {} # No extra info needed on reset

        logger.debug(f"Environment reset. Start Tick: {self.current_tick}")
        return observation, info

    def step(self, action: np.ndarray):
        """Executes one step in the environment."""
        current_sim_tick = self.current_tick # Tick for which action is applied

        # 1. Map action to physical configuration
        config_df = self._map_action_to_config(action)

        # 2. Get UE data for the *current* tick
        ue_data_df = self.ue_data_per_tick.get(current_sim_tick)
        if ue_data_df is None:
            logger.error(f"Missing UE data for tick {current_sim_tick}. Returning error state.")
            # Return minimal reward, indicate error state? Gym API expects obs, reward, terminated, truncated, info
            observation = (self.current_tick + 1) % 24 # Move to next tick
            reward = -1000 # Very high penalty
            terminated = False # Or True to end episode on error?
            truncated = False
            info = {"error": f"Missing UE data for tick {current_sim_tick}"}
            self.current_tick = observation
            return observation, reward, terminated, truncated, info


        # 3. Run backend simulation
        cell_selected_rf = self._run_backend_simulation(config_df, ue_data_df)

        # 4. Calculate reward based on simulation results
        reward, info = self._calculate_reward(cell_selected_rf)
        # Add config info for debugging if needed
        info["action_taken"] = action.tolist()


        # 5. Transition to the next state (next hour)
        self.current_tick = (current_sim_tick + 1) % 24
        observation = self.current_tick

        # 6. Define termination/truncation conditions
        terminated = False # Episode runs indefinitely in cycles of 24h unless stopped externally
        truncated = False # Could set True after N steps if desired

        return observation, reward, terminated, truncated, info

    def render(self):
        """Rendering is not implemented for this environment."""
        pass

    def close(self):
        """Clean up any resources (e.g., close network connections if client has them)."""
        logger.info("Closing CCO RL Environment.")
        # Add cleanup if needed (e.g., self.radp_client.close())
        pass