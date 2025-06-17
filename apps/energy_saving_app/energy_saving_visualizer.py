# energy_saving_visualizer.py

import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from stable_baselines3 import PPO
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
    from radp.digital_twin.utils.cell_selection import perform_attachment
except ImportError as e:
    print(f"FATAL: Error importing libraries: {e}."); sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TILT_SET = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

class EnergySavingVisualizer:
    """
    Visualizes network performance, comparing a baseline vs. an RL-optimized configuration.
    """
    def __init__(self, bdt_model_path: str, rl_model_path: str, topology_path: str, config_path: str, base_ue_data_dir: str):
        self.base_ue_data_dir = base_ue_data_dir
        
        topology_df = pd.read_csv(topology_path)
        config_df = pd.read_csv(config_path)
        self.site_config_df_base = pd.merge(topology_df, config_df, on='cell_id', how='left')
        self.site_config_df_base['cell_el_deg'].fillna(TILT_SET[len(TILT_SET)//2], inplace=True)
        
        required_cols = {'hTx': 25.0, 'hRx': 1.5, 'cell_az_deg': 0.0, 'cell_carrier_freq_mhz': 2100.0}
        for const, val in required_cols.items():
            col = getattr(c, const, const.lower())
            if col not in self.site_config_df_base.columns:
                self.site_config_df_base[col] = val
        
        logger.info("Loading BDT model map..."); self.bdt_model_map = BayesianDigitalTwin.load_model_map_from_pickle(bdt_model_path)
        logger.info(f"Loaded BDT map for {len(self.bdt_model_map)} cells.")

        logger.info("Loading RL agent..."); rl_model_path_zip = rl_model_path if rl_model_path.endswith(".zip") else f"{rl_model_path}.zip"
        self.rl_model = PPO.load(rl_model_path_zip)
        logger.info("RL agent loaded.")
        
        self.COL_LON = getattr(c, 'LOC_X', 'loc_x'); self.COL_LAT = getattr(c, 'LOC_Y', 'loc_y')
        self.COL_CELL_LON = getattr(c, 'CELL_LON', 'cell_lon'); self.COL_CELL_LAT = getattr(c, 'CELL_LAT', 'cell_lat')
        self.COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id'); self.COL_UE_ID = 'ue_id'
        self.COL_RXPOWER_DBM = getattr(c, 'RXPOWER_DBM', 'rxpower_dbm'); self.COL_RSRP_DBM = getattr(c, 'RSRP_DBM', 'rsrp_dbm')

    def _run_local_simulation(self, ue_data: pd.DataFrame, site_config: pd.DataFrame) -> pd.DataFrame:
        """Runs a local RF simulation and performs cell attachment."""
        all_preds_list = []
        active_cell_ids = site_config[self.COL_CELL_ID].unique()
        for cell_id in active_cell_ids:
            bdt_predictor = self.bdt_model_map.get(cell_id)
            if not bdt_predictor: continue
            
            cell_cfg_df = site_config[site_config[self.COL_CELL_ID] == cell_id]
            pred_frames = BayesianDigitalTwin.create_prediction_frames(site_config_df=cell_cfg_df, prediction_frame_template=ue_data)
            df_for_pred = pred_frames.get(cell_id)
            if df_for_pred is not None and not df_for_pred.empty:
                bdt_predictor.predict_distributed_gpmodel(prediction_dfs=[df_for_pred])
                all_preds_list.append(df_for_pred)
        
        if not all_preds_list: return pd.DataFrame()
        combined_preds = pd.concat(all_preds_list, ignore_index=True)
        return perform_attachment(combined_preds, site_config)

    def _plot_scenario(self, ax, title: str, ue_data: pd.DataFrame, attached_data: pd.DataFrame,
                       active_towers: pd.DataFrame, inactive_towers: pd.DataFrame, partial_towers: pd.DataFrame):
        """Helper to generate a single subplot."""
        ax.set_title(title, fontsize=16)
        
        # Plot towers based on their activity status first
        ax.scatter(active_towers[self.COL_CELL_LON], active_towers[self.COL_CELL_LAT], marker='^', c='green', s=150, edgecolors='black', label='Fully Active Sites', zorder=10)
        ax.scatter(inactive_towers[self.COL_CELL_LON], inactive_towers[self.COL_CELL_LAT], marker='^', c='red', s=150, alpha=0.7, edgecolors='black', label='Fully Inactive Sites', zorder=10)
        ax.scatter(partial_towers[self.COL_CELL_LON], partial_towers[self.COL_CELL_LAT], marker='^', c='yellow', s=150, edgecolors='black', label='Partially Active Sites', zorder=10)

        # Plot UEs colored by RSRP level
        if not attached_data.empty:
            plot_df = pd.merge(ue_data, attached_data[[self.COL_UE_ID, self.COL_RSRP_DBM, self.COL_CELL_ID]], on=self.COL_UE_ID, how="left")
            plot_df.rename(columns={self.COL_CELL_ID: 'serving_cell_id'}, inplace=True)
            
            served_ues = plot_df.dropna(subset=[self.COL_RSRP_DBM])
            cmap = plt.get_cmap('viridis_r'); vmin, vmax = -120, -70
            sc = ax.scatter(served_ues[self.COL_LON], served_ues[self.COL_LAT], c=served_ues[self.COL_RSRP_DBM], cmap=cmap, s=10, alpha=0.9, vmin=vmin, vmax=vmax, zorder=5)
            plt.colorbar(sc, ax=ax, label='RSRP (dBm)', fraction=0.046, pad=0.04)

            no_serve_ues = plot_df[plot_df['serving_cell_id'].isna()]
            ax.scatter(no_serve_ues[self.COL_LON], no_serve_ues[self.COL_LAT], c='darkorange', marker='x', s=25, label='Disconnected UEs', zorder=6)
        else: # Case where all cells might be off
            ax.scatter(ue_data[self.COL_LON], ue_data[self.COL_LAT], c='darkorange', marker='x', s=25, label='Disconnected UEs', zorder=6)
        
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(True, linestyle='--', alpha=0.4); ax.legend(loc='best', fontsize='small')

    def generate_comparison_plots(self, day: int, tick: int, output_dir: str):
        ue_data_dir = os.path.join(self.base_ue_data_dir, f"Day_{day}", "ue_data_gym_ready")
        ue_data_file = os.path.join(ue_data_dir, f"generated_ue_data_for_cco_{tick}.csv")
        if not os.path.exists(ue_data_file): logger.error(f"Test UE data file not found: {ue_data_file}"); return
        
        ue_data_df = pd.read_csv(ue_data_file)
        ue_data_df[self.COL_UE_ID] = range(len(ue_data_df))

        # --- Site-level logic ---
        site_id_col = getattr(c, 'SITE_ID', 'site_id')
        if site_id_col not in self.site_config_df_base.columns:
            self.site_config_df_base['site_id_temp'] = self.site_config_df_base[self.COL_CELL_ID].str.rsplit('_', n=2).str[1]
            site_id_col = 'site_id_temp'
        site_locations = self.site_config_df_base.drop_duplicates(subset=[site_id_col]).copy()

        # --- 1. Baseline Scenario ---
        logger.info("Simulating baseline scenario (initial tilts, all cells on)...")
        baseline_attached_df = self._run_local_simulation(ue_data_df, self.site_config_df_base)
        baseline_active_towers = site_locations; baseline_inactive_towers = pd.DataFrame(); baseline_partial_towers = pd.DataFrame()

        # --- 2. Optimized Scenario ---
        logger.info("Simulating RL-optimized scenario...")
        action_indices, _ = self.rl_model.predict(tick, deterministic=True)
        
        opt_config_df = self.site_config_df_base.copy()
        active_cell_ids = []
        for i, cell_action_idx in enumerate(action_indices):
            cell_id = self.site_config_df_base[self.COL_CELL_ID].iloc[i]
            if cell_action_idx < len(TILT_SET):
                active_cell_ids.append(cell_id)
                opt_config_df.loc[opt_config_df[self.COL_CELL_ID] == cell_id, getattr(c,'CELL_EL_DEG','cell_el_deg')] = TILT_SET[cell_action_idx]

        active_topology_for_sim = opt_config_df[opt_config_df[self.COL_CELL_ID].isin(active_cell_ids)]
        optimized_attached_df = self._run_local_simulation(ue_data_df, active_topology_for_sim)
        
        fully_active_sites, fully_inactive_sites, partially_active_sites = [], [], []
        for site_id, group in self.site_config_df_base.groupby(site_id_col):
            active_cells_in_site = set(group[self.COL_CELL_ID]).intersection(active_cell_ids)
            if len(active_cells_in_site) == 0: fully_inactive_sites.append(site_id)
            elif len(active_cells_in_site) == len(group): fully_active_sites.append(site_id)
            else: partially_active_sites.append(site_id)
        
        opt_active_towers = site_locations[site_locations[site_id_col].isin(fully_active_sites)]
        opt_inactive_towers = site_locations[site_locations[site_id_col].isin(fully_inactive_sites)]
        opt_partial_towers = site_locations[site_locations[site_id_col].isin(partially_active_sites)]

        # --- 3. Generate Plots ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), sharex=True, sharey=True)
        fig.suptitle(f'Energy Saving Comparison for Test Day {day}, Tick {tick}', fontsize=20)

        self._plot_scenario(ax1, 'Baseline: Initial Configuration', ue_data_df, baseline_attached_df, 
                            baseline_active_towers, baseline_inactive_towers, baseline_partial_towers)
        self._plot_scenario(ax2, 'Optimized: RL Agent Configuration', ue_data_df, optimized_attached_df,
                            opt_active_towers, opt_inactive_towers, opt_partial_towers)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"energy_saving_comparison_day{day}_tick_{tick}.png")
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Comparison plot saved to: {output_path}")
        plt.close(fig)

