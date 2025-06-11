import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Path and Import Setup ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
APPS_DIR = os.path.dirname(APP_DIR)
PROJECT_ROOT = os.path.dirname(APPS_DIR)
sys.path.insert(0, PROJECT_ROOT)

import torch
import gpytorch
from stable_baselines3 import PPO

from radp.digital_twin.utils import constants as c
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin, ExactGPModel, NormMethod
from radp.digital_twin.utils.cell_selection import perform_attachment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TILT_SET = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]


class EnergySavingVisualizer:
    """
    Visualizes the impact of the energy-saving RL model by comparing
    network coverage before and after optimization.
    """

    def __init__(self, bdt_model_path: str, rl_model_path: str, topology_path: str, config_path: str, ue_data_path_template: str):
        self.ue_data_path_template = ue_data_path_template
        
        topology_df = pd.read_csv(topology_path)
        config_df = pd.read_csv(config_path)
        self.site_config_df = pd.merge(topology_df, config_df, on='cell_id', how='left')
        
        self.site_config_df[getattr(c, 'CELL_EL_DEG', 'cell_el_deg')].fillna(TILT_SET[len(TILT_SET)//2], inplace=True)
        required_cols = {'HTX': 25.0, 'HRX': 1.5, 'CELL_AZ_DEG': 0.0, 'CELL_CARRIER_FREQ_MHZ': 2100.0}
        for const, val in required_cols.items():
            col = getattr(c, const, const.lower())
            if col not in self.site_config_df.columns:
                self.site_config_df[col] = val
        
        logger.info("Loading BDT model map...")
        self.bdt_model_map = BayesianDigitalTwin.load_model_map_from_pickle(bdt_model_path)
        logger.info(f"Loaded BDT map for {len(self.bdt_model_map)} cells.")

        logger.info("Loading RL agent...")
        self.rl_model = PPO.load(rl_model_path)
        logger.info("RL agent loaded.")
        
        self.COL_LON = getattr(c, 'LOC_X', 'loc_x')
        self.COL_LAT = getattr(c, 'LOC_Y', 'loc_y')
        self.COL_CELL_LON = getattr(c, 'CELL_LON', 'cell_lon')
        self.COL_CELL_LAT = getattr(c, 'CELL_LAT', 'cell_lat')
        self.COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')
        self.COL_UE_ID = 'ue_id'
        self.COL_RXPOWER_DBM = getattr(c, 'RXPOWER_DBM', 'rxpower_dbm')


    def _get_rf_predictions(self, ue_data: pd.DataFrame, site_config: pd.DataFrame, active_cell_ids: List[str]) -> pd.DataFrame:
        """Runs RF predictions for a given set of active cells."""
        all_preds_list = []
        for cell_id in active_cell_ids:
            bdt_predictor = self.bdt_model_map.get(cell_id)
            if not bdt_predictor:
                continue
            
            cell_cfg_df = site_config[site_config[self.COL_CELL_ID] == cell_id]
            pred_frames = BayesianDigitalTwin.create_prediction_frames(
                site_config_df=cell_cfg_df, prediction_frame_template=ue_data
            )
            if cell_id in pred_frames and not pred_frames[cell_id].empty:
                df_for_pred = pred_frames[cell_id]
                bdt_predictor.predict_distributed_gpmodel(prediction_dfs=[df_for_pred])
                all_preds_list.append(df_for_pred)
        
        return pd.concat(all_preds_list, ignore_index=True) if all_preds_list else pd.DataFrame()

    def _plot_scenario(self, ax, title: str, ue_data: pd.DataFrame, serving_data: pd.DataFrame, 
                       active_towers: pd.DataFrame, inactive_towers: pd.DataFrame, partial_towers: pd.DataFrame):
        """A helper function to generate a single plot for a given scenario."""
        ax.set_title(title, fontsize=16)
        
        plot_df = pd.merge(ue_data, serving_data, on=self.COL_UE_ID, how="left")
        
        unique_cells = sorted(plot_df['serving_cell_id'].dropna().unique())
        if unique_cells:
            cmap = plt.get_cmap('viridis', len(unique_cells))
            for i, cell_id in enumerate(unique_cells):
                cell_ues = plot_df[plot_df['serving_cell_id'] == cell_id]
                ax.scatter(cell_ues[self.COL_LON], cell_ues[self.COL_LAT], color=cmap(i), s=10, alpha=0.8, label=f"UEs ({cell_id})")

        no_serve_ues = plot_df[plot_df['serving_cell_id'].isna()]
        ax.scatter(no_serve_ues[self.COL_LON], no_serve_ues[self.COL_LAT], c='darkorange', marker='x', s=25, label='Disconnected UEs')

        # Updated plotting logic for towers with three states
        if not active_towers.empty:
            ax.scatter(active_towers[self.COL_CELL_LON], active_towers[self.COL_CELL_LAT], marker='^', c='green', s=150, edgecolors='black', label='Fully Active Sites')
        if not inactive_towers.empty:
            ax.scatter(inactive_towers[self.COL_CELL_LON], inactive_towers[self.COL_CELL_LAT], marker='^', c='red', s=150, alpha=0.8, edgecolors='black', label='Fully Inactive Sites')
        if not partial_towers.empty:
            ax.scatter(partial_towers[self.COL_CELL_LON], partial_towers[self.COL_CELL_LAT], marker='^', c='yellow', s=150, edgecolors='black', label='Partially Active Sites')

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize='small')


    def generate_comparison_plots(self, tick: int, output_dir: str):
        """
        Generates two plots for comparison: a baseline and an energy-saving optimized scenario.
        """
        ue_data_file = self.ue_data_path_template.format(tick=tick)
        if not os.path.exists(ue_data_file):
            logger.error(f"UE data file not found for tick {tick}: {ue_data_file}")
            return
        
        ue_data_df = pd.read_csv(ue_data_file)
        if self.COL_UE_ID not in ue_data_df.columns:
            ue_data_df[self.COL_UE_ID] = range(len(ue_data_df))

        def attach_ues(predictions_df: pd.DataFrame):
            if predictions_df.empty or self.COL_UE_ID not in predictions_df.columns:
                return pd.DataFrame(columns=[self.COL_UE_ID, 'serving_cell_id'])
            idx = predictions_df.groupby(self.COL_UE_ID)[self.COL_RXPOWER_DBM].idxmax()
            serving_data = predictions_df.loc[idx].copy()
            serving_data.rename(columns={self.COL_CELL_ID: 'serving_cell_id'}, inplace=True)
            return serving_data[[self.COL_UE_ID, 'serving_cell_id']]
            
        # --- Site-level logic ---
        # Create a site_id column, assuming format like 'cell_1_0' -> 'cell_1'
        self.site_config_df['site_id'] = self.site_config_df[self.COL_CELL_ID].str.rsplit('_', n=1).str[0]
        # Get unique physical tower locations based on site_id
        site_locations = self.site_config_df.drop_duplicates(subset=['site_id']).copy()

        # --- 1. Baseline Scenario ---
        logger.info("Simulating baseline scenario (all towers on)...")
        all_cell_ids = self.site_config_df[self.COL_CELL_ID].tolist()
        baseline_preds = self._get_rf_predictions(ue_data_df, self.site_config_df, all_cell_ids)
        baseline_serving_data = attach_ues(baseline_preds)
        # In baseline, all sites are fully active
        baseline_active_towers = site_locations
        baseline_inactive_towers = pd.DataFrame()
        baseline_partial_towers = pd.DataFrame()

        # --- 2. Optimized Scenario ---
        logger.info("Simulating optimized scenario (energy saving)...")
        action_indices, _ = self.rl_model.predict(tick, deterministic=True)
        
        active_cell_ids, inactive_cell_ids = [], []
        for i, cell_action_idx in enumerate(action_indices):
            cell_id = self.site_config_df[self.COL_CELL_ID].iloc[i]
            if cell_action_idx == len(TILT_SET):
                inactive_cell_ids.append(cell_id)
            else:
                active_cell_ids.append(cell_id)

        # Determine the status of each site for the optimized plot
        fully_active_sites, fully_inactive_sites, partially_active_sites = [], [], []
        for site_id, group in self.site_config_df.groupby('site_id'):
            all_cells_in_site = set(group[self.COL_CELL_ID])
            active_cells_in_site = all_cells_in_site.intersection(active_cell_ids)
            
            if len(active_cells_in_site) == 0:
                fully_inactive_sites.append(site_id)
            elif len(active_cells_in_site) == len(all_cells_in_site):
                fully_active_sites.append(site_id)
            else:
                partially_active_sites.append(site_id)
        
        opt_active_towers = site_locations[site_locations['site_id'].isin(fully_active_sites)]
        opt_inactive_towers = site_locations[site_locations['site_id'].isin(fully_inactive_sites)]
        opt_partial_towers = site_locations[site_locations['site_id'].isin(partially_active_sites)]
        
        active_topology_for_sim = self.site_config_df[self.site_config_df[self.COL_CELL_ID].isin(active_cell_ids)]
        optimized_preds = self._get_rf_predictions(ue_data_df, active_topology_for_sim, active_cell_ids)
        optimized_serving_data = attach_ues(optimized_preds)

        # --- 3. Generate Plots ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), sharex=True, sharey=True)
        fig.suptitle(f'Energy Saving Comparison for Tick {tick}', fontsize=20)

        self._plot_scenario(ax1, 'Baseline: All Towers Active', ue_data_df, baseline_serving_data,
                            baseline_active_towers, baseline_inactive_towers, baseline_partial_towers)
        self._plot_scenario(ax2, 'Optimized: Energy Saving Enabled', ue_data_df, optimized_serving_data,
                            opt_active_towers, opt_inactive_towers, opt_partial_towers)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"energy_saving_comparison_tick_{tick}.png")
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Comparison plot saved to: {output_path}")
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize the impact of the energy saving model.")
    parser.add_argument("--tick", type=int, required=True, help="The simulation tick (0-23) to visualize.")
    parser.add_argument("--bdt-model", type=str, default="bdt_model_map.pickle", help="Path to the BDT model pickle file.")
    parser.add_argument("--rl-model", type=str, default="energy_saver_agent.zip", help="Path to the trained RL agent zip file.")
    parser.add_argument("--topology", type=str, default="topology.csv", help="Path to the topology CSV file.")
    parser.add_argument("--config", type=str, default="config.csv", help="Path to the config CSV file.")
    parser.add_argument("--ue-data-dir", type=str, default="ue_data_gym_ready", help="Directory containing preprocessed UE data.")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory to save the output plots.")
    
    args = parser.parse_args()

    visualizer = EnergySavingVisualizer(
        bdt_model_path=args.bdt_model,
        rl_model_path=args.rl_model,
        topology_path=args.topology,
        config_path=args.config,
        ue_data_path_template=os.path.join(args.ue_data_dir, "generated_ue_data_for_cco_{tick}.csv")
    )
    visualizer.generate_comparison_plots(tick=args.tick, output_dir=args.output_dir)
