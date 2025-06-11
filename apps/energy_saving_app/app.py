import argparse
import os
import sys

# FIX: Correct the Python path to include the project root.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
APPS_DIR = os.path.dirname(APP_DIR)
PROJECT_ROOT = os.path.dirname(APPS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data_preprocessor import UEDataPreprocessor
from bdt_manager import BDTManager
from rl_trainer import run_rl_training
from rl_predictor import run_rl_prediction
# Import the new visualizer
from energy_saving_visualizer import EnergySavingVisualizer

def main():
    """
    Main function to run the energy saving application pipeline.
    Handles data preprocessing, BDT training, RL training, inference, and visualization.
    """
    parser = argparse.ArgumentParser(description="Energy Saving Application using BDT and RL.")
    parser.add_argument("--preprocess-data", action="store_true", help="Run UE data preprocessing.")
    parser.add_argument("--train-bdt", action="store_true", help="Train the Bayesian Digital Twin model map.")
    parser.add_argument("--train-rl", action="store_true", help="Train the reinforcement learning model.")
    parser.add_argument("--infer", action="store_true", help="Run inference with the trained RL model.")
    parser.add_argument("--visualize", action="store_true", help="Generate comparison plots for a given tick.")
    parser.add_argument("--tick", type=int, default=None, help="The specific tick (0-23) for inference or visualization.")
    
    args = parser.parse_args()

    # --- Configuration ---
    RAW_UE_DATA_DIR = os.path.join(APP_DIR, "ue_data_per_tick")
    GYM_UE_DATA_DIR = os.path.join(APP_DIR, "ue_data_gym_ready")
    TOPOLOGY_PATH = os.path.join(APP_DIR, "topology.csv")
    CONFIG_PATH = os.path.join(APP_DIR, "config.csv")
    BDT_TRAINING_DATA_PATH = os.path.join(APP_DIR, "dummy_ue_training_data.csv")
    BDT_MODEL_PATH = os.path.join(APP_DIR, "bdt_model_map.pickle")
    RL_MODEL_PATH = os.path.join(APP_DIR, "energy_saver_agent.zip")
    RL_LOG_DIR = os.path.join(APP_DIR, "rl_training_logs")
    PLOT_OUTPUT_DIR = os.path.join(APP_DIR, "plots")

    # --- Pipeline Execution ---
    
    if args.preprocess_data:
        print("--- Running UE Data Preprocessing Step ---")
        preprocessor = UEDataPreprocessor(input_dir=RAW_UE_DATA_DIR, output_dir=GYM_UE_DATA_DIR)
        preprocessor.run()

    if args.train_bdt:
        print("--- Running BDT Training Step ---")
        bdt_manager = BDTManager(
            topology_path=TOPOLOGY_PATH,
            training_data_path=BDT_TRAINING_DATA_PATH,
            model_path=BDT_MODEL_PATH
        )
        bdt_manager.train()

    if args.train_rl:
        print("--- Running RL Training Step ---")
        run_rl_training(
            bdt_model_path=BDT_MODEL_PATH, ue_data_dir=GYM_UE_DATA_DIR,
            topology_path=TOPOLOGY_PATH, config_path=CONFIG_PATH,
            rl_model_path=RL_MODEL_PATH, log_dir=RL_LOG_DIR
        )

    if args.infer:
        if args.tick is None:
            parser.error("--tick is required for inference.")
        else:
            print(f"--- Running Inference Step for Tick {args.tick} ---")
            run_rl_prediction(
                model_load_path=RL_MODEL_PATH, topology_path=TOPOLOGY_PATH,
                target_tick=args.tick
            )
            
    if args.visualize:
        if args.tick is None:
            parser.error("--tick is required for visualization.")
        else:
            print(f"--- Running Visualization Step for Tick {args.tick} ---")
            visualizer = EnergySavingVisualizer(
                bdt_model_path=BDT_MODEL_PATH,
                rl_model_path=RL_MODEL_PATH,
                topology_path=TOPOLOGY_PATH,
                config_path=CONFIG_PATH,  # FIX: Pass the config path
                ue_data_path_template=os.path.join(GYM_UE_DATA_DIR, "generated_ue_data_for_cco_{tick}.csv")
            )
            visualizer.generate_comparison_plots(tick=args.tick, output_dir=PLOT_OUTPUT_DIR)

    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()
