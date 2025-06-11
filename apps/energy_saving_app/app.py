import argparse
import os
import sys

# Add the app's root directory to the python path to allow for cleaner imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessor import UEDataPreprocessor
from bdt_manager import BDTManager
from rl_manager import RLManager

def main():
    """
    Main function to run the energy saving application pipeline.
    Handles data preprocessing, BDT training, RL training, and inference.
    """
    parser = argparse.ArgumentParser(description="Energy Saving Application using BDT and RL.")
    parser.add_argument("--preprocess-data", action="store_true", help="Run UE data preprocessing.")
    parser.add_argument("--train-bdt", action="store_true", help="Train the Bayesian Digital Twin model map.")
    parser.add_argument("--train-rl", action="store_true", help="Train the reinforcement learning model.")
    parser.add_argument("--infer", action="store_true", help="Run inference with the trained RL model.")
    parser.add_argument("--tick", type=int, default=None, help="The specific tick (0-23) to run inference for.")
    
    args = parser.parse_args()

    # --- Configuration ---
    # It's better to define paths relative to the app.py script location
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_UE_DATA_DIR = os.path.join(APP_DIR, "ue_data_per_tick") # Input for preprocessor
    GYM_UE_DATA_DIR = os.path.join(APP_DIR, "ue_data_gym_ready") # Output of preprocessor
    
    TOPOLOGY_PATH = os.path.join(APP_DIR, "topology.csv")
    CONFIG_PATH = os.path.join(APP_DIR, "config.csv")
    BDT_TRAINING_DATA_PATH = os.path.join(APP_DIR, "dummy_ue_training_data.csv")
    
    BDT_MODEL_PATH = os.path.join(APP_DIR, "bdt_model_map.pickle")
    RL_MODEL_PATH = os.path.join(APP_DIR, "energy_saver_agent.zip")
    RL_LOG_DIR = os.path.join(APP_DIR, "rl_training_logs")

    # --- Pipeline Execution ---
    
    if args.preprocess_data:
        print("--- Running UE Data Preprocessing Step ---")
        preprocessor = UEDataPreprocessor(input_dir=RAW_UE_DATA_DIR, output_dir=GYM_UE_DATA_DIR)
        preprocessor.run()
        print("--- Preprocessing Complete ---")

    if args.train_bdt:
        print("--- Running BDT Training Step ---")
        bdt_manager = BDTManager(
            topology_path=TOPOLOGY_PATH,
            config_path=CONFIG_PATH,
            training_data_path=BDT_TRAINING_DATA_PATH,
            model_path=BDT_MODEL_PATH
        )
        bdt_manager.train()
        print("--- BDT Training Complete ---")

    if args.train_rl:
        print("--- Running RL Training Step ---")
        bdt_manager = BDTManager(
            topology_path=TOPOLOGY_PATH,
            config_path=CONFIG_PATH,
            training_data_path=BDT_TRAINING_DATA_PATH,
            model_path=BDT_MODEL_PATH
        )
        # We need the BDT model loaded to train the RL agent
        if not os.path.exists(BDT_MODEL_PATH):
            print(f"Error: BDT model not found at {BDT_MODEL_PATH}. Please run --train-bdt first.")
        else:
            rl_manager = RLManager(
                bdt_manager=bdt_manager,
                ue_data_dir=GYM_UE_DATA_DIR,
                topology_path=TOPOLOGY_PATH,
                config_path=CONFIG_PATH,
                rl_model_path=RL_MODEL_PATH
            )
            rl_manager.train(log_dir=RL_LOG_DIR)
            print("--- RL Training Complete ---")

    if args.infer:
        if args.tick is None:
            print("Error: --tick argument is required for inference. Please specify a value between 0 and 23.")
        else:
            print(f"--- Running Inference Step for Tick {args.tick} ---")
            if not all([os.path.exists(BDT_MODEL_PATH), os.path.exists(RL_MODEL_PATH)]):
                print(f"Error: Model files not found. Ensure BDT model exists at {BDT_MODEL_PATH} and RL model at {RL_MODEL_PATH}.")
                print("Please run --train-bdt and --train-rl first.")
            else:
                 bdt_manager = BDTManager(
                    topology_path=TOPOLOGY_PATH,
                    config_path=CONFIG_PATH,
                    training_data_path=BDT_TRAINING_DATA_PATH,
                    model_path=BDT_MODEL_PATH
                )
                 rl_manager = RLManager(
                    bdt_manager=bdt_manager,
                    ue_data_dir=GYM_UE_DATA_DIR,
                    topology_path=TOPOLOGY_PATH,
                    config_path=CONFIG_PATH,
                    rl_model_path=RL_MODEL_PATH
                )
                 rl_manager.predict(target_tick=args.tick)
                 print("--- Inference Complete ---")

    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()

