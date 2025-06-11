import pandas as pd
import pickle
import os
import subprocess

from radp.client.client import RADPClient
from radp.client.helper import RADPHelper, ModelStatus
# FIX: Add imports for all classes contained within the pickled BDT object
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin, ExactGPModel, NormMethod
from radp.digital_twin.utils import constants as c

class BDTManager:
    """
    Manages the training, saving, loading, and inference of the Bayesian Digital Twin models.
    This manager handles a dictionary (map) of BDT models, one for each cell_id.
    """

    def __init__(self, topology_path, config_path, training_data_path, model_path="model.pickle"):
        """
        Initializes the BDTManager.

        Args:
            topology_path (str): Path to the topology.csv file.
            config_path (str): Path to the config.csv file.
            training_data_path (str): Path to the UE training data.
            model_path (str, optional): Path to save/load the trained model map. Defaults to "model.pickle".
        """
        self.topology_path = topology_path
        self.config_path = config_path
        self.training_data_path = training_data_path
        self.model_path = model_path
        self.model_map = None

    def train(self, model_id="bdt_energy_saving_model", container_name="radp_dev-training-1"):
        """
        Trains the BDT model using the RADP client and downloads the trained model
        from the specified Docker container.

        Args:
            model_id (str): A unique identifier for the model being trained.
            container_name (str): The name of the Docker container running the training service.
        """
        print(f"Initiating BDT model training for model_id: {model_id}...")

        try:
            # 1. Initialize RADP client and helper
            radp_client = RADPClient()
            radp_helper = RADPHelper(radp_client)

            # 2. Load data for training
            topology_df = pd.read_csv(self.topology_path)
            training_data_df = pd.read_csv(self.training_data_path)
            
            # 3. Call the train API via the client
            radp_client.train(
                model_id=model_id,
                params={},
                ue_training_data=training_data_df,
                topology=topology_df,
            )
            print(f"Training request sent for model_id: {model_id}. Waiting for completion...")

            # 4. Wait for the training to complete
            model_status: ModelStatus = radp_helper.resolve_model_status(
                model_id, wait_interval=30, max_attempts=100, verbose=True
            )

            if model_status.success:
                print(f"BDT model '{model_id}' trained successfully in the backend.")
                self._download_model_from_container(model_id, container_name)
            else:
                print(f"BDT model training failed: {model_status.error_message}")

        except Exception as e:
            print(f"An error occurred while managing BDT training: {e}")

    def _download_model_from_container(self, model_id: str, container_name: str):
        """
        Copies the trained model file from the Docker container to the local filesystem.
        """
        container_path = f"/srv/radp/models/{model_id}/model.pickle"
        local_path = self.model_path
        
        print(f"Attempting to download model from '{container_name}:{container_path}' to '{local_path}'...")

        try:
            # Construct the docker cp command
            command = ["docker", "cp", f"{container_name}:{container_path}", local_path]
            
            # Execute the command
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            print("Model downloaded successfully from Docker container.")
            if result.stdout:
                print(f"Docker stdout: {result.stdout}")

        except FileNotFoundError:
            print("Error: 'docker' command not found. Please ensure Docker is installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing 'docker cp': {e}")
            print(f"Stderr: {e.stderr}")
            print("Please ensure the container name and model path are correct and the container is running.")


    def load_model(self):
        """
        Loads the trained BDT model map from a pickle file.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please train the model first.")

        self.model_map = BayesianDigitalTwin.load_model_map_from_pickle(self.model_path)
        if not self.model_map:
             raise ValueError("Failed to load a valid model map. The file might be corrupted or empty.")
        print(f"BDT model map for {len(self.model_map)} cells loaded successfully.")
        return self.model_map

    def get_model_map(self):
        """
        Returns the loaded model map.
        """
        if self.model_map is None:
            self.load_model()
        return self.model_map
