import pandas as pd
import os
import subprocess

from radp.client.client import RADPClient
from radp.client.helper import RADPHelper, ModelStatus
from radp.digital_twin.utils import constants as c

class BDTManager:
    """
    Manages the training of the Bayesian Digital Twin model by orchestrating
    the backend training service.
    """

    def __init__(self, topology_path, training_data_path, model_path="model.pickle"):
        """
        Initializes the BDTManager.

        Args:
            topology_path (str): Path to the topology.csv file.
            training_data_path (str): Path to the UE training data.
            model_path (str, optional): Path where the trained model will be saved. Defaults to "model.pickle".
        """
        self.topology_path = topology_path
        self.training_data_path = training_data_path
        self.model_path = model_path

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
            radp_client = RADPClient()
            radp_helper = RADPHelper(radp_client)

            topology_df = pd.read_csv(self.topology_path)
            training_data_df = pd.read_csv(self.training_data_path)
            
            radp_client.train(
                model_id=model_id,
                params={},
                ue_training_data=training_data_df,
                topology=topology_df,
            )
            print(f"Training request sent for model_id: {model_id}. Waiting for completion...")

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
            command = ["docker", "cp", f"{container_name}:{container_path}", local_path]
            subprocess.run(command, check=True, capture_output=True, text=True)
            print("Model downloaded successfully from Docker container.")
        except FileNotFoundError:
            print("Error: 'docker' command not found. Please ensure Docker is installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing 'docker cp': {e}\nStderr: {e.stderr}")
            print("Please ensure the container name and model path are correct and the container is running.")

