## Prerequisites

Before running the application, ensure you have the following installed and configured:

1.  **Python 3.8+**
2.  **Docker:** The BDT model training is executed inside a Docker container. Make sure the Docker daemon is running.
3.  **RADP Environment:** The `radp` library and its dependencies must be installed.
4.  **Required Python Packages:** Install all necessary packages by running:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to create a `requirements.txt` file for the `energy_saving_app` that includes `pandas`, `numpy`, `gymnasium`, `stable-baselines3`, etc.)*

## Application Workflow & Usage

The application is designed to be run as a pipeline. Each step is triggered by a specific flag passed to `app.py`.

---

### **Step 1: Preprocess UE Data**

This initial step prepares the raw, per-hour UE location data for the simulation environment. It finds all `generated_ue_data_for_cco_*.csv` files, standardizes the column names for GPS coordinates (`lon`/`lat` to `loc_x`/`loc_y`), and saves the result.

-   **Command:**
    ```bash
    python app.py --preprocess-data
    ```

-   **Required Input Files:**
    -   A directory named `ue_data_per_tick/` must exist in the same folder as `app.py`.
    -   This directory must contain UE data files, typically named `generated_ue_data_for_cco_0.csv`, `generated_ue_data_for_cco_1.csv`, etc.

-   **Output:**
    -   Creates a new directory named `ue_data_gym_ready/`.
    -   This output directory will contain the processed UE data files, ready for use by the Gym environment.

---

### **Step 2: Train the Bayesian Digital Twin (BDT)**

This crucial step trains the underlying RF simulation model. It uses the `RADPClient` to send the topology and training data to a backend training service running in a Docker container. After the backend finishes training, this script copies the resulting model file from the container to your local directory.

-   **Prerequisites:**
    -   The `radp_dev-training-1` Docker container must be running.
    -   The user running the script must have permissions to execute `docker` commands.

-   **Command:**
    ```bash
    python app.py --train-bdt
    ```

-   **Required Input Files:**
    -   `topology.csv`: Defines the locations and IDs of all cell towers.
    -   `dummy_ue_training_data.csv`: Provides the RF measurements used to train the BDT model.

-   **Output:**
    -   `bdt_model_map.pickle`: The trained BDT model file, which is downloaded from the Docker container upon successful training.

---

### **Step 3: Train the RL Energy Saving Agent**

With the BDT model and UE data ready, this step trains the reinforcement learning agent. The agent interacts with the custom `TickAwareEnergyEnv` Gym environment, learning a policy to turn cells on/off or adjust their tilts to maximize a reward signal based on energy savings and network quality.

-   **Command:**
    ```bash
    python app.py --train-rl
    ```

-   **Required Input Files:**
    -   `bdt_model_map.pickle` (the output from Step 2).
    -   The `ue_data_gym_ready/` directory (the output from Step 1).
    -   `topology.csv`
    -   `config.csv`: Contains the initial/default configuration for cell parameters like tilt.

-   **Outputs:**
    -   `energy_saver_agent.zip`: The saved file containing the trained PPO agent from `stable-baselines3`.
    -   `rl_training_logs/`: A directory containing TensorBoard logs and intermediate model checkpoints, which can be used to monitor training progress.

---

### **Step 4: Run Inference to Get Optimal Configuration**

This is the final step, where the trained RL agent is used to predict the best network configuration for a specific time of day (tick).

-   **Command:**
    ```bash
    # Replace <T> with the desired hour (0-23)
    python app.py --infer --tick <T>
    ```
    **Example (for 10 AM):**
    ```bash
    python app.py --infer --tick 10
    ```

-   **Required Input Files:**
    -   `energy_saver_agent.zip` (the trained agent from Step 3).
    -   `bdt_model_map.pickle` (the BDT model from Step 2).
    -   `topology.csv`
    -   `config.csv`
    -   The `ue_data_gym_ready/` directory.

-   **Output:**
    -   The script will print a table to the console showing the predicted optimal state (`ON`/`OFF`) and electrical tilt (`cell_el_deg`) for each cell tower for the specified tick.

### Full Pipeline Example

To run the entire workflow from data preparation to final prediction, execute the following commands in sequence:

```bash
# 1. Prepare the UE data for the Gym
python app.py --preprocess-data

# 2. Train the core RF simulation model
python app.py --train-bdt

# 3. Train the RL decision-making agent
python app.py --train-rl

# 4. Predict the optimal configuration for 2 AM
python app.py --infer --tick 2
