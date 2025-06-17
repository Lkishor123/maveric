# Energy Saving Application using Reinforcement Learning

This application uses a combination of a Bayesian Digital Twin (BDT) and a Reinforcement Learning (RL) agent to determine the optimal on/off state and tilt configuration for cellular towers to save energy while maintaining network performance.

The entire workflow is orchestrated by `app.py`, which provides command-line flags to run specific stages of the pipeline, from data preparation to model training, inference, and visualization.

## Directory Structure

For the application to run correctly, the following files and directories must be present within the `energy_saving_app` folder:

```bash
energy_saving_app/
│
├── app.py                      # Main orchestrator script
├── bdt_manager.py              # Manages BDT model training
├── data_preprocessor.py        # Prepares UE data for the Gym environment
├── rl_trainer.py               # Contains the RL training logic and Gym environment
├── rl_predictor.py             # Handles inference using the trained RL agent
├── energy_saving_visualizer.py # Generates comparison plots
│
├── topology.csv                # Describes the physical layout of cell towers
├── config.csv                  # Initial configuration for cell towers (e.g., tilts)
├── dummy_ue_training_data.csv  # Data for training the BDT model
│
├── ue_data_per_tick/           # DIRECTORY containing raw UE location data per hour
│   ├── generated_ue_data_for_cco_0.csv
│   └── ... (up to 23)
│
└── (Generated Outputs)/
├── ue_data_gym_ready/      # Processed UE data, ready for the Gym
├── bdt_model_map.pickle    # The trained Bayesian Digital Twin model
├── energy_saver_agent.zip  # The trained RL agent
├── rl_training_logs/       # Logs and checkpoints from RL training
└── plots/                    # Output directory for visualization plots

```
## Prerequisites

Before running the application, ensure you have the following installed and configured:

1.  **Python 3.8+**
2.  **Docker:** The BDT model training is executed inside a Docker container. Make sure the Docker daemon is running.
3.  **RADP Environment:** The `radp` library and its dependencies must be installed and accessible via your `PYTHONPATH`. The `app.py` script attempts to handle this, but a global setup is recommended.
4.  **Required Python Packages:** Install all necessary packages. You may need to create a `requirements.txt` file that includes `pandas`, `numpy`, `gymnasium`, `stable-baselines3[extra]`, `matplotlib`, `torch`, and `gpytorch`.

## Application Workflow & Usage

The application is designed to be run as a pipeline. Each step is triggered by a specific flag passed to `app.py`.

---

### **Step 1: Preprocess UE Data**

This initial step prepares the raw, per-hour UE location data for the simulation environment.

-   **Command:**
    ```bash
    python app.py --preprocess-data
    ```
-   **Required Inputs:**
    -   A directory named `ue_data_per_tick/` containing `generated_ue_data_for_cco_{tick}.csv` files.
-   **Output:**
    -   Creates a new directory named `ue_data_gym_ready/` containing the processed UE data.

---

### **Step 2: Train the Bayesian Digital Twin (BDT)**

This step trains the underlying RF simulation model using a backend service running in Docker.

-   **Prerequisites:**
    -   The `radp_dev-training-1` Docker container must be running.
    -   The user must have permissions to execute `docker` commands.
-   **Command:**
    ```bash
    python app.py --train-bdt
    ```
-   **Required Inputs:**
    -   `topology.csv`
    -   `dummy_ue_training_data.csv`
-   **Output:**
    -   `bdt_model_map.pickle`: The trained BDT model file, downloaded from the Docker container.

---

### **Step 3: Train the RL Energy Saving Agent**

This step trains the PPO agent to learn the energy-saving policy.

-   **Command:**
    ```bash
    python app.py --train-rl
    ```
-   **Required Inputs:**
    -   `bdt_model_map.pickle` (from Step 2).
    -   `ue_data_gym_ready/` directory (from Step 1).
    -   `topology.csv`
    -   `config.csv`
-   **Outputs:**
    -   `energy_saver_agent.zip`: The saved, trained RL agent.
    -   `rl_training_logs/`: A directory with TensorBoard logs and model checkpoints.

---

### **Step 4: Run Inference**

Uses the trained agent to predict the optimal network configuration for a specific hour.

-   **Command:**
    ```bash
    # Replace <T> with the desired hour (0-23)
    python app.py --infer --tick <T>
    ```
-   **Required Inputs:**
    -   `energy_saver_agent.zip` (from Step 3).
    -   `topology.csv`
-   **Output:**
    -   Prints a table to the console showing the predicted optimal state (`ON`/`OFF`) and tilt for each cell tower.

---

### **Step 5: Visualize the Results**

This step generates a side-by-side plot comparing the network state before and after the energy-saving optimization for a specific hour.

-   **Command:**
    ```bash
    # Replace <T> with the desired hour (0-23)
    python app.py --visualize --tick <T>
    ```
-   **Required Inputs:**
    -   `energy_saver_agent.zip` (from Step 3).
    -   `bdt_model_map.pickle` (from Step 2).
    -   `topology.csv`
    -   The `ue_data_gym_ready/` directory (from Step 1).
-   **Output:**
    -   A `.png` image file saved to the `plots/` directory (e.g., `energy_saving_comparison_tick_8.png`). This image shows two subplots: the baseline scenario with all towers active, and the optimized scenario with some towers turned off. It visualizes which UEs remain connected, which are disconnected, and the status of each tower.

### Full Pipeline Example

To run the entire workflow from data preparation to final visualization, execute the following commands in sequence:

```bash
# 1. Prepare the UE data for the Gym
python app.py --preprocess-data

# 2. Train the core RF simulation model
python app.py --train-bdt

# 3. Train the RL decision-making agent
python app.py --train-rl

# 4. Predict the optimal configuration for 3 AM
python app.py --infer --tick 3

# 5. Visualize the impact of the optimization for 3 AM
python app.py --visualize --tick 3
