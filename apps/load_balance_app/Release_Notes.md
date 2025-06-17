# Release Notes: Load Balancing CCO Application

**Version:** 1.0
**Date:** June 18, 2025

## 1. Overview

The Load Balancing CCO Application is a modular pipeline designed to train, evaluate, and utilize a Reinforcement Learning (RL) agent for dynamic Coverage and Capacity Optimization (CCO).

The primary goal is to create an intelligent agent that suggests optimal cell antenna tilt configurations for each hour of the day. The agent's policy is trained on multiple days of simulated UE traffic data to learn time-of-day patterns. Its objective is to find a balance between three key network KPIs:

1.  **Coverage:** Ensuring UEs have adequate signal strength.
2.  **Load Balance:** Distributing network traffic evenly across active cells to prevent congestion.
3.  **Quality of Service (QoS):** Maintaining a high-quality signal (good SINR) for connected users.

A key architectural feature is the use of a pre-trained Bayesian Digital Twin (BDT) model for *local* RF simulation within the RL training loop. This decouples the computationally intensive RL training process from the live backend simulation service, enabling much faster iteration and learning.

## 2. Key Features

* **Modular Pipeline:** Each major step—data preprocessing, BDT model training, RL agent training, inference, and visualization—is handled by a separate, dedicated module orchestrated by a single main application.
* **Time-Aware RL Agent:** The RL agent learns a policy that is dependent on the hour of the day (tick 0-23), allowing it to adapt cell configurations to match daily traffic patterns.
* **Multi-Day Training:** The RL training process is designed to use UE traffic data from multiple days, making the learned policy more robust and generalizable.
* **Local BDT-based Simulation:** The RL environment loads a pre-trained BDT model and uses it to run RF simulations locally. This provides rapid feedback for the RL agent without the latency of continuous backend API calls.
* **Multi-Objective Reward Function:** The RL agent is trained to optimize a complex reward function that balances Coverage, Load, and QoS.
* **Command-Line Interface:** The entire pipeline is managed through a clear command-line interface in `main_app.py`.
* **Comparative Visualization:** Includes a dedicated module to generate side-by-side plots comparing network performance under a baseline configuration versus the RL-optimized configuration.

---

## 3. Application Pipeline & Module Breakdown

The application is designed to be run as a sequence of steps, orchestrated by `main_app.py`.

### **Step 1: Preprocess UE Data**

* **Module:** `data_preprocessor.py`
* **Function:** Prepares the raw, multi-day UE location data for use in the RL environment.
* **Input:** Reads per-tick CSV files (e.g., `generated_ue_data_for_cco_0.csv`) from directories like `generated_data/Day_*/ue_data_per_tick/`. These files are expected to have `lon` and `lat` columns.
* **Logic:** Iterates through all specified day directories, reads each CSV, renames the `lon` column to `loc_x` and `lat` to `loc_y` (as expected by the BDT's prediction frame generator), and saves the result.
* **Output:** Creates a new set of directories and files (e.g., `generated_data/Day_*/ue_data_gym_ready/`) containing the preprocessed data, ready for the Gym environment.

### **Step 2: Train Bayesian Digital Twin (BDT) Model**

* **Module:** `bdt_manager.py`
* **Function:** Manages the one-time, backend-intensive training of the core RF model.
* **Input:**
    * `topology.csv`: The full network topology.
    * `dummy_ue_training_data.csv`: **This must be a realistic, high-quality dataset** that maps cell configurations (including different tilts) and UE locations to measured RSRP values. The "dummy" name refers to its use in the pipeline test; the data content must be real for a useful model.
* **Logic:**
    1.  Uses `radp_client` to send the topology and training data to the backend `training` service.
    2.  Waits for the backend to train the BDT model and save it. The backend saves the model as a pickled Python dictionary (`Dict[str, BayesianDigitalTwin]`).
    3.  After successful training, it uses a `docker cp` command to download the saved `model.pickle` file from inside the specified Docker container to a local path (e.g., `./bdt_model_map.pickle`).
* **Output:** A `bdt_model_map.pickle` file, stored locally, containing the trained BDT models for all cells.

### **Step 3: Train the RL Agent**

* **Module:** `rl_trainer.py`
* **Function:** The core of the machine learning process. It orchestrates the training of the PPO agent.
* **Input:**
    * The `bdt_model_map.pickle` file generated in Step 2.
    * The preprocessed per-tick UE data from the `ue_data_gym_ready` directories for the specified training days.
    * `topology.csv` and `config.csv` (for initial state).
* **Logic:**
    1.  Loads the BDT model map from the pickle file.
    2.  Loads all per-tick UE DataFrames from the specified training day directories into a dictionary `{tick: [df_day0, df_day1, ...]}`.
    3.  Initializes the custom RL environment, `CCO_RL_Env`, passing it the loaded BDT models and the pool of UE data.
    4.  Initializes a Stable Baselines3 PPO agent.
    5.  Calls `model.learn()`. The PPO agent then interacts with the `CCO_RL_Env` for thousands of steps:
        * The agent observes the current `tick`.
        * It outputs an `action` (a set of tilts).
        * The environment's `step` method applies the action, **samples a random day's UE data for the current tick**, and runs a **local RF simulation** using the loaded BDT models.
        * A multi-objective reward is calculated and returned to the agent.
    6.  The agent updates its policy based on the rewards received.
* **Output:** A trained RL agent saved as a `.zip` file (e.g., `./cco_rl_agent_multiday.zip`).

### **Step 4: Inference with the RL Agent**

* **Module:** `rl_predictor.py`
* **Function:** Uses the trained RL agent to get an immediate recommendation for a specific hour.
* **Input:**
    * The path to the saved RL agent (`.zip` file).
    * The `topology.csv` file (to get the correct cell order).
    * A target `tick` (0-23) from the command line.
* **Logic:**
    1.  Loads the trained PPO agent.
    2.  Calls `model.predict(target_tick)` to get the deterministic best action for that hour.
    3.  Maps the numerical action array to a human-readable DataFrame showing each `cell_id` and its predicted optimal `cell_el_deg`.
* **Output:** Prints the recommended configuration to the console.

### Step 5: Visualize Performance

* **Module:** `cco_visualizer.py`
* **Function:** Provides a qualitative assessment of the RL agent's performance by comparing it against a baseline.
* **Input:**
    * The BDT model pickle file.
    * The trained RL agent zip file.
    * `topology.csv` and `config.csv`.
    * A specific `test-day` and `tick` to evaluate.
* **Logic:**
    1.  Loads all necessary models and data.
    2.  **Scenario A (Baseline):** Simulates RF performance for the specified tick using the *initial* configuration from `config.csv`.
    3.  **Scenario B (Optimized):** Uses the RL agent to predict the best configuration for that tick, then simulates RF performance using *that* configuration.
    4.  **Plotting:** Generates a side-by-side plot showing the UE coverage map for both scenarios, allowing for a visual comparison of the results. The UEs are colored by their RSRP level.
* **Output:** A `.png` comparison image saved to the `./plots/` directory.

---

## 5. How to Use

The pipeline is run from the command line using `main_app.py`.

**Example Full Workflow:**

```bash
# Set your Maveric project root
export MAVERIC_ROOT=/path/to/your/maveric/project

# Step 1: Preprocess UE data for training (days 0-3) and testing (day 4)
python main_app.py --preprocess-data --train-days 0 1 2 3 --test-day 4

# Step 2: Train the core BDT model and download it from the 'radp_dev-training-1' container
python main_app.py --train-bdt --bdt-model-id "bdt_load_balance_v1" --container "radp_dev-training-1"

# Step 3: Train the RL agent on the first 4 days of data for 120,000 steps
python main_app.py --train-rl --train-days 0 1 2 3 --total-timesteps 120000

# Step 4: Get a specific recommendation for a peak hour (e.g., tick 18)
python main_app.py --infer --tick 18

# Step 5: Visualize how the agent's recommendation performs on the test data for that hour
python main_app.py --visualize --test-day 4 --tick 18
