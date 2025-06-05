---
(venv) lk@lk-IdeaPad-5-Pro-14ACN6:~/Projects/accelcq-repos/cloudly/github/maveric/apps/energy_saving_demo$ python rl_energy_saving_trainer.py
2025-06-04 02:39:32.361327: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1748984972.387319  812986 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1748984972.395703  812986 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1748984972.418606  812986 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1748984972.418640  812986 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1748984972.418647  812986 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1748984972.418653  812986 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-06-04 02:39:32.424907: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-06-04 02:39:36,687 - __main__ - INFO - --- Starting RL Energy Saver Training Script ---
2025-06-04 02:39:37,077 - __main__ - INFO - Loaded BDT map for 15 cells for RL Gym.
2025-06-04 02:39:37,143 - __main__ - INFO - Creating TickAwareEnergyEnv for RL training...
2025-06-04 02:39:37,144 - __main__ - INFO - TickAwareEnergyEnv initialized for 15 cells.
2025-06-04 02:39:37,144 - __main__ - INFO - TickAwareEnergyEnv for RL training created successfully.
2025-06-04 02:39:37,145 - __main__ - INFO - Defining PPO agent...
Using cpu device
Wrapping the env in a DummyVecEnv.
2025-06-04 02:39:38,613 - __main__ - INFO - Starting RL agent training for 24000 timesteps...
Logging to ./rl_training_logs/PPOTickAwareEnergyRun_3






----------------------------------
| rollout/            |          |
|    ep_len_mean      | 24       |
|    ep_rew_mean      | -1.79e+03 |
| time/               |          |
|    fps              | 2        |
|    iterations       | 1        |
|    time_elapsed     | 956      |
|    total_timesteps  | 2048     |
----------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 24           |
|    ep_rew_mean          | -1.8e+03     |
| time/                   |              |
|    fps                  | 2            |
|    iterations           | 2            |
|    time_elapsed         | 1935         |
|    total_timesteps      | 4096         |
| train/                  |              |
|    approx_kl            | 0.0017705231 |
|    clip_fraction        | 0.000146     |
|    clip_range           | 0.2          |
|    entropy_loss         | -37.3        |
|    explained_variance   | 3.49e-05     |
|    learning_rate        | 0.0003       |
|    loss                 | 2.4e+05      |
|    n_updates            | 10           |
|    policy_gradient_loss | -0.0269      |
|    value_loss           | 4.47e+05     |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 24            |
|    ep_rew_mean          | -1.8e+03      |
| time/                   |               |
|    fps                  | 2             |
|    iterations           | 3             |
|    time_elapsed         | 2918          |
|    total_timesteps      | 6144          |
| train/                  |               |
|    approx_kl            | 0.00045126455 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -37.3         |
|    explained_variance   | -4.59e-05     |
|    learning_rate        | 0.0003        |
|    loss                 | 2.12e+05      |
|    n_updates            | 20            |
|    policy_gradient_loss | -0.0106       |
|    value_loss           | 4.41e+05      |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 24            |
|    ep_rew_mean          | -1.77e+03     |
| time/                   |               |
|    fps                  | 2             |
|    iterations           | 4             |
|    time_elapsed         | 3896          |
|    total_timesteps      | 8192          |
| train/                  |               |
|    approx_kl            | 0.00047511794 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -37.3         |
|    explained_variance   | -6.32e-06     |
|    learning_rate        | 0.0003        |
|    loss                 | 2.17e+05      |
|    n_updates            | 30            |
|    policy_gradient_loss | -0.0107       |
|    value_loss           | 4.42e+05      |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 24            |
|    ep_rew_mean          | -1.78e+03     |
| time/                   |               |
|    fps                  | 2             |
|    iterations           | 5             |
|    time_elapsed         | 4870          |
|    total_timesteps      | 10240         |
| train/                  |               |
|    approx_kl            | 0.00050965545 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -37.3         |
|    explained_variance   | -5.01e-06     |
|    learning_rate        | 0.0003        |
|    loss                 | 2.22e+05      |
|    n_updates            | 40            |
|    policy_gradient_loss | -0.0113       |
|    value_loss           | 4.23e+05      |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 24           |
|    ep_rew_mean          | -1.78e+03    |
| time/                   |              |
|    fps                  | 2            |
|    iterations           | 6            |
|    time_elapsed         | 5841         |
|    total_timesteps      | 12288        |
| train/                  |              |
|    approx_kl            | 0.0005565325 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -37.3        |
|    explained_variance   | 2.5e-06      |
|    learning_rate        | 0.0003       |
|    loss                 | 2.14e+05     |
|    n_updates            | 50           |
|    policy_gradient_loss | -0.0115      |
|    value_loss           | 4.18e+05     |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 24            |
|    ep_rew_mean          | -1.76e+03     |
| time/                   |               |
|    fps                  | 2             |
|    iterations           | 7             |
|    time_elapsed         | 6809          |
|    total_timesteps      | 14336         |
| train/                  |               |
|    approx_kl            | 0.00060240173 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -37.3         |
|    explained_variance   | 1.01e-06      |
|    learning_rate        | 0.0003        |
|    loss                 | 2e+05         |
|    n_updates            | 60            |
|    policy_gradient_loss | -0.0119       |
|    value_loss           | 4.2e+05       |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 24           |
|    ep_rew_mean          | -1.79e+03    |
| time/                   |              |
|    fps                  | 2            |
|    iterations           | 8            |
|    time_elapsed         | 7779         |
|    total_timesteps      | 16384        |
| train/                  |              |
|    approx_kl            | 0.0005929371 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -37.3        |
|    explained_variance   | -2.38e-07    |
|    learning_rate        | 0.0003       |
|    loss                 | 2.02e+05     |
|    n_updates            | 70           |
|    policy_gradient_loss | -0.0117      |
|    value_loss           | 4.03e+05     |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 24            |
|    ep_rew_mean          | -1.76e+03     |
| time/                   |               |
|    fps                  | 2             |
|    iterations           | 9             |
|    time_elapsed         | 8744          |
|    total_timesteps      | 18432         |
| train/                  |               |
|    approx_kl            | 0.00056272186 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -37.3         |
|    explained_variance   | 1.19e-07      |
|    learning_rate        | 0.0003        |
|    loss                 | 2.16e+05      |
|    n_updates            | 80            |
|    policy_gradient_loss | -0.0114       |
|    value_loss           | 4.11e+05      |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 24           |
|    ep_rew_mean          | -1.78e+03    |
| time/                   |              |
|    fps                  | 2            |
|    iterations           | 10           |
|    time_elapsed         | 9711         |
|    total_timesteps      | 20480        |
| train/                  |              |
|    approx_kl            | 0.0006833583 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -37.3        |
|    explained_variance   | -2.38e-07    |
|    learning_rate        | 0.0003       |
|    loss                 | 2.07e+05     |
|    n_updates            | 90           |
|    policy_gradient_loss | -0.0126      |
|    value_loss           | 3.96e+05     |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 24            |
|    ep_rew_mean          | -1.77e+03     |
| time/                   |               |
|    fps                  | 2             |
|    iterations           | 11            |
|    time_elapsed         | 10677         |
|    total_timesteps      | 22528         |
| train/                  |               |
|    approx_kl            | 0.00068307295 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -37.3         |
|    explained_variance   | -3.58e-07     |
|    learning_rate        | 0.0003        |
|    loss                 | 1.76e+05      |
|    n_updates            | 100           |
|    policy_gradient_loss | -0.0125       |
|    value_loss           | 4.01e+05      |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 24            |
|    ep_rew_mean          | -1.75e+03     |
| time/                   |               |
|    fps                  | 2             |
|    iterations           | 12            |
|    time_elapsed         | 11640         |
|    total_timesteps      | 24576         |
| train/                  |               |
|    approx_kl            | 0.00075237296 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -37.3         |
|    explained_variance   | 1.19e-07      |
|    learning_rate        | 0.0003        |
|    loss                 | 1.64e+05      |
|    n_updates            | 110           |
|    policy_gradient_loss | -0.0127       |
|    value_loss           | 3.89e+05      |
-------------------------------------------
2025-06-04 05:53:40,620 - __main__ - INFO - RL Training finished.
2025-06-04 05:53:40,626 - __main__ - INFO - Trained RL model saved to ./rl_energy_saver_agent_ppo.zip
2025-06-04 05:53:40,626 - __main__ - INFO - Closing TickAwareEnergyEnv.
2025-06-04 05:53:40,626 - __main__ - INFO - --- RL Energy Saver Training Script Finished ---
(venv) lk@lk-IdeaPad-5-Pro-14ACN6:~/Projects/accelcq-repos/cloudly/github/maveric/apps/energy_saving_demo$
(venv) lk@lk-IdeaPad-5-Pro-14ACN6:~/Projects/accelcq-repos/cloudly/github/maveric/apps/energy_saving_demo$
(venv) lk@lk-IdeaPad-5-Pro-14ACN6:~/Projects/accelcq-repos/cloudly/github/maveric/apps/energy_saving_demo$
(venv) lk@lk-IdeaPad-5-Pro-14ACN6:~/Projects/accelcq-repos/cloudly/github/maveric/apps/energy_saving_demo$
(venv) lk@lk-IdeaPad-5-Pro-14ACN6:~/Projects/accelcq-repos/cloudly/github/maveric/apps/energy_saving_demo$
(venv) lk@lk-IdeaPad-5-Pro-14ACN6:~/Projects/accelcq-repos/cloudly/github/maveric/apps/energy_saving_demo$
(venv) lk@lk-IdeaPad-5-Pro-14ACN6:~/Projects/accelcq-repos/cloudly/github/maveric/apps/energy_saving_demo$ python rl_energy_saving_
rl_energy_saving_predictor.py  rl_energy_saving_trainer.py
(venv) lk@lk-IdeaPad-5-Pro-14ACN6:~/Projects/accelcq-repos/cloudly/github/maveric/apps/energy_saving_demo$ python rl_energy_saving_predictor.py --5
2025-06-04 07:29:39.977482: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1749002379.991109 1073682 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1749002379.994822 1073682 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1749002380.007698 1073682 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002380.007729 1073682 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002380.007734 1073682 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002380.007739 1073682 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-06-04 07:29:40.010881: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
usage: rl_energy_saving_predictor.py [-h] [-m MODEL] [-t TOPOLOGY] --tick TICK
rl_energy_saving_predictor.py: error: the following arguments are required: --tick
(venv) lk@lk-IdeaPad-5-Pro-14ACN6:~/Projects/accelcq-repos/cloudly/github/maveric/apps/energy_saving_demo$ python rl_energy_saving_predictor.py --tick 5
2025-06-04 07:29:53.545487: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1749002393.558195 1073985 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1749002393.562124 1073985 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1749002393.573359 1073985 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002393.573387 1073985 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002393.573390 1073985 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002393.573393 1073985 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-06-04 07:29:53.576384: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-06-04 07:29:55,830 - __main__ - INFO - --- Running RL Energy Saver Prediction for Tick 5 ---
2025-06-04 07:29:56,885 - __main__ - INFO - Loaded trained RL model from ./rl_energy_saver_agent_ppo.zip
2025-06-04 07:29:56,887 - __main__ - INFO - Predicted raw action (indices) for tick 5: [11  0 11 11 11  2 11 11 11 11 11 11  2 11  7]

--- Predicted Optimal Configuration ---
--- For Tick/Hour: 5 ---
 cell_id state cell_el_deg
cell_1_0   OFF         N/A
cell_1_1    ON         0.0
cell_1_2   OFF         N/A
cell_2_0   OFF         N/A
cell_2_1   OFF         N/A
cell_2_2    ON         4.0
cell_3_0   OFF         N/A
cell_3_1   OFF         N/A
cell_3_2   OFF         N/A
cell_4_0   OFF         N/A
cell_4_1   OFF         N/A
cell_4_2   OFF         N/A
cell_5_0    ON         4.0
cell_5_1   OFF         N/A
cell_5_2    ON        14.0
(venv) lk@lk-IdeaPad-5-Pro-14ACN6:~/Projects/accelcq-repos/cloudly/github/maveric/apps/energy_saving_demo$ python rl_energy_saving_predictor.py --tick 6
2025-06-04 07:32:35.085852: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1749002555.099251 1076400 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1749002555.103160 1076400 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1749002555.114679 1076400 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002555.114706 1076400 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002555.114710 1076400 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002555.114713 1076400 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-06-04 07:32:35.117771: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-06-04 07:32:37,367 - __main__ - INFO - --- Running RL Energy Saver Prediction for Tick 6 ---
2025-06-04 07:32:38,477 - __main__ - INFO - Loaded trained RL model from ./rl_energy_saver_agent_ppo.zip
2025-06-04 07:32:38,480 - __main__ - INFO - Predicted raw action (indices) for tick 6: [11  2 11 11  8  2 11 11 11 11  0 11 11 11 11]

--- Predicted Optimal Configuration ---
--- For Tick/Hour: 6 ---
 cell_id state cell_el_deg
cell_1_0   OFF         N/A
cell_1_1    ON         4.0
cell_1_2   OFF         N/A
cell_2_0   OFF         N/A
cell_2_1    ON        16.0
cell_2_2    ON         4.0
cell_3_0   OFF         N/A
cell_3_1   OFF         N/A
cell_3_2   OFF         N/A
cell_4_0   OFF         N/A
cell_4_1    ON         0.0
cell_4_2   OFF         N/A
cell_5_0   OFF         N/A
cell_5_1   OFF         N/A
cell_5_2   OFF         N/A
(venv) lk@lk-IdeaPad-5-Pro-14ACN6:~/Projects/accelcq-repos/cloudly/github/maveric/apps/energy_saving_demo$ python rl_energy_saving_predictor.py --tick 7
2025-06-04 07:32:56.382833: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1749002576.396762 1076761 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1749002576.400769 1076761 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1749002576.412553 1076761 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002576.412581 1076761 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002576.412586 1076761 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002576.412589 1076761 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-06-04 07:32:56.415818: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-06-04 07:32:58,754 - __main__ - INFO - --- Running RL Energy Saver Prediction for Tick 7 ---
2025-06-04 07:32:59,837 - __main__ - INFO - Loaded trained RL model from ./rl_energy_saver_agent_ppo.zip
2025-06-04 07:32:59,840 - __main__ - INFO - Predicted raw action (indices) for tick 7: [11  4 11 11 11  5 11 11 11 11 11 11 11 11 11]

--- Predicted Optimal Configuration ---
--- For Tick/Hour: 7 ---
 cell_id state cell_el_deg
cell_1_0   OFF         N/A
cell_1_1    ON         8.0
cell_1_2   OFF         N/A
cell_2_0   OFF         N/A
cell_2_1   OFF         N/A
cell_2_2    ON        10.0
cell_3_0   OFF         N/A
cell_3_1   OFF         N/A
cell_3_2   OFF         N/A
cell_4_0   OFF         N/A
cell_4_1   OFF         N/A
cell_4_2   OFF         N/A
cell_5_0   OFF         N/A
cell_5_1   OFF         N/A
cell_5_2   OFF         N/A
(venv) lk@lk-IdeaPad-5-Pro-14ACN6:~/Projects/accelcq-repos/cloudly/github/maveric/apps/energy_saving_demo$ python rl_energy_saving_predictor.py --tick 8
2025-06-04 07:35:50.249919: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1749002750.262858 1079486 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1749002750.266856 1079486 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1749002750.278149 1079486 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002750.278175 1079486 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002750.278180 1079486 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002750.278182 1079486 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-06-04 07:35:50.281244: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-06-04 07:35:52,569 - __main__ - INFO - --- Running RL Energy Saver Prediction for Tick 8 ---
2025-06-04 07:35:53,620 - __main__ - INFO - Loaded trained RL model from ./rl_energy_saver_agent_ppo.zip
2025-06-04 07:35:53,623 - __main__ - INFO - Predicted raw action (indices) for tick 8: [11  0 11 11 11  4 11 11 11 11 11 11 11 11 11]

--- Predicted Optimal Configuration ---
--- For Tick/Hour: 8 ---
 cell_id state cell_el_deg
cell_1_0   OFF         N/A
cell_1_1    ON         0.0
cell_1_2   OFF         N/A
cell_2_0   OFF         N/A
cell_2_1   OFF         N/A
cell_2_2    ON         8.0
cell_3_0   OFF         N/A
cell_3_1   OFF         N/A
cell_3_2   OFF         N/A
cell_4_0   OFF         N/A
cell_4_1   OFF         N/A
cell_4_2   OFF         N/A
cell_5_0   OFF         N/A
cell_5_1   OFF         N/A
cell_5_2   OFF         N/A
(venv) lk@lk-IdeaPad-5-Pro-14ACN6:~/Projects/accelcq-repos/cloudly/github/maveric/apps/energy_saving_demo$ python rl_energy_saving_predictor.py --tick 8
2025-06-04 07:36:17.811520: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1749002777.824983 1080000 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1749002777.828942 1080000 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1749002777.840139 1080000 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002777.840168 1080000 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002777.840173 1080000 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002777.840176 1080000 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-06-04 07:36:17.843486: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-06-04 07:36:20,167 - __main__ - INFO - --- Running RL Energy Saver Prediction for Tick 8 ---
2025-06-04 07:36:21,227 - __main__ - INFO - Loaded trained RL model from ./rl_energy_saver_agent_ppo.zip
2025-06-04 07:36:21,230 - __main__ - INFO - Predicted raw action (indices) for tick 8: [11  0 11 11 11  4 11 11 11 11 11 11 11 11 11]

--- Predicted Optimal Configuration ---
--- For Tick/Hour: 8 ---
 cell_id state cell_el_deg
cell_1_0   OFF         N/A
cell_1_1    ON         0.0
cell_1_2   OFF         N/A
cell_2_0   OFF         N/A
cell_2_1   OFF         N/A
cell_2_2    ON         8.0
cell_3_0   OFF         N/A
cell_3_1   OFF         N/A
cell_3_2   OFF         N/A
cell_4_0   OFF         N/A
cell_4_1   OFF         N/A
cell_4_2   OFF         N/A
cell_5_0   OFF         N/A
cell_5_1   OFF         N/A
cell_5_2   OFF         N/A
(venv) lk@lk-IdeaPad-5-Pro-14ACN6:~/Projects/accelcq-repos/cloudly/github/maveric/apps/energy_saving_demo$ python rl_energy_saving_predictor.py --tick 9
2025-06-04 07:36:27.543638: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1749002787.556832 1080227 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1749002787.560544 1080227 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1749002787.571108 1080227 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002787.571132 1080227 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002787.571136 1080227 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1749002787.571139 1080227 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-06-04 07:36:27.574839: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-06-04 07:36:29,827 - __main__ - INFO - --- Running RL Energy Saver Prediction for Tick 9 ---
2025-06-04 07:36:30,878 - __main__ - INFO - Loaded trained RL model from ./rl_energy_saver_agent_ppo.zip
2025-06-04 07:36:30,881 - __main__ - INFO - Predicted raw action (indices) for tick 9: [11  5 11 11 11  4 11 11 11 11 10 11  2 11 11]

--- Predicted Optimal Configuration ---
--- For Tick/Hour: 9 ---
 cell_id state cell_el_deg
cell_1_0   OFF         N/A
cell_1_1    ON        10.0
cell_1_2   OFF         N/A
cell_2_0   OFF         N/A
cell_2_1   OFF         N/A
cell_2_2    ON         8.0
cell_3_0   OFF         N/A
cell_3_1   OFF         N/A
cell_3_2   OFF         N/A
cell_4_0   OFF         N/A
cell_4_1    ON        20.0
cell_4_2   OFF         N/A
cell_5_0    ON         4.0
cell_5_1   OFF         N/A
cell_5_2   OFF         N/A
(venv) lk@lk-IdeaPad-5-Pro-14ACN6:~/Projects/accelcq-repos/cloudly/github/maveric/apps/energy_saving_demo$