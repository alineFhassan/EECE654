# EECE654

This repository contains the scripts used to generate the results presented in the paper.

## Files

### MATLAB
- **`fmincon_project.m`**: Joint dynamic multi-slot optimization using fmincon (MATLAB optimization solver).

### Python - Reinforcement Learning Algorithms

#### TD3 (Twin Delayed DDPG)
- **`TD3_one_run.py`**: Single run implementation of the TD3 algorithm.
- **`TD3_multiple_runs.py`**: Same as `TD3_one_run.py` but with a for loop to perform multiple runs.

#### DDPG (Deep Deterministic Policy Gradient)
- **`DDPG_one_run.py`**: Single run implementation of the DDPG algorithm.
- **`DDPG_multiple_runs.py`**: Same as `DDPG_one_run.py` but with a for loop to perform multiple runs.

#### MERL
- **`merl_one_run.py`**: Single run implementation of the MERL algorithm.
- **`merl_multiple_runs.py`**: Same as `merl_one_run.py` but with a for loop to perform multiple runs.

## Note

The `*_multiple_runs.py` files are identical to their corresponding `*_one_run.py` files, except they include a for loop to execute multiple runs for statistical analysis.
