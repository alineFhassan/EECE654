# EECE654

This repository contains the scripts used to generate the results presented in the paper.

## Files

### MATLAB
- **`fmincon_project.m`**: Joint dynamic multi-slot optimization using fmincon (MATLAB optimization solver).

### Python - Reinforcement Learning Algorithms and Plotting

#### Baseline
- **`baseline_one_run.py`**: Baseline approach - single run implementation. Saves baseline results (reward, SNR, rate) to an Excel file.

#### TD3 (Twin Delayed DDPG)
- **`TD3_one_run.py`**: Single run implementation of the TD3 algorithm.
- **`TD3_multiple_runs.py`**: Same as `TD3_one_run.py` but with a for loop to perform multiple runs.

#### DDPG (Deep Deterministic Policy Gradient)
- **`DDPG_one_run.py`**: Single run implementation of the DDPG algorithm.
- **`DDPG_multiple_runs.py`**: Same as `DDPG_one_run.py` but with a for loop to perform multiple runs.

#### MERL
- **`merl_one_run.py`**: Single run implementation of the MERL algorithm.
- **`merl_multiple_runs.py`**: Same as `merl_one_run.py` but with a for loop to perform multiple runs.

#### Plotting
- **`plotting.py`**: Script to plot and compare the performance of all approaches (MERL, TD3, DDPG, Random baseline) over episodes.
  - Before running, make sure you have saved the results of each method in Excel files named:
    - `MERL_results.xlsx`
    - `TD3_results.xlsx`
    - `DDPG_results.xlsx`
    - `RandomBaseline_results.xlsx`
  - Each Excel file should contain three sheets: `EpisodeReward`, `EpisodeSNR`, and `EpisodeRate`, as produced by the corresponding training/baseline scripts.
  - Update the file paths in `plotting.py` if your Excel files are stored in a different directory, then run `plotting.py` to generate the plots.

## Notes

- **Multiple-run scripts**: The `*_multiple_runs.py` files are identical to their corresponding `*_one_run.py` files, except they include a for loop to execute multiple runs for statistical analysis.
- **Alternative way to reproduce results**: Instead of running the individual scripts, you can simply open and run the notebook `PA_ISAC.ipynb`, which contains the full pipeline needed to regenerate the results.
