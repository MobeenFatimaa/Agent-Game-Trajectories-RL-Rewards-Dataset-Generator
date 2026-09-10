# Agent-Game-Trajectories-RL-Rewards-Dataset-Generator

A Python-based generator and automated validation suite for constructing large-scale synthetic reinforcement learning datasets. This project simulates agent decision-making dynamics, action policy entropy, and state-transition rewards across multiple RL algorithms (PPO, DQN, Q-Learning, and Random-Walker) navigating a 2D Gridworld.

---

## Repository Overview

This repository provides scripts to generate and audit a **100,000-row step-level dataset** formatted specifically for publication on Kaggle and offline reinforcement learning benchmarking.

* **`generate_dataset.py`**: Programmatically generates step-level agent trajectories with exact row-count control.
* **`validate_dataset.py`**: Standalone automated test suite checking schema layout, math consistency, JSON formatting, and grid boundary conditions.
* **`requirements.txt`**: Minimal dependencies required for runtime execution.

---

## Dataset Schema

The generated dataset (`synthetic_rl_trajectories.csv`) contains **11 columns**:

| Column              | Data Type   | Description                                                                 |
| :------------------ | :---------- | :-------------------------------------------------------------------------- |
| `Episode_ID`        | String      | Unique identifier for each full game episode (e.g., `EP-00001`).            |
| `Step`              | Integer     | Sequential step counter within the episode ($1 \le \text{Step} \le 50$).    |
| `Agent_Type`        | Categorical | Policy algorithm (`PPO`, `DQN`, `Q-Learning`, `Random-Walker`).             |
| `State_JSON`        | JSON String | Pre-action state vector (`agent_pos`, `target_pos`, `dist_to_target`).      |
| `Action`            | Integer     | Discrete move index (`0`: UP, `1`: RIGHT, `2`: DOWN, `3`: LEFT).            |
| `Action_Name`       | String      | Human-readable movement direction (`UP`, `RIGHT`, `DOWN`, `LEFT`).          |
| `Reward`            | Float       | Immediate numeric payoff returned by the environment.                       |
| `Cumulative_Reward` | Float       | Sequential running total of rewards in the episode.                         |
| `Next_State_JSON`   | JSON String | Post-action state vector showing updated position and target distance.      |
| `Policy_Entropy`    | Float       | Measure of action policy uncertainty/randomness at decision time.           |
| `Done`              | Boolean     | Terminal flag indicating if episode ended on current step (`True`/`False`). |

---

## Quickstart & Setup

### 1. Clone Repository & Install Dependencies

```bash
git clone https://github.com/MobeenFatimaa/Agent-Game-Trajectories-RL-Rewards-Dataset-Generator.git
cd Agent-Game-Trajectories-RL-Rewards-Dataset-Generator
pip install -r requirements.txt
```

### 2. Generate Dataset

Run the generator script to create the 100,000-row dataset CSV:

```bash
python generate_dataset.py
```

### 3. Run Automated Validation

Execute the audit suite to verify data integrity:

```bash
python validate_dataset.py
```
----
Kaggle Daataset
https://www.kaggle.com/datasets/mobeenfatimah/gridworld-rl-trajectories100k-agent-decision-logs
## License

This project is licensed under the [MIT License](LICENSE).
