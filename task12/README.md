# Task XII: Temporal Difference Learning with PQC Embedding

This task extends Task XI by using a simple temporal difference algorithm (e.g., DQN) to train the PQC embedding pipeline. The reward function uses MSE.

## Steps
1. Generate normally distributed input data.
2. Use an MLP to estimate PQC parameters.
3. Prepare quantum states with 4-5 qubits using the estimated parameters.
4. Train using DQN with MSE reward.

## Files
- pqc_td_learning.py: Implementation and training
- requirements.txt: Dependencies
- README.md: Instructions

## Requirements
- Python 3.8+
- torch
- cirq
- numpy
- matplotlib

## Usage
1. Activate your .venv.
2. Run `python pqc_td_learning.py` to train and evaluate the pipeline with temporal difference learning.

---
