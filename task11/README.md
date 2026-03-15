# Task XI: Classical-Quantum Embedding with PQC

This task implements a simple embedding pipeline using a neural network (MLP) to estimate parameters for a Parameterized Quantum Circuit (PQC). The input data is sampled from a normal distribution, and the PQC prepares quantum states based on the estimated parameters. Training uses MSE loss.

## Steps
1. Generate normally distributed input data.
2. Use an MLP (2-3 linear layers) to estimate PQC parameters.
3. Prepare quantum states with 4-5 qubits using the estimated parameters.
4. Train the model using MSE loss.

## Files
- pqc_embedding.py: Implementation and training
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
2. Run `python pqc_embedding.py` to train and evaluate the embedding pipeline.

---
