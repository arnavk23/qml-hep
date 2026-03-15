# Task VII: Equivariant Quantum Neural Networks

This task implements a Z₂ × Z₂ equivariant quantum neural network (QNN) for a classification problem. The dataset is generated to respect Z₂ × Z₂ symmetry (mirroring along y=x), as described in https://arxiv.org/abs/2205.06217.

## Steps
1. Generate a symmetric dataset with two features (x₁, x₂) and two classes.
2. Train a standard QNN for classification.
3. Train a Z₂ × Z₂ equivariant QNN and compare results.

## Files
- equivariant_qnn.py: Implementation and comparison
- requirements.txt: Dependencies
- README.md: Instructions

## Requirements
- Python 3.8+
- cirq
- numpy
- matplotlib
- scikit-learn

## Usage
1. Activate your .venv.
2. Run `python equivariant_qnn.py` to generate data, train both QNNs, and compare results.

---
