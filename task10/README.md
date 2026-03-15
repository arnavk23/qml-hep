# Task 10: Diffusion Models for Fast Detector Simulation

This task implements a Diffusion Network model to represent events from task 1, compares original and reconstructed events visually and quantitatively, and sketches ideas for quantum diffusion architectures.

## Steps
1. Load the dataset from task1.
2. Train a Diffusion Network model for event reconstruction.
3. Show side-by-side comparison of original and reconstructed events.
4. Evaluate with a suitable metric (e.g., MSE, Wasserstein distance).
5. Compare to VAE results.
6. Sketch quantum diffusion architecture ideas.

## Files
- diffusion_model.py: Diffusion model implementation, visualization, evaluation
- quantum_diffusion_sketch.md: Quantum extension discussion and architecture sketch
- requirements.txt: Dependencies
- README.md: Instructions

## Requirements
- Python 3.8+
- torch
- numpy
- matplotlib
- scikit-learn

## Usage
1. Activate your .venv.
2. Run `python diffusion_model.py` to train and evaluate the diffusion model.
3. See quantum_diffusion_sketch.md for quantum architecture ideas.

---
