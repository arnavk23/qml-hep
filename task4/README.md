# Task IV: Quantum Generative Adversarial Network (QGAN)

This task implements a QGAN using Google Cirq and TensorFlow Quantum (TFQ) to separate signal events from background events in High Energy Physics data. The input is provided in NumPy NPZ format, with 100 training and 100 testing samples. Signal events are labeled as 1, background as 0.

## Steps:
1. Load the dataset from NPZ files.
2. Build QGAN architecture using Cirq and TFQ.
3. Train the QGAN to distinguish signal from background.
4. Evaluate performance using accuracy and AUC.
5. Fine-tune the model for improved results.

## Files:
- qgan.py: Main implementation
- requirements.txt: Dependencies
- README.md: Instructions

## Requirements
- Python 3.8+
- cirq
- tensorflow
- tensorflow-quantum
- numpy
- scikit-learn

## Usage
1. Download the input NPZ file (simulated with Delphes) and place it in this folder as `input_data.npz`.
	- The file should contain arrays: `X_train`, `y_train`, `X_test`, `y_test`.
	- Signal events are labeled as 1, background as 0.
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
3. Run the QGAN training and evaluation:
	```bash
	python qgan.py
	```

## Fine-tuning and Evaluation
To improve performance:
- Adjust quantum circuit depth and structure in `qgan.py`.
- Tune learning rates, batch sizes, and optimizer settings.
- Experiment with different numbers of qubits.
- Evaluate using classification accuracy and Area Under ROC Curve (AUC).
- Use cross-validation or grid search for hyperparameter optimization.

Results will be printed after training. For further analysis, modify the script to save predictions and plot ROC curves.
