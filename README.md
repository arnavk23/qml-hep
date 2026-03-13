# QML Workspace (GSoC Tasks)

This repository contains multiple quantum/ML tasks implemented as part of a GSoC-style project workflow.

## Repository structure

- `docs/` — project context and research planning notes
- `qmaml_hep/` — portable Q-MAML core for HEP-inspired few-shot tasks
- `scripts/` — experiment and benchmark runners for Q-MAML
- `task1/` — quantum circuit fundamentals (state prep, SWAP test)
- `task2/` — classical GNN jet classification (DGCNN vs GAT)
- `task3/` — open-ended commentary task (quantum computing/QML perspective)
- `task6/` — quantum representation learning with contrastive fidelity loss on MNIST
- `test/` — smoke tests for Q-MAML components
- `report/` — consolidated LaTeX report for the full workspace

## Task highlights

### Task 1: Quantum Computing Part
- Implemented 5-qubit circuit with Hadamard/CNOT/SWAP/RX operations
- Implemented SWAP-test circuit to estimate overlap between two 2-qubit states
- Generated diagrams and outputs in `task1/`

### Task 2: Classical GNN for Quark/Gluon Classification
- Graph construction from jet point clouds
- Two architectures implemented:
  - DGCNN (`task2/src/dgcnn.py`)
  - GAT (`task2/src/gat_net.py`)
- Training/evaluation pipeline in `task2/src/train.py`
- Figures in `task2/figures/`, checkpoints in `task2/models/`, logs in `task2/logs/`

### Task 3: Open Task
- Personal technical commentary on quantum computing/QML
- Includes algorithm/software perspective and proposed directions

### Task 6: Quantum Representation Learning
- MNIST pair sampling with same/different class labels
- Trainable image-to-quantum-state embedding
- SWAP-test fidelity circuit
- Contrastive objective to push same-class fidelity up and different-class fidelity down
- Outputs in `task6/outputs/` and `task6/outputs_strong/`

### Q-MAML Extension (Implemented Here)
- Added reusable package in `qmaml_hep/` with:
  - synthetic task generation (`data.py`)
  - variational quantum classifier (`model.py`)
  - MAML-style inner/outer optimization (`qmaml.py`)
  - classical + quantum baselines (`baselines.py`)
- Added execution scripts in `scripts/`:
  - `python -m scripts.run_experiment`
  - `python -m scripts.run_benchmarks`
  - `python -m scripts.summarize_benchmarks`
- Added smoke test: `python -m unittest test/test_qmaml_smoke.py -v`
- Setup dependencies via: `pip install -r requirements-qmaml.txt`

## Consolidated report

A full LaTeX report is provided at:
- `report/results_report.tex`
- compiled PDF: `report/results_report.pdf`

Build (PowerShell):

```powershell
cd report
.\build_report.ps1
```

If script execution is restricted, run directly:

```powershell
pdflatex -interaction=nonstopmode -output-directory report report\results_report.tex
pdflatex -interaction=nonstopmode -output-directory report report\results_report.tex
```

## Repository governance

- License: `LICENSE` (MIT)
- Contribution workflow: `CONTRIBUTING.md`
