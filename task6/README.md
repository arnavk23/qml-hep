# Task VI: Quantum Representation Learning (Contrastive)

This task implements a simple quantum contrastive representation learning pipeline on MNIST.

## What is implemented

- **MNIST loading** via `torchvision.datasets.MNIST`
- **Trainable image-to-quantum-state function**
  - image -> small trainable encoder (`pre_net`) -> normalized feature vector
  - `AmplitudeEmbedding` + trainable `StronglyEntanglingLayers`
- **Two-image SWAP test circuit**
  - embeds both images as quantum states with shared trainable parameters
  - performs SWAP test and returns fidelity estimate
- **Contrastive objective**
  - same class pairs: maximize fidelity
  - different class pairs: minimize fidelity

## File

- `quantum_contrastive_mnist.py` — full training pipeline

## GitHub-ready folder layout

```text
task6/
├── quantum_contrastive_mnist.py
├── requirements.txt
├── README.md
├── .gitignore
├── outputs/                # validated run artifacts
├── outputs_strong/         # stronger run artifacts
├── data/                   # local MNIST cache (ignored)
└── report/
  ├── report.tex
  └── build_report.ps1
```

## Contrastive loss used

For pair label $y \in \{0,1\}$ and fidelity $F$:

$$
\mathcal{L} = y(1-F)^2 + (1-y)F^2
$$

- if $y=1$ (same class), loss is minimized when $F \to 1$
- if $y=0$ (different class), loss is minimized when $F \to 0$

## Run

From repository root:

```bash
.venv\Scripts\python.exe task6\quantum_contrastive_mnist.py --epochs 5 --batch-size 8 --max-train 256 --max-val 128 --out-dir task6\outputs
```

## Outputs

- `outputs/training_curves.png` — loss and fidelity curves
- `outputs/quantum_contrastive_mnist.pt` — trained parameters + history

## Latest validated extended run

Configuration used:

```bash
.venv\Scripts\python.exe task6\quantum_contrastive_mnist.py --epochs 3 --batch-size 8 --lr 0.005 --max-train 128 --max-val 64 --out-dir task6\outputs
```

Best validation epoch from saved checkpoint history:

- `Best epoch`: 2 / 3
- `Best val loss`: 0.1451
- `Best val same-class fidelity`: 0.7556
- `Best val different-class fidelity`: 0.2593

Final training metrics at epoch 3:

- `Train loss`: 0.1224
- `Train same-class fidelity`: 0.7637
- `Train different-class fidelity`: 0.2240

## Quick smoke test

```bash
.venv\Scripts\python.exe task6\quantum_contrastive_mnist.py --epochs 1 --batch-size 4 --max-train 64 --max-val 32 --out-dir task6\outputs
```

## LaTeX report

Generated report source is in:

- `report/report.tex`

To build locally (PowerShell):

```powershell
cd task6\report
.\build_report.ps1
```

If `pdflatex` is not installed, install MiKTeX or TeX Live first.
