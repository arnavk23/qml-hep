# Contributing Guide

## Scope
This repository is organized as independent task folders (`task1`, `task2`, `task3`, `task6`) plus root-level documentation and reporting.

When contributing, preserve task boundaries:
- keep task-specific code and outputs inside its task folder,
- avoid introducing cross-task imports unless explicitly required.

## Environment
Use the local virtual environment at repository root:

```powershell
.venv\Scripts\python.exe -m pip install -r task1\requirements.txt
.venv\Scripts\python.exe -m pip install -r task2\requirements.txt
.venv\Scripts\python.exe -m pip install -r task6\requirements.txt
```

## Development conventions
- Prefer minimal, focused changes.
- Keep generated artifacts in the existing output folders (`task2/figures`, `task2/models`, `task6/outputs`, etc.).
- Do not commit dataset caches or virtual environments.
- Update README/report content when metrics or behavior changes.

## Reproducibility expectations
If you modify model/training code:
1. Run at least one smoke configuration.
2. Record command used.
3. Record key metrics (loss/AUC/fidelity) in the relevant README/report.

## Reporting updates
The consolidated report is maintained in:
- `report/results_report.tex`

Compile with:

```powershell
cd report
.\build_report.ps1
```

If PowerShell execution policy blocks scripts, run `pdflatex` directly as documented in root `README.md`.
