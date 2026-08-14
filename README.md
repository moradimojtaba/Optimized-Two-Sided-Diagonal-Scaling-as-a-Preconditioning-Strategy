# Optimized Two-Sided Diagonal Scaling as a Preconditioning Strategy — V6.1

This repository contains the V6.1 research benchmark for optimized two-sided diagonal scaling. V6.1 is the scientifically corrected version tested successfully on a local computer before GitHub automation.

## What V6.1 evaluates

- Unscaled baseline
- Ruiz equilibration
- L-BFGS-B diagonal scaling
- Ruiz → L-BFGS-B hybrid (main hybrid method)
- Reproducible SA/GA stochastic runs
- Bayesian optimization when available
- Linear-system experiments with solver/preconditioner cost accounting
- Scalability experiments

The scientific objective is to evaluate whether optimized diagonal scaling can function as a useful preconditioning strategy in terms of condition-number reduction, computational cost, and linear-system solution performance. It does **not** claim that scaling universally replaces ILU or AMG.

## Reproducibility

The benchmark uses a fixed seed (`20260813`) and writes `configuration.json` and `run_report.json`. Summary statistics exclude failed/skipped runs. Solver timing uses repeated trials and reports median/IQR; total solver cost includes preconditioner setup.

## Local execution

```bash
python -m pip install -r requirements.txt
python code/illcondition_v6_1.py
```

The script creates `illcondition_v6_results/` automatically, including CSV/Excel tables, convergence plots, histograms, scaling files, solver results, scalability results, and reproducibility metadata.

## GitHub Actions

The workflow at `.github/workflows/run_v6_1.yml` runs the benchmark on GitHub-hosted Ubuntu with Python 3.11 and publishes the complete `illcondition_v6_results/` directory as a workflow artifact. It is triggered manually or when the V6.1 code, dependencies, or workflow changes.

For the first GitHub validation, results are deliberately kept as an **Artifact** rather than automatically committed back into the repository. This avoids overwriting reference results until the GitHub run has been compared against the successfully tested local V6.1 run.
