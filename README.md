# Optimized Two-Sided Diagonal Scaling as a Preconditioning Strategy

This repository contains the computational code, experimental data, numerical results, and supplementary materials associated with the research study on optimized two-sided diagonal scaling as a preconditioning strategy for ill-conditioned linear systems.

## Research objective

The main objective of this study is to investigate whether diagonal scaling optimization can serve as an effective preconditioning strategy by reducing matrix condition numbers, improving the numerical solution of linear systems, and reducing computational cost.

The study focuses on the relationship between:

- condition-number reduction,
- computational cost,
- scalability,
- preconditioning,
- and linear-system solver performance.

## Methods

The computational experiments include:

- Unscaled baseline
- Ruiz scaling
- L-BFGS optimization
- Ruiz → L-BFGS hybrid scaling
- Simulated Annealing
- Genetic Algorithm
- Bayesian Optimization
- ILU preconditioning
- AMG preconditioning
- GMRES
- Conjugate Gradient (CG)

## Repository structure

```text
code/
    Main Python implementation

data/
    Experimental datasets and numerical results

manuscript/
    Manuscript source and PDF

results/
    Figures, histograms, and other numerical results
