#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
illcondition_v6.py
Research-grade numerical benchmark for optimized two-sided diagonal scaling.

Main research question:
Can optimized diagonal scaling act as a practical preconditioning strategy
by improving condition number and actual iterative linear-system performance
at a competitive computational cost?

Design goals:
- Preserve strengths of the original implementation: log-parameterization,
  real GA crossover/mutation, timing, classical test matrices.
- Preserve strengths of v5: Ruiz -> L-BFGS hybrid, log-condition objective,
  convergence histories, structured result export.
- Directly address reviewer concerns:
  1) computational effort: runtime + objective evaluations + iterations;
  2) standard preconditioners: ILU, AMG (when pyamg is available), IC(0) for SPD;
  3) broader benchmarks and scalability;
- Actual linear-system experiments with GMRES and CG where appropriate.
- Reproducible stochastic experiments.
- No manual editing required: all settings are defined below and output folders
  are created automatically.

The script is intentionally self-contained. Optional packages are detected and,
when possible, installed automatically:
  scipy, pandas, matplotlib, openpyxl, scikit-learn, scikit-optimize, pyamg

If an optional package cannot be installed, the corresponding method is marked
as unavailable rather than silently fabricating a result.

Python >= 3.10 recommended.
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import shutil
import subprocess
import platform
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

# ---------------------------------------------------------------------------
# Automatic dependency handling
# ---------------------------------------------------------------------------

def ensure_package(import_name: str, pip_name: Optional[str] = None) -> bool:
    """Try to import a package; if missing, try a quiet pip installation."""
    try:
        __import__(import_name)
        return True
    except Exception:
        pkg = pip_name or import_name
        print(f"[INFO] Optional dependency '{pkg}' not found. Trying installation...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            __import__(import_name)
            print(f"[OK] Installed '{pkg}'.")
            return True
        except Exception:
            print(f"[WARN] Could not install '{pkg}'. Related method will be skipped.")
            return False

# Required scientific stack
for _imp, _pip in [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
]:
    if not ensure_package(_imp, _pip):
        raise RuntimeError(
            f"Required package '{_pip}' is unavailable. "
            f"Please run this script in a Python environment with internet/pip access."
        )

# Optional
HAVE_OPENPYXL = ensure_package("openpyxl", "openpyxl")
HAVE_SKLEARN = ensure_package("sklearn", "scikit-learn")
HAVE_SKOPT = ensure_package("skopt", "scikit-optimize")
HAVE_PYAMG = ensure_package("pyamg", "pyamg")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import optimize
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix, diags, eye
from scipy.sparse.linalg import (
    LinearOperator,
    gmres,
    cg,
    spilu,
)
from scipy.linalg import solve_triangular

if HAVE_SKLEARN:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from scipy.stats import norm

if HAVE_SKOPT:
    from skopt import gp_minimize
    from skopt.space import Real

if HAVE_PYAMG:
    import pyamg

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Global configuration: no editing required
# ---------------------------------------------------------------------------

SEED = 20260808

# ===================== IMPORTANT: FIGURES SAVED TO ROOT =====================
# All figures are saved to the "figures/" folder in the project root.
# Numerical results are saved to "illcondition_v6_results/" (excluded from git).
# ============================================================================

OUTDIR = Path.cwd() / "illcondition_v6_results"      # Numerical results only
FIGDIR = Path.cwd() / "figures"                      # Figures for the paper
CONVDIR = FIGDIR / "convergence"                     # Convergence history figures
SCALE_DIR = OUTDIR / "scalings"                      # Saved scaling factors
RAW_DIR = OUTDIR / "raw_runs"                        # Raw per-run results
SCALABILITY_DIR = OUTDIR / "scalability"             # Scalability experiment

# Create all directories
for p in [OUTDIR, FIGDIR, CONVDIR, SCALE_DIR, RAW_DIR, SCALABILITY_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Core optimization
LOG_LOWER = -5.0
LOG_UPPER = 5.0
LBFGS_MAXITER = 120
LBFGS_HYBRID_MAXITER = 70
LBFGS_FTOL = 1e-10
LBFGS_GTOL = 1e-6

# Ruiz
RUIZ_MAXITER = 60
RUIZ_TOL = 1e-7

# Stochastic methods
N_STOCHASTIC_REPEATS = 10
SA_MAXITER = 70
SA_VISIT = 2.62
SA_ACCEPT = -5.0
GA_POPSIZE = 18
GA_GENS = 30
GA_ELITE = 2
GA_MUTATION_RATE = 0.15
GA_MUTATION_SCALE = 0.25

# BO: only used for small/moderate dimensions to avoid a misleading
# high-dimensional Gaussian-process benchmark.
BO_MAX_DIM = 12
BO_INITIAL_POINTS = 6
BO_CALLS = 18

# Linear solve experiments
SOLVER_TOL = 1e-8
SOLVER_MAXITER = 1000
SOLVER_RESTART = 50
SOLVER_TRIALS = 1

# Scalability sizes chosen to keep a normal workstation runtime reasonable.
SCALABILITY_DENSE_SIZES = [5, 10, 15, 20, 30]
SCALABILITY_SPARSE_SIZES = [100, 300, 600, 1000]

# Dense benchmark limits: expensive global optimizers are only run on
# moderate-size matrices. The scaling and preconditioner baselines still run.
GLOBAL_OPT_MAX_N = 15
LINEAR_SOLVE_MAX_N_DENSE = 100
AMG_MAX_N = 5000

# ---------------------------------------------------------------------------
# Utility classes
# ---------------------------------------------------------------------------

class EvaluationCounter:
    def __init__(self):
        self.count = 0

    def inc(self):
        self.count += 1

@dataclass
class MethodResult:
    matrix: str
    method: str
    n: int
    status: str
    seed: int
    initial_cond: float
    final_cond: float
    reduction_factor: float
    log_objective: float
    runtime_sec: float
    objective_evaluations: int
    iterations: int
    solver_applicable: bool = False
    solver: str = ""
    solver_status: str = ""
    solver_iterations: int = -1
    solver_time_sec: float = float("nan")
    solver_residual: float = float("nan")
    preconditioner_build_sec: float = float("nan")
    notes: str = ""

# ---------------------------------------------------------------------------
# Matrix generators
# ---------------------------------------------------------------------------

def hilbert(n: int) -> np.ndarray:
    i = np.arange(n, dtype=float)[:, None]
    j = np.arange(n, dtype=float)[None, :]
    return 1.0 / (i + j + 1.0)

def vandermonde(n: int) -> np.ndarray:
    # Nodes chosen away from exact duplicate values.
    x = np.linspace(0.15, 0.95, n)
    return np.vander(x, N=n, increasing=True)

def cauchy_matrix(n: int) -> np.ndarray:
    x = np.arange(1, n + 1, dtype=float)
    y = x + 0.37
    return 1.0 / (x[:, None] + y[None, :])

def rump4() -> np.ndarray:
    # A compact Rump-style ill-conditioned benchmark.
    return np.array([
        [ 1.0,  2.0, -1.0,  0.5],
        [ 2.0,  4.0, -2.0,  1.0],
        [ 3.0,  6.0, -2.999999,  1.5],
        [ 4.0,  8.0, -4.0,  2.000001],
    ], dtype=float)

def finite_difference(n: int, shift: float = 0.25) -> sparse.csr_matrix:
    main = (2.0 + shift) * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    return sparse.diags([off, main, off], [-1, 0, 1], format="csr")

def sparse_nonsymmetric(n: int, seed: int = SEED) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    main = 4.0 * np.ones(n)
    lower = -0.8 * np.ones(n - 1)
    upper = -1.2 * np.ones(n - 1)
    A = sparse.diags([lower, main, upper], [-1, 0, 1], format="lil")
    # A few controlled nonsymmetric perturbations
    for _ in range(max(1, n // 20)):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i != j:
            A[i, j] += float(rng.uniform(-0.2, 0.2))
    return A.tocsr()

def random_spectrum(n: int, cond_target: float = 1e8, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    s = np.logspace(0.0, math.log10(cond_target), n)
    return U @ np.diag(s) @ V.T

def spd_random(n: int, cond_target: float = 1e5, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.logspace(0.0, math.log10(cond_target), n)
    return Q @ np.diag(eigs) @ Q.T

def original_A1() -> np.ndarray:
    return np.array([
        [1., 1., 1.],
        [1., 2., 3.],
        [1., 3., 6.]
    ], dtype=float)

def original_A2() -> np.ndarray:
    return np.array([
        [0.0927, 17.08, 0.31, 12.75],
        [1.78, 54.02, 1.5, 14.77],
        [0.346, 0.068, 0.263, 0.023],
        [1.375, 45.15, 0.051, 1.431]
    ], dtype=float)

def build_benchmarks() -> List[Tuple[str, Any, str]]:
    """Return (name, matrix, family)."""
    return [
        ("A1_3x3", original_A1(), "original"),
        ("A2_4x4", original_A2(), "original"),
        ("Hilbert10", hilbert(10), "dense_structured"),
        ("Hilbert15", hilbert(15), "dense_structured"),
        ("Vandermonde10", vandermonde(10), "dense_structured"),
        ("Cauchy8", cauchy_matrix(8), "dense_structured"),
        ("Rump4", rump4(), "dense_structured"),
        ("FiniteDiff50", finite_difference(50), "sparse_spd"),
        ("SparseNonsym100", sparse_nonsymmetric(100), "sparse_nonsym"),
        ("RandomSpectrum12", random_spectrum(12, 1e8, SEED), "random_dense"),
        ("RandomSPD12", spd_random(12, 1e6, SEED), "random_spd"),
    ]

# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def as_dense(A) -> np.ndarray:
    if sparse.issparse(A):
        return A.toarray()
    return np.asarray(A, dtype=float)

def safe_cond(A) -> float:
    try:
        B = as_dense(A)
        if not np.all(np.isfinite(B)):
            return float("inf")
        s = np.linalg.svd(B, compute_uv=False)
        if len(s) == 0 or s[-1] <= np.finfo(float).tiny:
            return float("inf")
        return float(s[0] / s[-1])
    except Exception:
        return float("inf")

def apply_scaling(A, d1, d2):
    if sparse.issparse(A):
        return diags(d1) @ A @ diags(d2)
    return d1[:, None] * np.asarray(A) * d2[None, :]

def normalized_log_vector_to_scales(x: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    u = np.asarray(x[:n], dtype=float).copy()
    v = np.asarray(x[n:], dtype=float).copy()
    # Remove redundant global scaling degrees of freedom.
    u -= np.mean(u)
    v -= np.mean(v)
    u = np.clip(u, LOG_LOWER, LOG_UPPER)
    v = np.clip(v, LOG_LOWER, LOG_UPPER)
    d1 = np.exp(u)
    d2 = np.exp(v)
    return d1, d2

def scales_to_normalized_log(d1, d2) -> np.ndarray:
    d1 = np.maximum(np.asarray(d1, dtype=float), 1e-15)
    d2 = np.maximum(np.asarray(d2, dtype=float), 1e-15)
    u = np.log(d1)
    v = np.log(d2)
    u -= np.mean(u)
    v -= np.mean(v)
    return np.concatenate([u, v])

def make_counter_objective(A, counter: EvaluationCounter, history: Optional[dict] = None):
    n = A.shape[0]

    def objective(x):
        counter.inc()
        try:
            d1, d2 = normalized_log_vector_to_scales(x, n)
            B = apply_scaling(A, d1, d2)
            cond = safe_cond(B)
            val = math.log(max(cond, 1.0))
            if history is not None:
                history["evaluation"].append(counter.count)
                history["log_cond"].append(val)
                history["cond"].append(cond)
            if not np.isfinite(val):
                return 1e100
            return float(val)
        except Exception:
            if history is not None:
                history["evaluation"].append(counter.count)
                history["log_cond"].append(1e100)
                history["cond"].append(float("inf"))
            return 1e100

    return objective

def result_from_x(A, x) -> Tuple[float, np.ndarray, np.ndarray]:
    n = A.shape[0]
    d1, d2 = normalized_log_vector_to_scales(x, n)
    cond = safe_cond(apply_scaling(A, d1, d2))
    return cond, d1, d2

# ---------------------------------------------------------------------------
# Ruiz equilibration
# ---------------------------------------------------------------------------

def ruiz_equilibration(A, maxiter=RUIZ_MAXITER, tol=RUIZ_TOL):
    """Two-sided Ruiz equilibration using Euclidean row/column norms."""
    B = A.copy().astype(float)
    n = B.shape[0]
    r = np.ones(n)
    c = np.ones(n)

    for it in range(1, maxiter + 1):
        old = B.copy() if not sparse.issparse(B) and n <= 100 else None

        if sparse.issparse(B):
            row_norms = np.sqrt(np.asarray(B.multiply(B).sum(axis=1)).ravel())
        else:
            row_norms = np.linalg.norm(B, axis=1)

        row_norms = np.maximum(row_norms, 1e-15)
        sr = 1.0 / np.sqrt(row_norms)
        B = diags(sr) @ B if sparse.issparse(B) else sr[:, None] * B
        r *= sr

        if sparse.issparse(B):
            col_norms = np.sqrt(np.asarray(B.multiply(B).sum(axis=0)).ravel())
        else:
            col_norms = np.linalg.norm(B, axis=0)

        col_norms = np.maximum(col_norms, 1e-15)
        sc = 1.0 / np.sqrt(col_norms)
        B = B @ diags(sc) if sparse.issparse(B) else B * sc[None, :]
        c *= sc

        if old is not None:
            rel = np.linalg.norm(B - old) / max(np.linalg.norm(old), 1e-15)
            if rel < tol:
                return r, c, B, it

    return r, c, B, maxiter

# ---------------------------------------------------------------------------
# L-BFGS and hybrid
# ---------------------------------------------------------------------------

def run_lbfgs(A, x0=None, maxiter=LBFGS_MAXITER) -> Dict[str, Any]:
    n = A.shape[0]
    if x0 is None:
        x0 = np.zeros(2 * n)

    x0 = np.clip(np.asarray(x0, dtype=float), LOG_LOWER, LOG_UPPER)
    counter = EvaluationCounter()
    history = {"iteration": [0], "evaluation": [], "cond": [], "log_cond": []}
    objective = make_counter_objective(A, counter, history)

    initial_cond = safe_cond(A)
    history["cond"].append(initial_cond)
    history["log_cond"].append(math.log(max(initial_cond, 1.0)))

    iteration_counter = {"k": 0}

    def callback(xk):
        iteration_counter["k"] += 1
        d1, d2 = normalized_log_vector_to_scales(xk, n)
        cond = safe_cond(apply_scaling(A, d1, d2))
        history["iteration"].append(iteration_counter["k"])
        history["cond"].append(cond)
        history["log_cond"].append(math.log(max(cond, 1.0)))

    t0 = time.perf_counter()
    res = optimize.minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=[(LOG_LOWER, LOG_UPPER)] * (2 * n),
        callback=callback,
        options={
            "maxiter": int(maxiter),
            "ftol": LBFGS_FTOL,
            "gtol": LBFGS_GTOL,
            "maxls": 20,
        },
    )
    elapsed = time.perf_counter() - t0

    final_cond, d1, d2 = result_from_x(A, res.x)
    return {
        "x": res.x,
        "d1": d1,
        "d2": d2,
        "fun": final_cond,
        "runtime": elapsed,
        "evaluations": counter.count,
        "iterations": iteration_counter["k"],
        "history": history,
        "success": bool(res.success),
        "message": str(res.message),
    }

def run_ruiz_lbfgs(A) -> Dict[str, Any]:
    n = A.shape[0]
    t0 = time.perf_counter()
    r, c, B_ruiz, ruiz_iters = ruiz_equilibration(A)
    x0 = scales_to_normalized_log(r, c)

    # Evaluate starting point once for the refinement history.
    res = run_lbfgs(A, x0=x0, maxiter=LBFGS_HYBRID_MAXITER)
    elapsed = time.perf_counter() - t0
    res["runtime"] = elapsed
    res["ruiz_iterations"] = ruiz_iters
    res["ruiz_cond"] = safe_cond(B_ruiz)
    return res

# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------

def run_sa(A, seed: int) -> Dict[str, Any]:
    n = A.shape[0]
    dim = 2 * n
    bounds = [(LOG_LOWER, LOG_UPPER)] * dim
    counter = EvaluationCounter()
    history = {"evaluation": [], "log_cond": [], "cond": []}
    objective = make_counter_objective(A, counter, history)

    t0 = time.perf_counter()
    res = optimize.dual_annealing(
        objective,
        bounds=bounds,
        maxiter=SA_MAXITER,
        visit=SA_VISIT,
        accept=SA_ACCEPT,
        seed=seed,
        no_local_search=True,
    )
    elapsed = time.perf_counter() - t0

    final_cond, d1, d2 = result_from_x(A, res.x)
    return {
        "x": res.x,
        "d1": d1,
        "d2": d2,
        "fun": final_cond,
        "runtime": elapsed,
        "evaluations": counter.count,
        "iterations": SA_MAXITER,
        "history": history,
        "success": bool(res.success),
        "message": str(res.message),
    }

# ---------------------------------------------------------------------------
# Genetic Algorithm with selection, crossover, mutation, elitism
# ---------------------------------------------------------------------------

def tournament_select(pop, scores, rng, k=3):
    idx = rng.integers(0, len(pop), size=k)
    best = idx[np.argmin(scores[idx])]
    return pop[best].copy()

def run_ga(A, seed: int) -> Dict[str, Any]:
    n = A.shape[0]
    dim = 2 * n
    rng = np.random.default_rng(seed)
    counter = EvaluationCounter()

    def objective(x):
        counter.inc()
        d1, d2 = normalized_log_vector_to_scales(x, n)
        cond = safe_cond(apply_scaling(A, d1, d2))
        return math.log(max(cond, 1.0))

    t0 = time.perf_counter()
    pop = rng.uniform(LOG_LOWER, LOG_UPPER, size=(GA_POPSIZE, dim))
    best_x = pop[0].copy()
    best_score = float("inf")
    history = {"generation": [], "cond": [], "evaluation": []}

    for gen in range(GA_GENS):
        scores = np.array([objective(ind) for ind in pop])
        order = np.argsort(scores)

        if scores[order[0]] < best_score:
            best_score = float(scores[order[0]])
            best_x = pop[order[0]].copy()

        history["generation"].append(gen)
        history["cond"].append(math.exp(min(best_score, 700.0)))
        history["evaluation"].append(counter.count)

        elites = pop[order[:GA_ELITE]].copy()
        new_pop = [e for e in elites]

        while len(new_pop) < GA_POPSIZE:
            p1 = tournament_select(pop, scores, rng)
            p2 = tournament_select(pop, scores, rng)

            # Arithmetic crossover
            alpha = rng.random(dim)
            child = alpha * p1 + (1.0 - alpha) * p2

            # Gaussian mutation
            mask = rng.random(dim) < GA_MUTATION_RATE
            if np.any(mask):
                child[mask] += rng.normal(
                    0.0, GA_MUTATION_SCALE, size=np.sum(mask)
                )

            child = np.clip(child, LOG_LOWER, LOG_UPPER)
            new_pop.append(child)

        pop = np.asarray(new_pop)

    elapsed = time.perf_counter() - t0
    final_cond, d1, d2 = result_from_x(A, best_x)

    return {
        "x": best_x,
        "d1": d1,
        "d2": d2,
        "fun": final_cond,
        "runtime": elapsed,
        "evaluations": counter.count,
        "iterations": GA_GENS,
        "history": history,
        "success": True,
        "message": "GA completed",
    }

# ---------------------------------------------------------------------------
# Bayesian Optimization
# ---------------------------------------------------------------------------

def run_bo(A, seed: int) -> Dict[str, Any]:
    """
    Real Bayesian optimization using scikit-optimize when available.
    Restricted to dim <= BO_MAX_DIM. The restriction is intentional:
    GP-based BO is not a sensible default for large 2n-dimensional spaces.
    """
    n = A.shape[0]
    dim = 2 * n
    if not HAVE_SKOPT:
        return {"status": "unavailable", "message": "scikit-optimize unavailable"}

    if dim > BO_MAX_DIM:
        return {
            "status": "skipped",
            "message": f"BO restricted to dimension <= {BO_MAX_DIM}; current dimension={dim}"
        }

    bounds = [(LOG_LOWER, LOG_UPPER)] * dim
    counter = EvaluationCounter()
    history = {"evaluation": [], "log_cond": [], "cond": []}
    objective = make_counter_objective(A, counter, history)

    t0 = time.perf_counter()
    try:
        res = gp_minimize(
            objective,
            dimensions=[Real(lo, hi) for lo, hi in bounds],
            n_initial_points=BO_INITIAL_POINTS,
            n_calls=BO_CALLS,
            acq_func="EI",
            random_state=seed,
            noise="gaussian",
        )
        elapsed = time.perf_counter() - t0
        x = np.asarray(res.x, dtype=float)
        final_cond, d1, d2 = result_from_x(A, x)
        return {
            "status": "completed",
            "x": x,
            "d1": d1,
            "d2": d2,
            "fun": final_cond,
            "runtime": elapsed,
            "evaluations": counter.count,
            "iterations": BO_CALLS,
            "history": history,
            "success": True,
            "message": "scikit-optimize GP/EI completed",
        }
    except Exception as exc:
        return {"status": "failed", "message": repr(exc)}

# ---------------------------------------------------------------------------
# Sparse / standard preconditioners
# ---------------------------------------------------------------------------

def is_spd(A) -> bool:
    try:
        B = as_dense(A)
        if not np.allclose(B, B.T, atol=1e-10, rtol=1e-8):
            return False
        np.linalg.cholesky(B)
        return True
    except Exception:
        return False

def make_sparse_for_preconditioner(A):
    if sparse.issparse(A):
        return A.tocsr()
    return csr_matrix(np.asarray(A, dtype=float))

def build_ilu(A):
    S = make_sparse_for_preconditioner(A).tocsc()
    t0 = time.perf_counter()
    try:
        ilu = spilu(S, drop_tol=1e-4, fill_factor=10)
        elapsed = time.perf_counter() - t0
        return ilu, elapsed, "completed"
    except Exception as exc:
        return None, time.perf_counter() - t0, f"failed: {exc}"

def build_amg(A):
    if not HAVE_PYAMG:
        return None, 0.0, "unavailable: pyamg"
    S = make_sparse_for_preconditioner(A)
    t0 = time.perf_counter()
    try:
        ml = pyamg.smoothed_aggregation_solver(S, symmetry="symmetric")
        elapsed = time.perf_counter() - t0
        return ml, elapsed, "completed"
    except Exception as exc:
        return None, time.perf_counter() - t0, f"failed: {exc}"

def incomplete_cholesky_ic0(A, drop_tol=0.0):
    """
    Simple dense IC(0)-style factorization using the lower-triangular
    nonzero pattern of A. Intended for small/moderate SPD benchmarks.
    """
    B = as_dense(A)
    n = B.shape[0]
    B = 0.5 * (B + B.T)
    L = np.zeros_like(B)

    pattern = np.abs(np.tril(B)) > 0
    for i in range(n):
        for j in range(i + 1):
            if not pattern[i, j] and i != j:
                continue
            s = B[i, j]
            kmax = j
            for k in range(kmax):
                if pattern[i, k] and pattern[j, k]:
                    s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 1e-14:
                    return None, "failed: matrix not numerically SPD under IC(0)"
                L[i, j] = math.sqrt(s)
            else:
                if abs(L[j, j]) < 1e-15:
                    return None, "failed: zero diagonal"
                L[i, j] = s / L[j, j]
    return L, "completed"

# ---------------------------------------------------------------------------
# Linear-system experiments
# ---------------------------------------------------------------------------

class IterationCounter:
    def __init__(self):
        self.count = 0

    def callback(self, *args, **kwargs):
        self.count += 1

def gmres_solve(A, b, M=None):
    S = make_sparse_for_preconditioner(A)
    counter = IterationCounter()
    t0 = time.perf_counter()
    try:
        try:
            x, info = gmres(
                S, b, M=M,
                rtol=SOLVER_TOL, atol=0.0,
                restart=SOLVER_RESTART,
                maxiter=SOLVER_MAXITER,
                callback=counter.callback,
                callback_type="pr_norm",
            )
        except TypeError:
            x, info = gmres(
                S, b, M=M,
                tol=SOLVER_TOL,
                restart=SOLVER_RESTART,
                maxiter=SOLVER_MAXITER,
                callback=counter.callback,
            )
        elapsed = time.perf_counter() - t0
        residual = np.linalg.norm(S @ x - b) / max(np.linalg.norm(b), 1e-15)
        status = "converged" if info == 0 else f"not_converged_info_{info}"
        return status, counter.count, elapsed, residual
    except Exception as exc:
        return f"failed: {exc}", counter.count, time.perf_counter() - t0, float("nan")

def cg_solve(A, b, M=None):
    S = make_sparse_for_preconditioner(A)
    counter = IterationCounter()
    t0 = time.perf_counter()
    try:
        try:
            x, info = cg(
                S, b, M=M,
                rtol=SOLVER_TOL, atol=0.0,
                maxiter=SOLVER_MAXITER,
                callback=counter.callback,
            )
        except TypeError:
            x, info = cg(
                S, b, M=M,
                tol=SOLVER_TOL,
                maxiter=SOLVER_MAXITER,
                callback=counter.callback,
            )
        elapsed = time.perf_counter() - t0
        residual = np.linalg.norm(S @ x - b) / max(np.linalg.norm(b), 1e-15)
        status = "converged" if info == 0 else f"not_converged_info_{info}"
        return status, counter.count, elapsed, residual
    except Exception as exc:
        return f"failed: {exc}", counter.count, time.perf_counter() - t0, float("nan")

def diagonal_scaled_system(A, d1, d2, b):
    """
    For B = D1 A D2 and y = D2^{-1} x:
        B y = D1 b
    Then x = D2 y.
    """
    B = apply_scaling(A, d1, d2)
    rhs = d1 * b
    return B, rhs

def run_solver_suite(A, d1=None, d2=None, label="unscaled"):
    n = A.shape[0]
    if n > LINEAR_SOLVE_MAX_N_DENSE and not sparse.issparse(A):
        return []

    if sparse.issparse(A):
        S = A.tocsr()
    else:
        S = csr_matrix(np.asarray(A, dtype=float))

    if d1 is not None and d2 is not None:
        B, rhs = diagonal_scaled_system(S, d1, d2, np.ones(n))
    else:
        B = S
        rhs = np.ones(n)

    results = []

    # Determine appropriate solver.
    spd = is_spd(B)

    # Baseline no preconditioner.
    if spd:
        status, iters, sec, res = cg_solve(B, rhs, M=None)
        results.append({
            "solver": "CG",
            "system": label,
            "status": status,
            "iterations": iters,
            "time_sec": sec,
            "residual": res,
            "preconditioner": "None",
            "preconditioner_build_sec": 0.0,
        })
    else:
        status, iters, sec, res = gmres_solve(B, rhs, M=None)
        results.append({
            "solver": "GMRES",
            "system": label,
            "status": status,
            "iterations": iters,
            "time_sec": sec,
            "residual": res,
            "preconditioner": "None",
            "preconditioner_build_sec": 0.0,
        })

    # ILU + GMRES
    ilu, build_sec, ilu_status = build_ilu(B)
    if ilu is not None:
        M = LinearOperator(B.shape, ilu.solve)
        status, iters, sec, res = gmres_solve(B, rhs, M=M)
        results.append({
            "solver": "GMRES",
            "system": label,
            "status": status,
            "iterations": iters,
            "time_sec": sec,
            "residual": res,
            "preconditioner": "ILU",
            "preconditioner_build_sec": build_sec,
        })
    else:
        results.append({
            "solver": "GMRES",
            "system": label,
            "status": ilu_status,
            "iterations": -1,
            "time_sec": float("nan"),
            "residual": float("nan"),
            "preconditioner": "ILU",
            "preconditioner_b
