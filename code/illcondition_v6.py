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
OUTDIR = Path.cwd() / "illcondition_v6_results"
FIGDIR = OUTDIR / "figures"
CONVDIR = FIGDIR / "convergence"
SCALE_DIR = OUTDIR / "scalings"
RAW_DIR = OUTDIR / "raw_runs"
SCALABILITY_DIR = OUTDIR / "scalability"

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
            "preconditioner_build_sec": build_sec,
        })

    # AMG + CG/GMRES
    ml, build_sec, amg_status = build_amg(B)
    if ml is not None:
        M = ml.aspreconditioner()
        if spd:
            status, iters, sec, res = cg_solve(B, rhs, M=M)
            solver_name = "CG"
        else:
            status, iters, sec, res = gmres_solve(B, rhs, M=M)
            solver_name = "GMRES"
        results.append({
            "solver": solver_name,
            "system": label,
            "status": status,
            "iterations": iters,
            "time_sec": sec,
            "residual": res,
            "preconditioner": "AMG",
            "preconditioner_build_sec": build_sec,
        })
    else:
        results.append({
            "solver": "CG" if spd else "GMRES",
            "system": label,
            "status": amg_status,
            "iterations": -1,
            "time_sec": float("nan"),
            "residual": float("nan"),
            "preconditioner": "AMG",
            "preconditioner_build_sec": build_sec,
        })

    # IC(0) only for SPD and moderate n.
    if spd and n <= 300:
        t0 = time.perf_counter()
        L, ic_status = incomplete_cholesky_ic0(B)
        build_sec = time.perf_counter() - t0
        if L is not None:
            def ic_solve(v):
                y = solve_triangular(L, v, lower=True, check_finite=False)
                return solve_triangular(L.T, y, lower=False, check_finite=False)
            M = LinearOperator(B.shape, ic_solve)
            status, iters, sec, res = cg_solve(B, rhs, M=M)
            results.append({
                "solver": "CG",
                "system": label,
                "status": status,
                "iterations": iters,
                "time_sec": sec,
                "residual": res,
                "preconditioner": "ICHOL_IC0",
                "preconditioner_build_sec": build_sec,
            })
        else:
            results.append({
                "solver": "CG",
                "system": label,
                "status": ic_status,
                "iterations": -1,
                "time_sec": float("nan"),
                "residual": float("nan"),
                "preconditioner": "ICHOL_IC0",
                "preconditioner_build_sec": build_sec,
            })

    return results

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def reduction_factor(initial, final):
    if not np.isfinite(initial) or not np.isfinite(final) or final <= 0:
        return float("nan")
    return initial / final

def make_method_row(name, method, A, seed, status, initial_cond, final_cond,
                    runtime, evaluations, iterations, notes="", **kwargs):
    return MethodResult(
        matrix=name,
        method=method,
        n=A.shape[0],
        status=status,
        seed=seed,
        initial_cond=initial_cond,
        final_cond=final_cond,
        reduction_factor=reduction_factor(initial_cond, final_cond),
        log_objective=math.log(max(final_cond, 1.0)) if np.isfinite(final_cond) else float("nan"),
        runtime_sec=float(runtime),
        objective_evaluations=int(evaluations),
        iterations=int(iterations),
        notes=notes,
        **kwargs,
    )

def save_scaling(name, method, d1, d2):
    if d1 is None or d2 is None:
        return
    np.savez(
        SCALE_DIR / f"{name}__{method}.npz",
        d1=np.asarray(d1),
        d2=np.asarray(d2),
    )

def plot_history(name, method, history):
    if not history:
        return
    try:
        if "cond" in history and len(history["cond"]) >= 2:
            x = history.get("evaluation", history.get("generation", range(len(history["cond"]))))
            y = np.maximum(np.asarray(history["cond"], dtype=float), 1e-300)
            plt.figure(figsize=(8.5, 5.5))
            plt.plot(x, y, linewidth=2)
            plt.yscale("log")
            plt.xlabel("Objective evaluations / generations")
            plt.ylabel(r"$\kappa_2$")
            plt.title(f"{method} convergence — {name}")
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            safe = f"{name}__{method}".replace("/", "_").replace(" ", "_")
            plt.savefig(CONVDIR / f"{safe}.png", dpi=220)
            plt.close()
    except Exception:
        pass

def run_one_matrix(name, A, family, index):
    initial_cond = safe_cond(A)
    n = A.shape[0]
    dense = not sparse.issparse(A)

    print(f"\n{'='*78}")
    print(f"{name} | family={family} | n={n} | initial kappa={initial_cond:.6e}")
    print(f"{'='*78}")

    rows = []
    scaling_cache = {}

    # No scaling baseline
    rows.append(make_method_row(
        name, "Unscaled", A, SEED, "baseline", initial_cond, initial_cond,
        0.0, 1, 0, "Original matrix"
    ))

    # Ruiz
    t0 = time.perf_counter()
    r, c, B_ruiz, ruiz_iters = ruiz_equilibration(A)
    runtime = time.perf_counter() - t0
    ruiz_cond = safe_cond(B_ruiz)
    save_scaling(name, "Ruiz", r, c)
    scaling_cache["Ruiz"] = (r, c)
    rows.append(make_method_row(
        name, "Ruiz", A, SEED, "completed", initial_cond, ruiz_cond,
        runtime, 0, ruiz_iters,
        f"Ruiz iterations={ruiz_iters}"
    ))
    print(f"Ruiz:       kappa={ruiz_cond:.6e}  time={runtime:.4f}s")

    # L-BFGS and Hybrid
    if n <= GLOBAL_OPT_MAX_N:
        lb = run_lbfgs(A)
        save_scaling(name, "L-BFGS", lb["d1"], lb["d2"])
        scaling_cache["L-BFGS"] = (lb["d1"], lb["d2"])
        rows.append(make_method_row(
            name, "L-BFGS", A, SEED,
            "completed" if lb["success"] else "completed_with_warning",
            initial_cond, lb["fun"], lb["runtime"],
            lb["evaluations"], lb["iterations"], lb["message"]
        ))
        plot_history(name, "L-BFGS", lb["history"])
        print(f"L-BFGS:     kappa={lb['fun']:.6e}  time={lb['runtime']:.4f}s evals={lb['evaluations']}")

        hy = run_ruiz_lbfgs(A)
        save_scaling(name, "Ruiz-LBFGS", hy["d1"], hy["d2"])
        scaling_cache["Ruiz-LBFGS"] = (hy["d1"], hy["d2"])
        rows.append(make_method_row(
            name, "Ruiz-LBFGS", A, SEED,
            "completed" if hy["success"] else "completed_with_warning",
            initial_cond, hy["fun"], hy["runtime"],
            hy["evaluations"], hy["iterations"],
            f"Ruiz iterations={hy['ruiz_iterations']}; {hy['message']}"
        ))
        plot_history(name, "Ruiz-LBFGS", hy["history"])
        print(f"Hybrid:     kappa={hy['fun']:.6e}  time={hy['runtime']:.4f}s evals={hy['evaluations']}")
    else:
        print(f"L-BFGS/Hybrid skipped for n>{GLOBAL_OPT_MAX_N}")

    # Stochastic methods
    if n <= GLOBAL_OPT_MAX_N:
        for rep in range(N_STOCHASTIC_REPEATS):
            seed = SEED + 1000 * index + rep

            sa = run_sa(A, seed)
            if sa.get("status", "completed") not in ("unavailable", "skipped"):
                save_scaling(name, f"SA_run{rep+1}", sa["d1"], sa["d2"])
                rows.append(make_method_row(
                    name, "SA", A, seed, "completed",
                    initial_cond, sa["fun"], sa["runtime"],
                    sa["evaluations"], sa["iterations"],
                    f"replicate={rep+1}; {sa['message']}"
                ))
                if rep == 0:
                    plot_history(name, "SA", sa["history"])

            ga = run_ga(A, seed + 500)
            save_scaling(name, f"GA_run{rep+1}", ga["d1"], ga["d2"])
            rows.append(make_method_row(
                name, "GA", A, seed + 500, "completed",
                initial_cond, ga["fun"], ga["runtime"],
                ga["evaluations"], ga["iterations"],
                f"replicate={rep+1}"
            ))
            if rep == 0:
                plot_history(name, "GA", ga["history"])

        # BO only once per matrix; it is already expensive and intentionally
        # dimension-limited.
        bo = run_bo(A, SEED + 9000 + index)
        if bo.get("status") == "completed":
            save_scaling(name, "BO", bo["d1"], bo["d2"])
            scaling_cache["BO"] = (bo["d1"], bo["d2"])
            rows.append(make_method_row(
                name, "BO", A, SEED + 9000 + index, "completed",
                initial_cond, bo["fun"], bo["runtime"],
                bo["evaluations"], bo["iterations"], bo["message"]
            ))
            plot_history(name, "BO", bo["history"])
            print(f"BO:         kappa={bo['fun']:.6e}  time={bo['runtime']:.4f}s")
        else:
            rows.append(make_method_row(
                name, "BO", A, SEED + 9000 + index,
                bo.get("status", "skipped"), initial_cond, float("nan"),
                0.0, 0, 0, bo.get("message", "")
            ))

        # Keep best deterministic/stochastic scaling for actual solve tests.
        if "Ruiz-LBFGS" in scaling_cache:
            preferred = "Ruiz-LBFGS"
        elif "L-BFGS" in scaling_cache:
            preferred = "L-BFGS"
        else:
            preferred = "Ruiz"

        # Linear solve tests
        solve_rows = []
        base_solver_rows = run_solver_suite(A, label="Unscaled")
        for sr in base_solver_rows:
            sr.update({"matrix": name, "family": family, "scaling_method": "Unscaled"})
            solve_rows.append(sr)

        if preferred in scaling_cache:
            d1, d2 = scaling_cache[preferred]
            scaled_solver_rows = run_solver_suite(
                A, d1=d1, d2=d2, label=preferred
            )
            for sr in scaled_solver_rows:
                sr.update({"matrix": name, "family": family, "scaling_method": preferred})
                solve_rows.append(sr)

        pd.DataFrame(solve_rows).to_csv(
            RAW_DIR / f"{name}__solver_results.csv", index=False
        )
    else:
        solve_rows = []

    # For sparse / large systems, always perform standard preconditioner tests.
    if sparse.issparse(A) and n <= AMG_MAX_N:
        sparse_solver_rows = run_solver_suite(A, label="SparseOriginal")
        for sr in sparse_solver_rows:
            sr.update({"matrix": name, "family": family, "scaling_method": "Unscaled"})
        pd.DataFrame(sparse_solver_rows).to_csv(
            RAW_DIR / f"{name}__sparse_solver_results.csv", index=False
        )

    return rows

# ---------------------------------------------------------------------------
# Statistical summaries
# ---------------------------------------------------------------------------

def summarize_methods(results_df):
    if results_df.empty:
        return pd.DataFrame()

    work = results_df.copy()
    stochastic = work[work["method"].isin(["SA", "GA"])].copy()

    summaries = []
    for method in ["Ruiz", "L-BFGS", "Ruiz-LBFGS", "SA", "GA", "BO"]:
        d = work[work["method"] == method]
        if d.empty:
            continue
        summaries.append({
            "method": method,
            "matrices": int(d["matrix"].nunique()),
            "mean_final_cond": d["final_cond"].mean(),
            "median_final_cond": d["final_cond"].median(),
            "mean_reduction_factor": d["reduction_factor"].mean(),
            "median_reduction_factor": d["reduction_factor"].median(),
            "mean_runtime_sec": d["runtime_sec"].mean(),
            "median_runtime_sec": d["runtime_sec"].median(),
            "mean_objective_evaluations": d["objective_evaluations"].mean(),
            "median_objective_evaluations": d["objective_evaluations"].median(),
        })
    return pd.DataFrame(summaries)

def summarize_stochastic(results_df):
    rows = []
    for method in ["SA", "GA"]:
        d = results_df[results_df["method"] == method]
        if d.empty:
            continue
        for matrix, g in d.groupby("matrix"):
            vals = g["final_cond"].replace([np.inf, -np.inf], np.nan).dropna()
            times = g["runtime_sec"].replace([np.inf, -np.inf], np.nan).dropna()
            evals = g["objective_evaluations"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) == 0:
                continue
            rows.append({
                "matrix": matrix,
                "method": method,
                "runs": len(vals),
                "best_cond": vals.min(),
                "median_cond": vals.median(),
                "mean_cond": vals.mean(),
                "std_cond": vals.std(ddof=1) if len(vals) > 1 else 0.0,
                "worst_cond": vals.max(),
                "mean_runtime_sec": times.mean() if len(times) else np.nan,
                "std_runtime_sec": times.std(ddof=1) if len(times) > 1 else 0.0,
                "mean_evaluations": evals.mean() if len(evals) else np.nan,
            })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_condition_summary(df):
    for matrix in df["matrix"].unique():
        d = df[df["matrix"] == matrix].copy()
        # Aggregate stochastic methods.
        vals = {}
        for method, g in d.groupby("method"):
            if method in ("SA", "GA"):
                vals[method] = g["final_cond"].median()
            else:
                vals[method] = g["final_cond"].iloc[0]

        labels = list(vals.keys())
        y = [max(vals[k], 1e-300) if np.isfinite(vals[k]) else np.nan for k in labels]

        plt.figure(figsize=(10, 5.5))
        plt.bar(labels, y)
        plt.yscale("log")
        plt.ylabel(r"Spectral condition number $\kappa_2$")
        plt.title(f"Conditioning comparison — {matrix}")
        plt.xticks(rotation=35, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        safe = matrix.replace("/", "_").replace(" ", "_")
        plt.savefig(FIGDIR / f"condition__{safe}.png", dpi=220)
        plt.close()

def plot_runtime_summary(df):
    agg = df.groupby("method")["runtime_sec"].median().sort_values()
    if agg.empty:
        return
    plt.figure(figsize=(9, 5.5))
    plt.bar(agg.index, agg.values)
    plt.yscale("log")
    plt.ylabel("Median runtime (s)")
    plt.title("Computational cost by method")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGDIR / "runtime_summary.png", dpi=220)
    plt.close()

# ---------------------------------------------------------------------------
# Scalability
# ---------------------------------------------------------------------------

def scalability_experiment():
    rows = []

    # Dense: compare inexpensive deterministic methods and Hybrid.
    for n in SCALABILITY_DENSE_SIZES:
        A = hilbert(n)
        initial = safe_cond(A)

        t0 = time.perf_counter()
        r, c, B, iters = ruiz_equilibration(A)
        rt = time.perf_counter() - t0
        rows.append({
            "family": "Hilbert",
            "n": n,
            "method": "Ruiz",
            "initial_cond": initial,
            "final_cond": safe_cond(B),
            "runtime_sec": rt,
            "evaluations": 0,
            "iterations": iters,
        })

        if n <= GLOBAL_OPT_MAX_N:
            lb = run_lbfgs(A)
            rows.append({
                "family": "Hilbert",
                "n": n,
                "method": "L-BFGS",
                "initial_cond": initial,
                "final_cond": lb["fun"],
                "runtime_sec": lb["runtime"],
                "evaluations": lb["evaluations"],
                "iterations": lb["iterations"],
            })
            hy = run_ruiz_lbfgs(A)
            rows.append({
                "family": "Hilbert",
                "n": n,
                "method": "Ruiz-LBFGS",
                "initial_cond": initial,
                "final_cond": hy["fun"],
                "runtime_sec": hy["runtime"],
                "evaluations": hy["evaluations"],
                "iterations": hy["iterations"],
            })

    # Sparse: Ruiz only plus standard preconditioner build costs.
    for n in SCALABILITY_SPARSE_SIZES:
        A = finite_difference(n)
        initial = safe_cond(A)

        t0 = time.perf_counter()
        r, c, B, iters = ruiz_equilibration(A)
        rt = time.perf_counter() - t0
        rows.append({
            "family": "FiniteDifference",
            "n": n,
            "method": "Ruiz",
            "initial_cond": initial,
            "final_cond": safe_cond(B),
            "runtime_sec": rt,
            "evaluations": 0,
            "iterations": iters,
        })

        ilu, build_sec, status = build_ilu(A)
        rows.append({
            "family": "FiniteDifference",
            "n": n,
            "method": "ILU_build",
            "initial_cond": initial,
            "final_cond": initial,
            "runtime_sec": build_sec,
            "evaluations": 0,
            "iterations": 0,
            "status": status,
        })

        ml, build_sec, status = build_amg(A)
        rows.append({
            "family": "FiniteDifference",
            "n": n,
            "method": "AMG_build",
            "initial_cond": initial,
            "final_cond": initial,
            "runtime_sec": build_sec,
            "evaluations": 0,
            "iterations": 0,
            "status": status,
        })

    df = pd.DataFrame(rows)
    df.to_csv(SCALABILITY_DIR / "scalability_results.csv", index=False)

    # Runtime scaling plot.
    for family in df["family"].unique():
        d = df[df["family"] == family]
        plt.figure(figsize=(8.5, 5.5))
        for method, g in d.groupby("method"):
            g = g.sort_values("n")
            plt.plot(g["n"], g["runtime_sec"], marker="o", label=method)
        plt.yscale("log")
        plt.xlabel("Matrix dimension n")
        plt.ylabel("Runtime / build time (s)")
        plt.title(f"Scalability — {family}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(
            SCALABILITY_DIR / f"runtime_scalability__{family}.png",
            dpi=220
        )
        plt.close()

    return df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*78)
    print("ILL-CONDITIONED MATRIX DIAGONAL SCALING — VERSION 6")
    print("="*78)
    print(f"Python: {platform.python_version()}")
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {__import__('scipy').__version__}")
    print(f"Output: {OUTDIR}")
    print(f"Seed: {SEED}")
    print(f"Optional packages: sklearn={HAVE_SKLEARN}, skopt={HAVE_SKOPT}, pyamg={HAVE_PYAMG}")
    print("="*78)

    # Save configuration and environment.
    config = {
        "version": "v6",
        "seed": SEED,
        "log_bounds": [LOG_LOWER, LOG_UPPER],
        "lbfgs_maxiter": LBFGS_MAXITER,
        "lbfgs_hybrid_maxiter": LBFGS_HYBRID_MAXITER,
        "ruiz_maxiter": RUIZ_MAXITER,
        "sa_maxiter": SA_MAXITER,
        "ga_popsize": GA_POPSIZE,
        "ga_gens": GA_GENS,
        "stochastic_repeats": N_STOCHASTIC_REPEATS,
        "bo_max_dim": BO_MAX_DIM,
        "bo_calls": BO_CALLS,
        "global_opt_max_n": GLOBAL_OPT_MAX_N,
        "solver_tol": SOLVER_TOL,
        "solver_maxiter": SOLVER_MAXITER,
        "have_sklearn": HAVE_SKLEARN,
        "have_skopt": HAVE_SKOPT,
        "have_pyamg": HAVE_PYAMG,
    }
    with open(OUTDIR / "configuration.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    benchmarks = build_benchmarks()

    all_rows = []
    start_all = time.perf_counter()

    for idx, (name, A, family) in enumerate(benchmarks):
        try:
            rows = run_one_matrix(name, A, family, idx)
            all_rows.extend([asdict(r) for r in rows])
        except Exception as exc:
            print(f"[ERROR] Benchmark {name} failed: {exc}")
            all_rows.append({
                "matrix": name,
                "method": "BENCHMARK_ERROR",
                "n": A.shape[0],
                "status": "failed",
                "seed": SEED,
                "initial_cond": safe_cond(A),
                "final_cond": np.nan,
                "reduction_factor": np.nan,
                "log_objective": np.nan,
                "runtime_sec": np.nan,
                "objective_evaluations": 0,
                "iterations": 0,
                "solver_applicable": False,
                "solver": "",
                "solver_status": "",
                "solver_iterations": -1,
                "solver_time_sec": np.nan,
                "solver_residual": np.nan,
                "preconditioner_build_sec": np.nan,
                "notes": repr(exc),
            })

    total_time = time.perf_counter() - start_all

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUTDIR / "all_method_results.csv", index=False)

    if HAVE_OPENPYXL:
        try:
            with pd.ExcelWriter(OUTDIR / "all_results.xlsx", engine="openpyxl") as writer:
                results_df.to_excel(writer, sheet_name="method_results", index=False)
                summarize_methods(results_df).to_excel(writer, sheet_name="method_summary", index=False)
                summarize_stochastic(results_df).to_excel(writer, sheet_name="stochastic", index=False)
        except Exception as exc:
            print(f"[WARN] Excel export failed: {exc}")

    method_summary = summarize_methods(results_df)
    stochastic_summary = summarize_stochastic(results_df)

    method_summary.to_csv(OUTDIR / "method_summary.csv", index=False)
    stochastic_summary.to_csv(OUTDIR / "stochastic_summary.csv", index=False)

    # Condition/runtime plots
    plot_condition_summary(results_df[results_df["status"].isin(["baseline", "completed", "completed_with_warning"])])
    plot_runtime_summary(results_df[results_df["status"].isin(["completed", "completed_with_warning"])])

    # Collect all solver outputs.
    solver_files = list(RAW_DIR.glob("*solver_results.csv")) + list(RAW_DIR.glob("*sparse_solver_results.csv"))
    solver_frames = []
    for f in solver_files:
        try:
            solver_frames.append(pd.read_csv(f))
        except Exception:
            pass
    if solver_frames:
        solver_df = pd.concat(solver_frames, ignore_index=True)
    else:
        solver_df = pd.DataFrame()
    solver_df.to_csv(OUTDIR / "solver_results_all.csv", index=False)

    # Scalability
    try:
        scalability_df = scalability_experiment()
    except Exception as exc:
        print(f"[WARN] Scalability experiment failed: {exc}")
        scalability_df = pd.DataFrame()

    # Compact final report.
    report = {
        "version": "v6",
        "total_runtime_sec": total_time,
        "benchmarks": len(benchmarks),
        "method_rows": len(results_df),
        "solver_rows": len(solver_df),
        "stochastic_repeats": N_STOCHASTIC_REPEATS,
        "pyamg_available": HAVE_PYAMG,
        "skopt_available": HAVE_SKOPT,
        "sklearn_available": HAVE_SKLEARN,
        "output_directory": str(OUTDIR.resolve()),
        "files": [
            "all_method_results.csv",
            "method_summary.csv",
            "stochastic_summary.csv",
            "solver_results_all.csv",
            "configuration.json",
            "all_results.xlsx" if HAVE_OPENPYXL else "Excel export unavailable",
        ],
    }
    with open(OUTDIR / "run_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*78)
    print("V6 COMPLETED")
    print("="*78)
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Results directory: {OUTDIR.resolve()}")
    print("\nMethod summary:")
    if not method_summary.empty:
        print(method_summary.to_string(index=False))
    print("\nStochastic summary:")
    if not stochastic_summary.empty:
        print(stochastic_summary.to_string(index=False))
    print("\nKey output files:")
    print(f"  {OUTDIR / 'all_method_results.csv'}")
    print(f"  {OUTDIR / 'method_summary.csv'}")
    print(f"  {OUTDIR / 'stochastic_summary.csv'}")
    print(f"  {OUTDIR / 'solver_results_all.csv'}")
    print(f"  {OUTDIR / 'configuration.json'}")
    if HAVE_OPENPYXL:
        print(f"  {OUTDIR / 'all_results.xlsx'}")
    print("="*78)

if __name__ == "__main__":
    main()
