#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
illcondition_v6_1.py
Research-grade benchmark for optimized two-sided diagonal scaling.

V6.1 scientific corrections:
1. IC(0) is used only for sparse SPD matrices. Dense SPD matrices use
   a dense Cholesky preconditioner only when explicitly requested.
2. AMG uses symmetry="symmetric" only for symmetric/SPD systems and
   symmetry="nonsymmetric" for nonsymmetric systems.
3. Solver experiments report BOTH scaled-system residual and residual
   in the original system ||Ax-b||/||b||.
4. Solver timing uses repeated trials and reports median/IQR.
5. Total solver cost = preconditioner setup + median solve time.
6. Summary statistics exclude skipped/failed runs.
7. L-BFGS/Hybrid convergence histories use callback-aligned iterations,
   not objective-evaluation counts.
8. Exceptions are recorded instead of silently swallowed.
9. A reproducibility configuration and run report are written.
10. The script does not install packages silently. Use requirements.txt.
11. Two-sided scaling is described as a transformed-system strategy;
    no unsupported claim that it universally replaces ILU/AMG is made.

The script creates:
illcondition_v6_results/
    figures/
        convergence/
    histograms/
    scalings/
    raw_runs/
    scalability/
    tables/
    configuration.json
    run_report.json
    all_method_results.csv
    method_summary.csv
    stochastic_summary.csv
    solver_results_all.csv
    scalability_results.csv
    all_results.xlsx
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import optimize, sparse
from scipy.linalg import solve_triangular
from scipy.sparse import diags
from scipy.sparse.linalg import cg, gmres, LinearOperator, spilu

try:
    import pyamg
    HAVE_PYAMG = True
except Exception:
    HAVE_PYAMG = False

try:
    from skopt import gp_minimize
    from skopt.space import Real
    HAVE_SKOPT = True
except Exception:
    HAVE_SKOPT = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 20260813

OUTDIR = Path.cwd() / "illcondition_v6_results"
FIGDIR = OUTDIR / "figures"
CONVDIR = FIGDIR / "convergence"
HISTDIR = OUTDIR / "histograms"
SCALE_DIR = OUTDIR / "scalings"
RAW_DIR = OUTDIR / "raw_runs"
SCALABILITY_DIR = OUTDIR / "scalability"
TABLEDIR = OUTDIR / "tables"

for p in [OUTDIR, FIGDIR, CONVDIR, HISTDIR, SCALE_DIR, RAW_DIR,
          SCALABILITY_DIR, TABLEDIR]:
    p.mkdir(parents=True, exist_ok=True)

LOG_LOWER = -5.0
LOG_UPPER = 5.0

RUIZ_MAXITER = 60
RUIZ_TOL = 1e-7

LBFGS_MAXITER = 120
LBFGS_HYBRID_MAXITER = 70
LBFGS_FTOL = 1e-10
LBFGS_GTOL = 1e-6

N_STOCHASTIC_REPEATS = 10
SA_MAXITER = 70
GA_POPSIZE = 18
GA_GENS = 30
GA_ELITE = 2
GA_MUTATION_RATE = 0.15
GA_MUTATION_SCALE = 0.25

BO_MAX_DIM = 12
BO_INITIAL_POINTS = 6
BO_CALLS = 18

SOLVER_TOL = 1e-8
SOLVER_MAXITER = 1000
SOLVER_RESTART = 50
SOLVER_TRIALS = 5

GLOBAL_OPT_MAX_N = 15
LINEAR_SOLVE_MAX_N_DENSE = 100
AMG_MAX_N = 5000

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MethodResult:
    matrix: str
    method: str
    n: int
    family: str
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
    solver_time_sec: float = np.nan
    solver_time_iqr_sec: float = np.nan
    solver_residual_scaled: float = np.nan
    solver_residual_original: float = np.nan
    preconditioner_build_sec: float = np.nan
    total_solver_cost_sec: float = np.nan
    notes: str = ""


class EvaluationCounter:
    def __init__(self):
        self.count = 0

    def inc(self):
        self.count += 1


# =============================================================================
# MATRIX GENERATORS
# =============================================================================

def hilbert(n):
    i = np.arange(n, dtype=float)[:, None]
    j = np.arange(n, dtype=float)[None, :]
    return 1.0 / (i + j + 1.0)


def vandermonde(n):
    x = np.linspace(0.15, 0.95, n)
    return np.vander(x, N=n, increasing=True)


def cauchy_matrix(n):
    x = np.arange(1, n + 1, dtype=float)
    y = x + 0.37
    return 1.0 / (x[:, None] + y[None, :])


def near_dependent4():
    # Explicitly named NearDependent4: this is NOT claimed to be
    # the canonical Rump matrix.
    return np.array([
        [1.0, 2.0, -1.0, 0.5],
        [2.0, 4.0, -2.0, 1.0],
        [3.0, 6.0, -2.999999, 1.5],
        [4.0, 8.0, -4.0, 2.000001],
    ], dtype=float)


def finite_difference(n, shift=0.25):
    main = (2.0 + shift) * np.ones(n)
    off = -np.ones(n - 1)
    return sparse.diags([off, main, off], [-1, 0, 1], format="csr")


def sparse_nonsymmetric(n, seed=SEED):
    rng = np.random.default_rng(seed)
    main = 4.0 * np.ones(n)
    lower = -0.8 * np.ones(n - 1)
    upper = -1.2 * np.ones(n - 1)
    A = sparse.diags([lower, main, upper], [-1, 0, 1], format="lil")
    for _ in range(max(1, n // 20)):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i != j:
            A[i, j] += float(rng.uniform(-0.2, 0.2))
    return A.tocsr()


def random_spectrum(n, cond_target=1e8, seed=SEED):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    s = np.logspace(0.0, math.log10(cond_target), n)
    return U @ np.diag(s) @ V.T


def spd_random(n, cond_target=1e5, seed=SEED):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.logspace(0.0, math.log10(cond_target), n)
    return Q @ np.diag(eigs) @ Q.T


def original_A1():
    return np.array([[1., 1., 1.],
                     [1., 2., 3.],
                     [1., 3., 6.]], dtype=float)


def original_A2():
    return np.array([
        [0.0927, 17.08, 0.31, 12.75],
        [1.78, 54.02, 1.5, 14.77],
        [0.346, 0.068, 0.263, 0.023],
        [1.375, 45.15, 0.051, 1.431]
    ], dtype=float)


def build_benchmarks():
    return [
        ("A1_3x3", original_A1(), "original"),
        ("A2_4x4", original_A2(), "original"),
        ("Hilbert10", hilbert(10), "dense_structured"),
        ("Hilbert15", hilbert(15), "dense_structured"),
        ("Vandermonde10", vandermonde(10), "dense_structured"),
        ("Cauchy8", cauchy_matrix(8), "dense_structured"),
        ("NearDependent4", near_dependent4(), "dense_structured"),
        ("FiniteDiff50", finite_difference(50), "sparse_spd"),
        ("SparseNonsym100", sparse_nonsymmetric(100), "sparse_nonsym"),
        ("RandomSpectrum12", random_spectrum(12, 1e8, SEED), "random_dense"),
        ("RandomSPD12", spd_random(12, 1e6, SEED), "random_spd"),
    ]


# =============================================================================
# BASIC NUMERICAL UTILITIES
# =============================================================================

def as_dense(A):
    return A.toarray() if sparse.issparse(A) else np.asarray(A, dtype=float)


def is_symmetric(A, tol=1e-10):
    B = as_dense(A)
    scale = max(1.0, np.linalg.norm(B, ord=np.inf))
    return bool(np.linalg.norm(B - B.T, ord=np.inf) <= tol * scale)


def is_spd(A):
    if not is_symmetric(A):
        return False
    try:
        np.linalg.cholesky(as_dense(A))
        return True
    except np.linalg.LinAlgError:
        return False


def safe_cond(A):
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


def normalize_logs(x, n):
    u = np.asarray(x[:n], dtype=float).copy()
    v = np.asarray(x[n:], dtype=float).copy()
    u -= np.mean(u)
    v -= np.mean(v)
    u = np.clip(u, LOG_LOWER, LOG_UPPER)
    v = np.clip(v, LOG_LOWER, LOG_UPPER)
    return np.exp(u), np.exp(v)


def scales_to_logs(d1, d2):
    d1 = np.maximum(np.asarray(d1, dtype=float), 1e-15)
    d2 = np.maximum(np.asarray(d2, dtype=float), 1e-15)
    u = np.log(d1)
    v = np.log(d2)
    u -= np.mean(u)
    v -= np.mean(v)
    return np.r_[u, v]


def make_objective(A, counter, history=None):
    n = A.shape[0]

    def objective(x):
        counter.inc()
        d1, d2 = normalize_logs(x, n)
        B = apply_scaling(A, d1, d2)
        cond = safe_cond(B)
        val = math.log(max(cond, 1.0))
        if history is not None:
            history["eval_objective"].append(val)
        return val

    return objective


def result_from_scaling(name, method, A, family, d1, d2, status,
                        seed, runtime, evaluations, iterations, notes=""):
    initial = safe_cond(A)
    B = apply_scaling(A, d1, d2)
    final = safe_cond(B)
    return MethodResult(
        matrix=name,
        method=method,
        n=A.shape[0],
        family=family,
        status=status,
        seed=seed,
        initial_cond=initial,
        final_cond=final,
        reduction_factor=(initial / final if np.isfinite(final) and final > 0 else np.nan),
        log_objective=(math.log(max(final, 1.0)) if np.isfinite(final) else np.inf),
        runtime_sec=runtime,
        objective_evaluations=evaluations,
        iterations=iterations,
        notes=notes,
    )


# =============================================================================
# SCALING METHODS
# =============================================================================

def unscaled(A, name, family):
    n = A.shape[0]
    return result_from_scaling(
        name, "Unscaled", A, family,
        np.ones(n), np.ones(n), "completed", SEED, 0.0, 0, 0,
        "No scaling."
    ), np.ones(n), np.ones(n), {"iterations": [], "objective": []}


def ruiz_scale(A):
    B = A.copy()
    n = A.shape[0]
    d1 = np.ones(n)
    d2 = np.ones(n)
    history = []

    for k in range(RUIZ_MAXITER):
        if sparse.issparse(B):
            row_norm = np.asarray(np.sqrt(B.multiply(B).sum(axis=1))).ravel()
            col_norm = np.asarray(np.sqrt(B.multiply(B).sum(axis=0))).ravel()
        else:
            row_norm = np.linalg.norm(B, axis=1)
            col_norm = np.linalg.norm(B, axis=0)

        row_norm = np.maximum(row_norm, 1e-15)
        col_norm = np.maximum(col_norm, 1e-15)

        r = 1.0 / np.sqrt(row_norm)
        c = 1.0 / np.sqrt(col_norm)

        d1 *= r
        d2 *= c
        B = apply_scaling(B, r, c)

        current = safe_cond(B)
        history.append(current)

        if k > 0 and abs(history[-1] - history[-2]) <= RUIZ_TOL * max(1.0, history[-2]):
            break

    return d1, d2, history


def run_ruiz(name, A, family):
    t0 = time.perf_counter()
    d1, d2, hist = ruiz_scale(A)
    elapsed = time.perf_counter() - t0
    r = result_from_scaling(
        name, "Ruiz", A, family, d1, d2, "completed",
        SEED, elapsed, 0, len(hist),
        "Iterative Ruiz equilibration."
    )
    return r, d1, d2, {"iterations": hist, "objective": []}


def run_lbfgs(name, A, family, x0=None, maxiter=LBFGS_MAXITER,
              method_name="L-BFGS"):
    n = A.shape[0]
    counter = EvaluationCounter()
    history = {"eval_objective": [], "iteration_objective": [],
               "iteration_cond": []}

    if x0 is None:
        x0 = np.zeros(2 * n)

    objective = make_objective(A, counter, history)

    def callback(xk):
        d1, d2 = normalize_logs(xk, n)
        cond = safe_cond(apply_scaling(A, d1, d2))
        history["iteration_objective"].append(math.log(max(cond, 1.0)))
        history["iteration_cond"].append(cond)

    t0 = time.perf_counter()
    try:
        res = optimize.minimize(
            objective,
            np.asarray(x0, dtype=float),
            method="L-BFGS-B",
            bounds=[(LOG_LOWER, LOG_UPPER)] * (2 * n),
            callback=callback,
            options={
                "maxiter": maxiter,
                "ftol": LBFGS_FTOL,
                "gtol": LBFGS_GTOL,
                "maxls": 40,
            },
        )
        elapsed = time.perf_counter() - t0
        d1, d2 = normalize_logs(res.x, n)
        status = "completed" if np.isfinite(res.fun) else "failed"
        notes = str(res.message)
        r = result_from_scaling(
            name, method_name, A, family, d1, d2, status,
            SEED, elapsed, counter.count,
            len(history["iteration_cond"]), notes
        )
        return r, d1, d2, history
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        d1, d2 = normalize_logs(x0, n)
        r = result_from_scaling(
            name, method_name, A, family, d1, d2, "failed",
            SEED, elapsed, counter.count, len(history["iteration_cond"]),
            f"{type(exc).__name__}: {exc}"
        )
        return r, d1, d2, history


def run_hybrid(name, A, family):
    t0 = time.perf_counter()
    d1r, d2r, ruiz_hist = ruiz_scale(A)
    x0 = scales_to_logs(d1r, d2r)

    r, d1, d2, hist = run_lbfgs(
        name, A, family, x0=x0,
        maxiter=LBFGS_HYBRID_MAXITER,
        method_name="Ruiz-LBFGS"
    )
    # Include Ruiz time in the hybrid runtime because it is part of the method.
    r.runtime_sec = time.perf_counter() - t0
    r.notes = "Ruiz initialization followed by L-BFGS-B."
    hist["ruiz_condition"] = ruiz_hist
    return r, d1, d2, hist


def run_sa(name, A, family, seed):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    counter = EvaluationCounter()
    history = {"iteration_objective": [], "iteration_cond": []}

    def obj(x):
        counter.inc()
        d1, d2 = normalize_logs(x, n)
        return math.log(max(safe_cond(apply_scaling(A, d1, d2)), 1.0))

    x = rng.uniform(-1, 1, 2 * n)
    best_x = x.copy()
    best_val = obj(x)

    t0 = time.perf_counter()
    for k in range(SA_MAXITER):
        temp = max(1e-4, 1.0 - k / SA_MAXITER)
        proposal = np.clip(x + rng.normal(0, 0.35 * temp, 2 * n),
                           LOG_LOWER, LOG_UPPER)
        pv = obj(proposal)
        delta = pv - best_val
        if pv < best_val or rng.random() < math.exp(-delta / max(temp, 1e-6)):
            x = proposal
            if pv < best_val:
                best_x = proposal.copy()
                best_val = pv
        history["iteration_objective"].append(best_val)
        history["iteration_cond"].append(math.exp(min(best_val, 700)))

    elapsed = time.perf_counter() - t0
    d1, d2 = normalize_logs(best_x, n)
    r = result_from_scaling(
        name, "SA", A, family, d1, d2, "completed",
        seed, elapsed, counter.count, SA_MAXITER,
        "Simulated annealing; one reproducible run."
    )
    return r, d1, d2, history


def run_ga(name, A, family, seed):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    counter = EvaluationCounter()
    history = {"iteration_objective": [], "iteration_cond": []}

    def obj(x):
        counter.inc()
        d1, d2 = normalize_logs(x, n)
        return math.log(max(safe_cond(apply_scaling(A, d1, d2)), 1.0))

    pop = rng.uniform(-1, 1, size=(GA_POPSIZE, 2 * n))
    vals = np.array([obj(x) for x in pop])
    t0 = time.perf_counter()

    for g in range(GA_GENS):
        order = np.argsort(vals)
        pop = pop[order]
        vals = vals[order]
        elites = pop[:GA_ELITE].copy()

        children = []
        while len(children) < GA_POPSIZE - GA_ELITE:
            a, b = rng.integers(0, max(2, GA_POPSIZE // 2), size=2)
            p1, p2 = pop[a], pop[b]
            alpha = rng.random(2 * n)
            child = alpha * p1 + (1 - alpha) * p2
            mask = rng.random(2 * n) < GA_MUTATION_RATE
            child[mask] += rng.normal(0, GA_MUTATION_SCALE, np.sum(mask))
            children.append(np.clip(child, LOG_LOWER, LOG_UPPER))

        pop = np.vstack([elites, np.asarray(children)])
        vals = np.array([obj(x) for x in pop])

        history["iteration_objective"].append(float(np.min(vals)))
        history["iteration_cond"].append(float(math.exp(min(np.min(vals), 700))))

    elapsed = time.perf_counter() - t0
    idx = int(np.argmin(vals))
    d1, d2 = normalize_logs(pop[idx], n)
    r = result_from_scaling(
        name, "GA", A, family, d1, d2, "completed",
        seed, elapsed, counter.count, GA_GENS,
        "Real-coded genetic algorithm with crossover and mutation."
    )
    return r, d1, d2, history


def run_bo(name, A, family):
    n = A.shape[0]
    if not HAVE_SKOPT:
        return None, None, None, {"error": "scikit-optimize unavailable"}
    if n > BO_MAX_DIM:
        return None, None, None, {"skipped": True, "reason": f"n>{BO_MAX_DIM}"}

    counter = EvaluationCounter()

    def obj(x):
        counter.inc()
        d1, d2 = normalize_logs(np.asarray(x), n)
        return math.log(max(safe_cond(apply_scaling(A, d1, d2)), 1.0))

    dims = [Real(LOG_LOWER, LOG_UPPER) for _ in range(2 * n)]
    t0 = time.perf_counter()
    try:
        res = gp_minimize(
            obj,
            dims,
            n_initial_points=BO_INITIAL_POINTS,
            n_calls=BO_CALLS,
            random_state=SEED,
        )
        elapsed = time.perf_counter() - t0
        d1, d2 = normalize_logs(np.asarray(res.x), n)
        r = result_from_scaling(
            name, "BO", A, family, d1, d2, "completed",
            SEED, elapsed, counter.count, len(res.func_vals),
            "Bayesian optimization; reported only for completed runs."
        )
        hist = {
            "iteration_objective": list(map(float, res.func_vals)),
            "iteration_cond": [math.exp(min(float(v), 700)) for v in res.func_vals],
        }
        return r, d1, d2, hist
    except Exception as exc:
        return None, None, None, {"error": f"{type(exc).__name__}: {exc}"}


# =============================================================================
# PRECONDITIONERS
# =============================================================================

def ilu_preconditioner(A):
    S = sparse.csc_matrix(A)
    ilu = spilu(S)
    M = LinearOperator(A.shape, matvec=ilu.solve)
    return M


def ic0_preconditioner(A):
    """
    IC(0) is intentionally restricted to sparse SPD matrices.

    For a sparse matrix, IC(0) preserves the sparsity pattern of the lower
    triangle. Dense matrices are NOT accepted here, because applying the
    same definition to a dense matrix would effectively permit a complete
    Cholesky factorization and would not represent sparse IC(0).
    """
    if not sparse.issparse(A):
        raise ValueError("IC(0) is restricted to sparse matrices.")
    if not is_spd(A):
        raise ValueError("IC(0) requires an SPD matrix.")

    B = sparse.csr_matrix(A)
    n = B.shape[0]
    L = np.zeros((n, n), dtype=float)

    # Explicit IC(0) using the original lower-triangular sparsity pattern.
    pattern = B.toarray()
    for i in range(n):
        for j in range(i + 1):
            if j == i:
                s = float(B[i, i])
                for k in range(j):
                    s -= L[i, k] ** 2
                if s <= 0:
                    raise np.linalg.LinAlgError("IC(0) non-positive pivot.")
                L[i, j] = math.sqrt(s)
            elif pattern[i, j] != 0:
                s = float(B[i, j])
                for k in range(j):
                    s -= L[i, k] * L[j, k]
                if abs(L[j, j]) <= 1e-15:
                    raise np.linalg.LinAlgError("IC(0) zero pivot.")
                L[i, j] = s / L[j, j]

    def solve(v):
        y = solve_triangular(L, v, lower=True, check_finite=False)
        return solve_triangular(L.T, y, lower=False, check_finite=False)

    return LinearOperator(A.shape, matvec=solve)


def amg_preconditioner(A):
    if not HAVE_PYAMG:
        raise RuntimeError("pyamg is unavailable.")
    S = sparse.csr_matrix(A)
    if is_symmetric(A):
        ml = pyamg.smoothed_aggregation_solver(S, symmetry="symmetric")
    else:
        ml = pyamg.smoothed_aggregation_solver(S, symmetry="nonsymmetric")
    return ml.aspreconditioner()


def select_preconditioner(A, kind):
    if kind == "None":
        return None
    if kind == "ILU":
        return ilu_preconditioner(A)
    if kind == "IC0":
        return ic0_preconditioner(A)
    if kind == "AMG":
        return amg_preconditioner(A)
    raise ValueError(f"Unknown preconditioner: {kind}")


# =============================================================================
# SOLVER EXPERIMENTS
# =============================================================================

def solver_choice(A):
    # CG is used only when the ACTUAL matrix supplied to the solver is SPD.
    return "CG" if is_spd(A) else "GMRES"


def call_solver(A, b, solver_name, M):
    history = []

    def callback(xk):
        try:
            history.append(float(np.linalg.norm(b - A @ xk) / max(np.linalg.norm(b), 1e-30)))
        except Exception:
            pass

    t0 = time.perf_counter()

    if solver_name == "CG":
        try:
            x, info = cg(A, b, rtol=SOLVER_TOL, atol=0.0,
                         maxiter=SOLVER_MAXITER, M=M, callback=callback)
        except TypeError:
            x, info = cg(A, b, tol=SOLVER_TOL,
                         maxiter=SOLVER_MAXITER, M=M, callback=callback)
    else:
        try:
            x, info = gmres(A, b, rtol=SOLVER_TOL, atol=0.0,
                             restart=SOLVER_RESTART,
                             maxiter=SOLVER_MAXITER, M=M,
                             callback=callback, callback_type="pr_norm")
        except TypeError:
            x, info = gmres(A, b, tol=SOLVER_TOL,
                            restart=SOLVER_RESTART,
                            maxiter=SOLVER_MAXITER, M=M,
                            callback=callback)

    elapsed = time.perf_counter() - t0
    residual = float(np.linalg.norm(b - A @ x) /
                     max(np.linalg.norm(b), 1e-30))
    iterations = len(history)
    status = "converged" if info == 0 and residual <= SOLVER_TOL * 10 else f"not_converged(info={info})"
    return x, status, iterations, elapsed, residual


def solve_experiment(name, A_original, family, method, d1, d2):
    if A_original.shape[0] > LINEAR_SOLVE_MAX_N_DENSE and not sparse.issparse(A_original):
        return None

    # Algebraically equivalent transformed system:
    # B y = D1 b,  x = D2 y
    # This is a transformed-system experiment, not a claim that D1,D2
    # universally replace conventional preconditioners for A.
    A = A_original
    B = apply_scaling(A, d1, d2)
    n = A.shape[0]

    solver_name = solver_choice(B)

    if sparse.issparse(B):
        Bsolve = B.tocsr()
    else:
        Bsolve = np.asarray(B, dtype=float)

    rng = np.random.default_rng(SEED)
    b0 = rng.normal(size=n)
    b_scaled = d1 * b0

    preconditioners = ["None", "ILU", "AMG"]
    if sparse.issparse(B) and is_spd(B):
        preconditioners.append("IC0")

    rows = []

    for prec_name in preconditioners:
        try:
            build_times = []
            M = None
            for _ in range(1):
                t0 = time.perf_counter()
                M = select_preconditioner(Bsolve, prec_name)
                build_times.append(time.perf_counter() - t0)
            build_time = float(np.median(build_times))

            solve_times = []
            scaled_residuals = []
            original_residuals = []
            iteration_values = []
            statuses = []

            for trial in range(SOLVER_TRIALS):
                # Same RHS across trials for comparability.
                b = b_scaled.copy()
                x_y, status, iterations, solve_time, scaled_resid = call_solver(
                    Bsolve, b, solver_name, M
                )
                x_original = d2 * x_y
                original_resid = float(
                    np.linalg.norm(as_dense(A_original) @ x_original - b0)
                    / max(np.linalg.norm(b0), 1e-30)
                )
                solve_times.append(solve_time)
                scaled_residuals.append(scaled_resid)
                original_residuals.append(original_resid)
                iteration_values.append(iterations)
                statuses.append(status)

            median_time = float(np.median(solve_times))
            iqr_time = float(np.percentile(solve_times, 75) -
                             np.percentile(solve_times, 25))
            total_cost = build_time + median_time

            rows.append({
                "matrix": name,
                "family": family,
                "method": method,
                "n": n,
                "scaling_method": method,
                "preconditioner": prec_name,
                "solver": solver_name,
                "solver_status": ";".join(statuses),
                "solver_iterations": int(np.median(iteration_values)),
                "solver_time_sec": median_time,
                "solver_time_iqr_sec": iqr_time,
                "solver_residual_scaled": float(np.max(scaled_residuals)),
                "solver_residual_original": float(np.max(original_residuals)),
                "preconditioner_build_sec": build_time,
                "total_solver_cost_sec": total_cost,
                "solver_trials": SOLVER_TRIALS,
            })

        except Exception as exc:
            rows.append({
                "matrix": name,
                "family": family,
                "method": method,
                "n": n,
                "scaling_method": method,
                "preconditioner": prec_name,
                "solver": solver_name,
                "solver_status": "error",
                "solver_iterations": -1,
                "solver_time_sec": np.nan,
                "solver_time_iqr_sec": np.nan,
                "solver_residual_scaled": np.nan,
                "solver_residual_original": np.nan,
                "preconditioner_build_sec": np.nan,
                "total_solver_cost_sec": np.nan,
                "solver_trials": SOLVER_TRIALS,
                "notes": f"{type(exc).__name__}: {exc}",
            })

    return rows


# =============================================================================
# OUTPUTS
# =============================================================================

def save_scaling(name, method, d1, d2):
    np.savez(SCALE_DIR / f"{name}__{method}.npz", d1=d1, d2=d2)


def save_convergence(name, method, hist):
    path = CONVDIR / f"{name}__{method}.csv"

    if "iteration_objective" in hist:
        df = pd.DataFrame({
            "iteration": np.arange(1, len(hist["iteration_objective"]) + 1),
            "log_objective": hist["iteration_objective"],
            "condition_number": hist.get(
                "iteration_cond",
                [np.nan] * len(hist["iteration_objective"])
            ),
        })
        df.to_csv(path, index=False)

        if len(df) > 0:
            plt.figure(figsize=(7, 5))
            plt.semilogy(df["iteration"], np.maximum(df["condition_number"], 1.0))
            plt.xlabel("Iteration")
            plt.ylabel("Condition number")
            plt.title(f"{name} — {method} convergence")
            plt.tight_layout()
            plt.savefig(CONVDIR / f"{name}__{method}.png", dpi=180)
            plt.close()

    if "eval_objective" in hist and hist["eval_objective"]:
        pd.DataFrame({
            "evaluation": np.arange(1, len(hist["eval_objective"]) + 1),
            "log_objective": hist["eval_objective"],
        }).to_csv(CONVDIR / f"{name}__{method}__evaluations.csv", index=False)


def save_histograms(results_df):
    good = results_df[results_df["status"] == "completed"].copy()
    if good.empty:
        return

    for method in sorted(good["method"].unique()):
        x = good.loc[good["method"] == method, "reduction_factor"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(x) == 0:
            continue
        plt.figure(figsize=(7, 5))
        plt.hist(np.log10(np.maximum(x, 1e-15)), bins=min(12, max(3, len(x))))
        plt.xlabel("log10(condition-number reduction factor)")
        plt.ylabel("Frequency")
        plt.title(f"{method} — reduction factor")
        plt.tight_layout()
        plt.savefig(HISTDIR / f"reduction__{method}.png", dpi=180)
        plt.close()


def summarize_methods(results_df):
    completed = results_df[results_df["status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame()

    completed["log10_final_cond"] = np.log10(
        completed["final_cond"].replace([np.inf, -np.inf], np.nan)
    )

    summary = (
        completed.groupby("method")
        .agg(
            matrices=("matrix", "nunique"),
            mean_final_cond=("final_cond", "mean"),
            median_final_cond=("final_cond", "median"),
            median_log10_final_cond=("log10_final_cond", "median"),
            mean_reduction_factor=("reduction_factor", "mean"),
            median_reduction_factor=("reduction_factor", "median"),
            mean_runtime_sec=("runtime_sec", "mean"),
            median_runtime_sec=("runtime_sec", "median"),
            mean_objective_evaluations=("objective_evaluations", "mean"),
            median_objective_evaluations=("objective_evaluations", "median"),
            mean_iterations=("iterations", "mean"),
            median_iterations=("iterations", "median"),
        )
        .reset_index()
    )
    return summary


def stochastic_summary(stochastic_df):
    if stochastic_df.empty:
        return pd.DataFrame()

    completed = stochastic_df[stochastic_df["status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame()

    return (
        completed.groupby(["matrix", "method"])
        .agg(
            runs=("run", "count"),
            best_final_cond=("final_cond", "min"),
            median_final_cond=("final_cond", "median"),
            mean_final_cond=("final_cond", "mean"),
            std_final_cond=("final_cond", "std"),
            median_runtime_sec=("runtime_sec", "median"),
            median_objective_evaluations=("objective_evaluations", "median"),
        )
        .reset_index()
    )


def write_configuration():
    cfg = {
        "version": "V6.1",
        "seed": SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "pandas": pd.__version__,
        "pyamg_available": HAVE_PYAMG,
        "skopt_available": HAVE_SKOPT,
        "solver_trials": SOLVER_TRIALS,
        "solver_tolerance": SOLVER_TOL,
        "global_opt_max_n": GLOBAL_OPT_MAX_N,
        "notes": [
            "IC(0) restricted to sparse SPD matrices.",
            "AMG symmetry mode selected from actual matrix symmetry.",
            "Original-system residual is reported in addition to scaled residual.",
            "Skipped/failed runs are excluded from completed-method summaries.",
            "L-BFGS convergence is callback/iteration aligned.",
            "Total solver cost = preconditioner setup + median solve time.",
        ],
    }
    with open(OUTDIR / "configuration.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    started = time.perf_counter()
    write_configuration()

    all_results: List[MethodResult] = []
    stochastic_rows = []
    solver_rows = []
    errors = []

    benchmarks = build_benchmarks()

    for name, A, family in benchmarks:
        print(f"\n=== {name} | n={A.shape[0]} | {family} ===")

        methods = []

        try:
            r, d1, d2, hist = unscaled(A, name, family)
            all_results.append(r)
            methods.append((r, d1, d2, hist))
        except Exception as exc:
            errors.append({"matrix": name, "method": "Unscaled",
                           "error": traceback.format_exc()})

        try:
            r, d1, d2, hist = run_ruiz(name, A, family)
            all_results.append(r)
            methods.append((r, d1, d2, hist))
        except Exception:
            errors.append({"matrix": name, "method": "Ruiz",
                           "error": traceback.format_exc()})

        if A.shape[0] <= GLOBAL_OPT_MAX_N:
            try:
                r, d1, d2, hist = run_lbfgs(name, A, family)
                all_results.append(r)
                methods.append((r, d1, d2, hist))
            except Exception:
                errors.append({"matrix": name, "method": "L-BFGS",
                               "error": traceback.format_exc()})

            try:
                r, d1, d2, hist = run_hybrid(name, A, family)
                all_results.append(r)
                methods.append((r, d1, d2, hist))
            except Exception:
                errors.append({"matrix": name, "method": "Ruiz-LBFGS",
                               "error": traceback.format_exc()})

            # Reproducible stochastic replicates.
            for method_name, runner in [("SA", run_sa), ("GA", run_ga)]:
                for run in range(N_STOCHASTIC_REPEATS):
                    seed = SEED + 1000 * run + (1 if method_name == "SA" else 2)
                    try:
                        r, d1, d2, hist = runner(name, A, family, seed)
                        r.method = method_name
                        all_results.append(r)
                        stochastic_rows.append({
                            **asdict(r),
                            "run": run + 1,
                        })
                    except Exception:
                        errors.append({"matrix": name, "method": method_name,
                                       "run": run + 1,
                                       "error": traceback.format_exc()})

            try:
                r, d1, d2, hist = run_bo(name, A, family)
                if r is not None:
                    all_results.append(r)
                    methods.append((r, d1, d2, hist))
            except Exception:
                errors.append({"matrix": name, "method": "BO",
                               "error": traceback.format_exc()})

        # Save deterministic method outputs and convergence.
        for r, d1, d2, hist in methods:
            save_scaling(name, r.method, d1, d2)
            save_convergence(name, r.method, hist)

        # Solver experiments: Unscaled + Ruiz + Ruiz-LBFGS where available.
        for r, d1, d2, hist in methods:
            if r.method in {"Unscaled", "Ruiz", "Ruiz-LBFGS"}:
                try:
                    rows = solve_experiment(name, A, family, r.method, d1, d2)
                    if rows:
                        solver_rows.extend(rows)
                except Exception:
                    errors.append({"matrix": name,
                                   "method": r.method,
                                   "component": "solver",
                                   "error": traceback.format_exc()})

    results_df = pd.DataFrame([asdict(r) for r in all_results])
    results_df.to_csv(OUTDIR / "all_method_results.csv", index=False)

    summary_df = summarize_methods(results_df)
    summary_df.to_csv(OUTDIR / "method_summary.csv", index=False)

    stochastic_df = pd.DataFrame(stochastic_rows)
    if not stochastic_df.empty:
        stochastic_df.to_csv(OUTDIR / "stochastic_runs.csv", index=False)
        stochastic_sum = stochastic_summary(stochastic_df)
    else:
        stochastic_sum = pd.DataFrame()
    stochastic_sum.to_csv(OUTDIR / "stochastic_summary.csv", index=False)

    solver_df = pd.DataFrame(solver_rows)
    solver_df.to_csv(OUTDIR / "solver_results_all.csv", index=False)

    save_histograms(results_df)

    # A compact scalability experiment using deterministic matrix families.
    scalability_rows = []
    for n in [100, 300, 600, 1000]:
        A = finite_difference(n)
        t0 = time.perf_counter()
        d1, d2, hist = ruiz_scale(A)
        elapsed = time.perf_counter() - t0
        B = apply_scaling(A, d1, d2)
        scalability_rows.append({
            "matrix": "FiniteDiff",
            "n": n,
            "method": "Ruiz",
            "initial_cond": safe_cond(A),
            "final_cond": safe_cond(B),
            "reduction_factor": safe_cond(A) / safe_cond(B),
            "runtime_sec": elapsed,
            "iterations": len(hist),
        })

    scalability_df = pd.DataFrame(scalability_rows)
    scalability_df.to_csv(OUTDIR / "scalability_results.csv", index=False)
    scalability_df.to_csv(SCALABILITY_DIR / "scalability_results.csv", index=False)

    # Combined Excel workbook.
    if not results_df.empty:
        with pd.ExcelWriter(OUTDIR / "all_results.xlsx", engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name="all_method_results", index=False)
            summary_df.to_excel(writer, sheet_name="method_summary", index=False)
            if not stochastic_df.empty:
                stochastic_df.to_excel(writer, sheet_name="stochastic_runs", index=False)
                stochastic_sum.to_excel(writer, sheet_name="stochastic_summary", index=False)
            solver_df.to_excel(writer, sheet_name="solver_results", index=False)
            scalability_df.to_excel(writer, sheet_name="scalability", index=False)

    report = {
        "version": "V6.1",
        "started_seed": SEED,
        "elapsed_sec": time.perf_counter() - started,
        "matrices": len(benchmarks),
        "method_rows": len(results_df),
        "solver_rows": len(solver_df),
        "errors_recorded": len(errors),
        "pyamg_available": HAVE_PYAMG,
        "skopt_available": HAVE_SKOPT,
        "status": "completed_with_errors_recorded" if errors else "completed",
        "errors": errors,
    }
    with open(OUTDIR / "run_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("V6.1 COMPLETED")
    print(f"Output directory: {OUTDIR.resolve()}")
    print(f"Method rows: {len(results_df)}")
    print(f"Solver rows: {len(solver_df)}")
    print(f"Recorded errors: {len(errors)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
