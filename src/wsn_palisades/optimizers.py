"""K-subset selection: random / greedy / simple-NSGA-III / seeded-NSGA-III.

All four return a uniform result dict so they're directly comparable in the
k-sweep. NSGA-III uses a feasibility-only ``GeometricRepair`` operator and the
seeded variant additionally seeds the initial population with multiple greedy
solutions across a balanced weight simplex.
"""

from __future__ import annotations

import itertools
import os
from inspect import signature
from multiprocessing.pool import ThreadPool
from typing import Optional, Sequence

import numpy as np
from geopy.distance import geodesic
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem

# StarmapParallelization moved from pymoo.core.problem to pymoo.parallelization in 0.6.1.x
try:
    from pymoo.parallelization import StarmapParallelization  # pymoo >= 0.6.1
except ImportError:  # pragma: no cover
    from pymoo.core.problem import StarmapParallelization  # pymoo <= 0.6.0
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.ref_dirs import get_reference_directions

from .params import RANDOM_TRIALS_DEFAULT, SensorParams, nsga_params_for_k

NSGA_THREADS = min(max(1, (os.cpu_count() or 4) - 1), 8)


# --- weight & score helpers -------------------------------------------------


def _normalize_weights4(weights4, *, default=(0.25, 0.25, 0.25, 0.25), strict: bool = True):
    w = np.asarray(weights4, dtype=float).reshape(4)
    if not np.all(np.isfinite(w)):
        raise ValueError(f"weights4 contains non-finite values: {weights4}")
    s = float(w.sum())
    if s <= 0:
        if strict:
            raise ValueError(f"weights4 sum to {s}; pass valid weights (e.g., {default})")
        w = np.asarray(default, float)
        s = float(w.sum())
    return tuple(map(float, w / s))


def _scalar_score4(cov, gmean, mean_d, solar_norm, Dmax, weights4):
    w_cov, w_g, w_d, w_solar = _normalize_weights4(weights4, strict=False)
    cov_n = float(np.clip(cov / 100.0, 0.0, 1.0))
    g_n = float(np.clip(gmean, 0.0, 1.0))
    d_n = 0.0 if Dmax <= 0 else float(np.clip(mean_d / Dmax, 0.0, 1.0))
    s_n = float(np.clip(solar_norm, 0.0, 1.0))
    return float(w_cov * cov_n + w_g * g_n + w_d * d_n + w_solar * s_n)


def _distance_to_ideal(cov, gmean, mean_d, solar_norm, Dmax, weights4):
    """Weighted Euclidean distance to the ideal (1,1,1,1) in normalised objective space."""
    w_cov, w_g, w_d, w_solar = _normalize_weights4(weights4, strict=False)
    cov_n = float(np.clip(cov / 100.0, 0.0, 1.0))
    g_n = float(np.clip(gmean, 0.0, 1.0))
    d_n = 0.0 if Dmax <= 0 else float(np.clip(mean_d / Dmax, 0.0, 1.0))
    s_n = float(np.clip(solar_norm, 0.0, 1.0))
    dc, dg, dd, ds = 1.0 - cov_n, 1.0 - g_n, 1.0 - d_n, 1.0 - s_n
    return float(np.sqrt(w_cov * dc * dc + w_g * dg * dg + w_d * dd * dd + w_solar * ds * ds))


# --- pack utilities ---------------------------------------------------------


def _maybe_pack_masks(packs):
    if "_masks_packed" not in packs or "_grid_len" not in packs:
        masks = packs["masks"].astype(np.uint8)
        packs["_grid_len"] = masks.shape[1]
        packs["_masks_packed"] = np.packbits(masks, axis=1, bitorder="little")


def _cov_union_fast(packs, idxs) -> float:
    idxs = np.asarray(list(map(int, idxs)), dtype=int)
    if "_masks_packed" in packs and "_grid_len" in packs:
        arr = packs["_masks_packed"][idxs]
        u = np.bitwise_or.reduce(arr, axis=0)
        bits = np.unpackbits(u, bitorder="little")[: packs["_grid_len"]]
        return 100.0 * float(bits.mean())
    if idxs.size == 0:
        return 0.0
    return 100.0 * float(np.any(packs["masks"][idxs], axis=0).mean())


def _prep_optimizer_data(packs):
    _maybe_pack_masks(packs)
    masks = packs["masks"]
    metrics = packs["metrics"].reset_index(drop=True).copy()
    coords = list(zip(metrics.lon.values, metrics.lat.values))

    if "D" in packs and "Dmax" in packs:
        D = np.asarray(packs["D"], float)
        Dmax = float(packs["Dmax"])
    else:
        n = len(coords)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = geodesic((coords[i][1], coords[i][0]), (coords[j][1], coords[j][0])).meters
                D[i, j] = D[j, i] = d
        Dmax = float(np.max(D)) if len(coords) > 1 else 1.0
        packs["D"], packs["Dmax"] = D, Dmax
    return masks, metrics, coords, D, Dmax


def _feasible(idxs, D, min_sep) -> bool:
    idxs = np.asarray(list(map(int, idxs)), dtype=int)
    if idxs.size < 2:
        return True
    sub = D[np.ix_(idxs, idxs)]
    iu = np.triu_indices(len(idxs), 1)
    return bool((sub[iu] >= min_sep).all())


def _spacing_value(D, idxs, spacing_metric: str = "mean") -> float:
    idxs = np.asarray(list(map(int, idxs)), dtype=int)
    if idxs.size < 2:
        return 0.0
    sub = D[np.ix_(idxs, idxs)]
    iu = np.triu_indices(len(idxs), 1)
    v = sub[iu]
    if spacing_metric == "min":
        return float(v.min())
    if spacing_metric == "p10":
        return float(np.percentile(v, 10.0))
    return float(v.mean())


def _mean_solar_norm(idxs, metrics) -> float:
    col = "solar_norm_robust" if "solar_norm_robust" in metrics.columns else "solar_norm"
    if not len(idxs):
        return 0.0
    return float(metrics.loc[list(map(int, idxs)), col].mean())


def _score_solution4(idxs, packs, metrics, D, Dmax, weights, spacing_metric: str = "mean"):
    idxs = list(map(int, idxs))
    cov = _cov_union_fast(packs, idxs) if idxs else 0.0
    g = float(metrics.loc[idxs, "gamma_mean"].mean()) if idxs else 0.0
    d = _spacing_value(D, idxs, spacing_metric=spacing_metric)
    s = _mean_solar_norm(idxs, metrics)
    scalar = _scalar_score4(cov, g, d, s, Dmax, weights)
    dist = _distance_to_ideal(cov, g, d, s, Dmax, weights)
    return cov, g, d, s, scalar, dist


# --- pymoo evaluator shim ---------------------------------------------------


def _make_pymoo_evaluator(n_threads: Optional[int] = None):
    pool = ThreadPool(processes=(n_threads or max(1, (os.cpu_count() or 4) - 1)))
    try:
        params = signature(Evaluator.__init__).parameters
        if "runner" in params:
            return Evaluator(runner=StarmapParallelization(pool.starmap)), pool
        if "map_func" in params:
            return Evaluator(map_func=pool.map), pool
        pool.close(); pool.join()
        return None, None
    except Exception:
        try:
            pool.close(); pool.join()
        except Exception:
            pass
        return None, None


# --- history extraction -----------------------------------------------------


def _extract_history_pop_F(res, *, max_gens=None, every: int = 1):
    if not hasattr(res, "history") or res.history is None:
        return None
    out = []
    for gen_i, h in enumerate(res.history):
        if (gen_i % every) != 0:
            continue
        F = h.pop.get("F") if (hasattr(h, "pop") and h.pop is not None) else None
        if F is not None:
            out.append(np.asarray(F, dtype=float))
        if max_gens is not None and len(out) >= max_gens:
            break
    return out or None


def _attach_history_payload(out_dict, res, *, history_every: int = 1, history_max_gens=None):
    pop_F = _extract_history_pop_F(res, max_gens=history_max_gens, every=history_every)
    if pop_F is None:
        return out_dict
    out_dict["history"] = {
        "pop_F_by_gen": pop_F,
        "every": int(history_every),
        "n_gen": int(len(pop_F)),
        "n_obj": int(pop_F[-1].shape[1]),
        "pop_size_last": int(pop_F[-1].shape[0]),
    }
    return out_dict


# --- repair operator --------------------------------------------------------


class GeometricRepair(Repair):
    """Feasibility-only repair: dedup + min-sep fix.

    Replacement choice maximises min distance to the remaining nodes (geometry only),
    so the repair is unbiased w.r.t. coverage / γ / solar.
    """

    def __init__(self, D, n: int, k: int, min_sep_m: float, tries: int = 800):
        super().__init__()
        self.D = D
        self.n = int(n)
        self.k = int(k)
        self.min_sep_m = float(min_sep_m)
        self.tries = int(tries)

    def _best_geometric_replacement(self, current_idxs, drop_pos: int):
        idxs = list(map(int, current_idxs))
        chosen = set(idxs)
        others = [idxs[t] for t in range(self.k) if t != drop_pos]
        best_j, best_minsep = None, -1.0
        for _ in range(self.tries):
            j = int(np.random.randint(0, self.n))
            if j in chosen:
                continue
            if others:
                dists = self.D[j, others]
                if np.any(dists < self.min_sep_m):
                    continue
                minsep = float(np.min(dists))
            else:
                minsep = 1e18
            if minsep > best_minsep:
                best_minsep = minsep
                best_j = j
        return best_j

    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        for r in range(Y.shape[0]):
            idxs = list(map(int, Y[r]))

            seen = set()
            for i in range(self.k):
                if idxs[i] in seen:
                    j = self._best_geometric_replacement(idxs, drop_pos=i)
                    if j is not None:
                        idxs[i] = j
                seen.add(idxs[i])

            guard = 0
            changed = True
            while changed and guard < 4 * self.k:
                changed = False
                guard += 1
                for a in range(self.k):
                    for b in range(a + 1, self.k):
                        if self.D[idxs[a], idxs[b]] < self.min_sep_m:
                            drop = a if (np.random.rand() < 0.5) else b
                            j = self._best_geometric_replacement(idxs, drop_pos=drop)
                            if j is not None:
                                idxs[drop] = j
                                changed = True
            Y[r] = np.array(idxs, dtype=int)
        return Y


# --- result builder ---------------------------------------------------------


def _make_result(idxs, cov, g, d, s, scalar, dist, weights):
    return {
        "idxs": np.asarray(idxs, dtype=int),
        "coverage_pct": float(cov),
        "gamma_mean": float(g),
        "d_mean_m": float(d),
        "solar_mean": float(s),
        "scalar_score4": float(scalar),
        "dist_to_ideal": float(dist),
        "weights": tuple(map(float, weights)),
        "F": np.array([-cov, -g, -d, -s], dtype=float),
    }


# --- random and greedy ------------------------------------------------------


def random_select(
    packs,
    k: int,
    SP: SensorParams,
    weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    trials: int = RANDOM_TRIALS_DEFAULT,
):
    masks, metrics, coords, D, Dmax = _prep_optimizer_data(packs)
    n = len(coords)

    best = None
    for _ in range(trials):
        idxs = np.random.choice(n, size=k, replace=False)
        if not _feasible(idxs, D, SP.min_sep_m):
            continue
        cov, g, d, s, scalar, dist = _score_solution4(
            idxs, packs, metrics, D, Dmax, weights, spacing_metric="mean"
        )
        if best is None or scalar > best["scalar_score4"]:
            best = _make_result(idxs, cov, g, d, s, scalar, dist, weights)

    if best is None:
        idxs = np.random.choice(n, size=k, replace=False)
        cov, g, d, s, scalar, dist = _score_solution4(
            idxs, packs, metrics, D, Dmax, weights, spacing_metric="mean"
        )
        best = _make_result(idxs, cov, g, d, s, scalar, dist, weights)

    return best


def greedy_select(
    packs,
    k: int,
    SP: SensorParams,
    weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
):
    masks, metrics, coords, D, Dmax = _prep_optimizer_data(packs)
    n = len(coords)

    selected: list[int] = []
    for _ in range(k):
        best_i, best_scalar = None, -1e18
        for i in range(n):
            if i in selected:
                continue
            if not all(D[i, j] >= SP.min_sep_m for j in selected):
                continue
            idxs_try = selected + [int(i)]
            _, _, _, _, scalar, _ = _score_solution4(
                idxs_try, packs, metrics, D, Dmax, weights, spacing_metric="mean"
            )
            if scalar > best_scalar:
                best_scalar = scalar
                best_i = int(i)
        if best_i is None:
            break
        selected.append(best_i)

    cov, g, d, s, scalar, dist = _score_solution4(
        selected, packs, metrics, D, Dmax, weights, spacing_metric="mean"
    )
    return _make_result(selected, cov, g, d, s, scalar, dist, weights)


# --- shared NSGA-III runner -------------------------------------------------


def _build_problem_class(k: int, n: int, packs, metrics, D, Dmax, SP: SensorParams,
                         spacing_metric: str):
    solar_col = "solar_norm_robust" if "solar_norm_robust" in metrics.columns else "solar_norm"

    class SensorPlace(Problem):
        def __init__(self):
            super().__init__(
                n_var=k,
                n_obj=4,
                n_constr=int(k * (k - 1) / 2),
                xl=0,
                xu=n - 1,
                vtype=int,
            )

        def _evaluate(self, X, out, *args, **kwargs):
            m = X.shape[0]
            F = np.zeros((m, 4))
            G = np.zeros((m, int(k * (k - 1) / 2)))
            for r in range(m):
                idxs = list(map(int, X[r]))
                if len(set(idxs)) != len(idxs):
                    F[r, :] = 1e6
                    G[r, :] = 1e6
                    continue
                cov = _cov_union_fast(packs, idxs)
                g = float(metrics.loc[idxs, "gamma_mean"].mean()) if idxs else 0.0
                d = _spacing_value(D, idxs, spacing_metric=spacing_metric)
                s = float(metrics.loc[idxs, solar_col].mean()) if idxs else 0.0
                F[r] = [
                    -np.clip(cov / 100.0, 0.0, 1.0),
                    -np.clip(g, 0.0, 1.0),
                    -np.clip(0.0 if Dmax <= 0 else d / Dmax, 0.0, 1.0),
                    -np.clip(s, 0.0, 1.0),
                ]
                gi = 0
                for a in range(k):
                    for b in range(a + 1, k):
                        G[r, gi] = max(0.0, SP.min_sep_m - D[idxs[a], idxs[b]])
                        gi += 1
            out["F"] = F
            out["G"] = G

    return SensorPlace


def _run_nsga(
    packs,
    k: int,
    SP: SensorParams,
    sampling: Sampling,
    weights,
    spacing_metric: str,
    final_pick: str,
    max_gen: int,
    pop_mult: float,
    partitions: int,
    use_threads: bool,
    n_threads: int,
    random_seed: int,
):
    masks, metrics, coords, D, Dmax = _prep_optimizer_data(packs)
    n = len(coords)
    SensorPlace = _build_problem_class(k, n, packs, metrics, D, Dmax, SP, spacing_metric)

    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=partitions)
    pop_size = int(pop_mult * len(ref_dirs))
    repair = GeometricRepair(D=D, n=n, k=k, min_sep_m=SP.min_sep_m, tries=800)

    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=pop_size,
        eliminate_duplicates=True,
        sampling=sampling,
        repair=repair,
    )
    termination = DefaultMultiObjectiveTermination(
        xtol=1e-8, cvtol=1e-6, ftol=0.010, period=30, n_max_gen=max_gen
    )

    evaluator, pool = _make_pymoo_evaluator(n_threads=n_threads) if use_threads else (None, None)
    try:
        if evaluator is not None:
            res = minimize(
                SensorPlace(),
                algorithm,
                termination,
                seed=random_seed,
                verbose=False,
                evaluator=evaluator,
                save_history=True,
            )
        else:
            res = minimize(
                SensorPlace(),
                algorithm,
                termination,
                seed=random_seed,
                verbose=False,
                save_history=True,
            )
    finally:
        if pool is not None:
            try:
                pool.close(); pool.join()
            except Exception:
                pass

    X_all = res.X
    G_all = res.G
    feas_idx = (
        np.where(np.all(G_all <= 1e-9, axis=1))[0]
        if G_all is not None
        else np.arange(len(X_all))
    )
    if len(feas_idx) == 0 and G_all is not None:
        feas_idx = np.where(np.all(G_all <= 0.0, axis=1))[0]
    if len(feas_idx) == 0:
        return None, res, metrics, D, Dmax

    X = X_all[feas_idx]
    best_idx = None
    best_key = None
    for r_i in range(X.shape[0]):
        idxs = list(map(int, X[r_i]))
        cov, g, d, s, scalar, dist = _score_solution4(
            idxs, packs, metrics, D, Dmax, weights, spacing_metric=spacing_metric
        )
        key = dist if final_pick == "distance" else -scalar
        if best_key is None or key < best_key:
            best_key = key
            best_idx = r_i

    idxs = X[best_idx].astype(int)
    cov, g, d, s, scalar, dist = _score_solution4(
        idxs, packs, metrics, D, Dmax, weights, spacing_metric=spacing_metric
    )
    out = _make_result(idxs, cov, g, d, s, scalar, dist, weights)
    return _attach_history_payload(out, res, history_every=2), res, metrics, D, Dmax


# --- samplers ---------------------------------------------------------------


class _SimpleRandomSampling(Sampling):
    def __init__(self, n: int, k: int, D, min_sep_m: float):
        super().__init__()
        self.n, self.k, self.D, self.min_sep_m = n, k, D, min_sep_m

    def _do(self, problem, n_samples, **kwargs):
        samples = []
        tries = 0
        while len(samples) < n_samples and tries < 20000:
            cand = np.random.choice(self.n, size=self.k, replace=False)
            tries += 1
            if _feasible(cand, self.D, self.min_sep_m):
                samples.append(cand.astype(int))
        while len(samples) < n_samples:
            samples.append(np.random.choice(self.n, size=self.k, replace=False).astype(int))
        return np.array(samples[:n_samples], dtype=int)


def _simplex_weights_grid_balanced(M: int = 4, L: int = 4, min_nonzero: int = 2):
    grid = []
    for parts in itertools.product(range(L + 1), repeat=M):
        if sum(parts) != L:
            continue
        if sum(1 for x in parts if x > 0) < min_nonzero:
            continue
        w = np.array(parts, float)
        if w.sum() <= 0:
            continue
        grid.append(w / w.sum())
    return np.array(grid, float)


class _FeasibleSeededSampling(Sampling):
    def __init__(self, packs, n: int, k: int, D, min_sep_m: float, SP: SensorParams,
                 external_seeds, multi_seed: bool, wgrid_L: int):
        super().__init__()
        self.packs = packs
        self.n, self.k, self.D, self.min_sep_m = n, k, D, min_sep_m
        self.SP = SP
        self.external_seeds = external_seeds
        self.multi_seed = multi_seed
        self.wgrid_L = wgrid_L

    def _do(self, problem, n_samples, **kwargs):
        seeds, seen = [], set()

        def _add(seed):
            t = tuple(map(int, seed))
            if len(t) == self.k and t not in seen and _feasible(t, self.D, self.min_sep_m):
                seeds.append(np.array(t, dtype=int))
                seen.add(t)

        if self.external_seeds is not None:
            ext = (
                self.external_seeds
                if isinstance(self.external_seeds, (list, tuple))
                else [self.external_seeds]
            )
            for seed in ext:
                try:
                    _add(seed)
                except Exception:
                    pass

        if self.multi_seed:
            for wv in _simplex_weights_grid_balanced(M=4, L=self.wgrid_L, min_nonzero=3):
                try:
                    _add(greedy_select(self.packs, self.k, self.SP, weights=tuple(wv))["idxs"])
                except Exception:
                    pass

        tries = 0
        while len(seeds) < n_samples and tries < 12000:
            cand = np.random.choice(self.n, size=self.k, replace=False)
            tries += 1
            if _feasible(cand, self.D, self.min_sep_m):
                _add(cand)

        if len(seeds) == 0:
            seeds = [np.random.choice(self.n, size=self.k, replace=False).astype(int)]
        return np.array(seeds[: n_samples], dtype=int)


# --- public NSGA-III wrappers ----------------------------------------------


def simple_nsga_select(
    packs,
    k: int,
    SP: SensorParams,
    max_gen: Optional[int] = None,
    weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    pop_mult: Optional[float] = None,
    partitions: Optional[int] = None,
    spacing_metric: str = "mean",
    final_pick: str = "distance",
    use_threads: bool = True,
    n_threads: int = NSGA_THREADS,
    random_seed: int = 42,
):
    """NSGA-III with random feasible sampling — no greedy seeding."""
    p = nsga_params_for_k(k)
    if max_gen is None:
        max_gen = max(200, p["max_gen"] // 3)
    if pop_mult is None:
        pop_mult = p["pop_mult"]
    if partitions is None:
        partitions = p["partitions"]

    _, _, _, D, _ = _prep_optimizer_data(packs)
    n = len(packs["candidates"])
    sampling = _SimpleRandomSampling(n=n, k=k, D=D, min_sep_m=SP.min_sep_m)

    out, _, _, _, _ = _run_nsga(
        packs, k, SP,
        sampling=sampling,
        weights=weights,
        spacing_metric=spacing_metric,
        final_pick=final_pick,
        max_gen=max_gen,
        pop_mult=pop_mult,
        partitions=partitions,
        use_threads=use_threads,
        n_threads=n_threads,
        random_seed=random_seed,
    )
    if out is None:
        return random_select(packs, k, SP, weights=weights)
    return out


def nsga_select(
    packs,
    k: int,
    SP: SensorParams,
    max_gen: Optional[int] = None,
    weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    external_seeds=None,
    pop_mult: Optional[float] = None,
    partitions: Optional[int] = None,
    spacing_metric: str = "mean",
    final_pick: str = "distance",
    multi_seed: bool = True,
    wgrid_L: int = 4,
    use_threads: bool = True,
    n_threads: int = NSGA_THREADS,
    random_seed: int = 42,
):
    """Seeded NSGA-III: initial population includes greedy solutions for many weight vectors."""
    p = nsga_params_for_k(k)
    if max_gen is None:
        max_gen = p["max_gen"]
    if pop_mult is None:
        pop_mult = p["pop_mult"]
    if partitions is None:
        partitions = p["partitions"]

    _, _, _, D, _ = _prep_optimizer_data(packs)
    n = len(packs["candidates"])
    sampling = _FeasibleSeededSampling(
        packs=packs, n=n, k=k, D=D, min_sep_m=SP.min_sep_m, SP=SP,
        external_seeds=external_seeds, multi_seed=multi_seed, wgrid_L=wgrid_L,
    )

    out, _, _, _, _ = _run_nsga(
        packs, k, SP,
        sampling=sampling,
        weights=weights,
        spacing_metric=spacing_metric,
        final_pick=final_pick,
        max_gen=max_gen,
        pop_mult=pop_mult,
        partitions=partitions,
        use_threads=use_threads,
        n_threads=n_threads,
        random_seed=random_seed,
    )
    if out is None:
        return greedy_select(packs, k, SP, weights=weights)
    return out


__all__ = [
    "GeometricRepair",
    "random_select",
    "greedy_select",
    "simple_nsga_select",
    "nsga_select",
    "_prep_optimizer_data",
    "_score_solution4",
    "_scalar_score4",
    "_distance_to_ideal",
    "_cov_union_fast",
    "_feasible",
    "_spacing_value",
    "_normalize_weights4",
]
