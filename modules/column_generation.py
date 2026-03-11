#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 21:39:34 2026

@author: anahitamolavi
"""

# Standard library
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

# Third-party libraries
import numpy as np
import pandas as pd

# Optimization/modeling
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Modules
from modules import utilities


# ---------------------------- Cache --------------------------- #
@dataclass
class DFCache:
    created_s: np.ndarray
    ready_s: np.ndarray
    p_lat: np.ndarray
    p_lon: np.ndarray
    d_lat: np.ndarray
    d_lon: np.ndarray
    delivery_id: np.ndarray


def make_cache(df: pd.DataFrame) -> DFCache:
    d = df.reset_index(drop=True)
    created_s = (
        pd.to_datetime(d["created_at"], utc=True).astype("int64") // 10**9
    ).to_numpy(np.int64)
    ready_s = (
        pd.to_datetime(d["food_ready_time"], utc=True).astype("int64") // 10**9
    ).to_numpy(np.int64)

    p_lat = d["pickup_lat"].to_numpy(np.float64)
    p_lon = d["pickup_long"].to_numpy(np.float64)
    d_lat = d["dropoff_lat"].to_numpy(np.float64)
    d_lon = d["dropoff_long"].to_numpy(np.float64)

    delivery_id = d["delivery_id"].to_numpy()

    return DFCache(created_s=created_s, ready_s=ready_s, p_lat=p_lat, p_lon=p_lon, d_lat=d_lat, d_lon=d_lon, delivery_id=delivery_id)

@contextmanager
def timed(label: str, timings: dict):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    timings[label] = timings.get(label, 0.0) + dt
    print(f"[time] {label}: {dt:.3f}s")

# ---------------------------- Route column definition ---------------------------- #
@dataclass(frozen=True)
class RouteCol:
    rid: int
    deliveries: Tuple[int, ...]          # delivery indices (0..n-1)
    cost_s: float                        # route end - T0
    dur_sum_s: float                     # sum of (dropoff - created) across deliveries


# ---------------------------- Route simulation (uses cache) -------------------------- #
def simulate_route(
    seq: List[int],
    cache: DFCache,
    T0_s: int,
    L_s: Optional[float] = None,
) -> Tuple[float, float]:
    created_s = cache.created_s
    ready_s = cache.ready_s
    p_lat, p_lon = cache.p_lat, cache.p_lon
    d_lat, d_lon = cache.d_lat, cache.d_lon

    t = T0_s
    dur_sum = 0.0
    prev_drop_lat = None
    prev_drop_lon = None

    for idx, i in enumerate(seq):
        if idx == 0:
            deadhead = 0.0
        else:
            deadhead = utilities.travel_time_s(prev_drop_lat, prev_drop_lon, p_lat[i], p_lon[i])

        arrive_pickup = t + deadhead
        pickup_time = max(arrive_pickup, ready_s[i], T0_s)

        intra = utilities.travel_time_s(p_lat[i], p_lon[i], d_lat[i], d_lon[i])
        dropoff_time = pickup_time + intra

        dur_i = dropoff_time - created_s[i]

        dur_sum += dur_i

        t = dropoff_time
        prev_drop_lat = d_lat[i]
        prev_drop_lon = d_lon[i]

    cost = t - T0_s
    return float(cost), float(dur_sum)

def simulate_route_records(
    seq: List[int],
    cache: DFCache,
    T0_s: int,
) -> Tuple[List[Dict[str, float]], float]:
    """
    Simulate a route and return per-delivery assignment records + route end time.
    """
    created_s = cache.created_s
    ready_s = cache.ready_s
    p_lat, p_lon = cache.p_lat, cache.p_lon
    d_lat, d_lon = cache.d_lat, cache.d_lon

    t = float(T0_s)
    prev_drop_lat = None
    prev_drop_lon = None

    recs: List[Dict[str, float]] = []

    for idx, i in enumerate(seq):
        if idx == 0:
            deadhead = 0.0
        else:
            deadhead = utilities.travel_time_s(prev_drop_lat, prev_drop_lon, p_lat[i], p_lon[i])

        arrive_pickup = t + deadhead
        pickup_time = max(arrive_pickup, float(ready_s[i]), float(T0_s))

        intra = utilities.travel_time_s(p_lat[i], p_lon[i], d_lat[i], d_lon[i])
        dropoff_time = pickup_time + intra

        duration_s = float(dropoff_time - created_s[i])

        recs.append(
            {
                "delivery_id": cache.delivery_id[i],
                "PickupTime_s": float(pickup_time),
                "DropoffTime_s": float(dropoff_time),
                "Duration_s": float(duration_s),
            }
        )

        t = float(dropoff_time)
        prev_drop_lat = d_lat[i]
        prev_drop_lon = d_lon[i]

    return recs, float(t)



# ---------------------------- Candidate route generation ---------------------------- #

def build_knn_pickup(df: pd.DataFrame, k: int = 15) -> List[List[int]]:
    """KNN lists by pickup proximity (cheap, no sklearn)"""
    coords = df[["pickup_lat", "pickup_long"]].to_numpy(dtype=np.float64)
    n = len(coords)
    knn: List[List[int]] = []

    kk = min(k, n - 1)
    for i in range(n):
        diff = coords - coords[i]
        d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]

        # k+1 smallest (includes self) without full sort
        idx = np.argpartition(d2, kth=kk)[:kk + 1]
        idx = idx[idx != i]
        idx = idx[np.argsort(d2[idx])][:kk]
        knn.append(idx.astype(int).tolist())

    return knn


def build_feasible_routes_cover(
    df: pd.DataFrame,
    cache: DFCache,             # <-- NEW (STEP 1)
    T0_s: int,
    max_drivers: int = 50,
    max_len: int = 12,
    k_nn: int = 30,
) -> List[Tuple[int, ...]]:
    """
    Deterministically build <= max_drivers routes covering all deliveries.
    Goal: feasibility (not optimality).
    """
    df = df.reset_index(drop=True)
    n = len(df)

    ready_s = cache.ready_s
    knn = build_knn_pickup(df, k=k_nn)

    # process in increasing ready time (helps reduce waiting)
    order = np.argsort(ready_s).tolist()
    unassigned = set(order)

    routes: List[List[int]] = []
    # Start up to max_drivers routes with earliest ready deliveries
    for _ in range(min(max_drivers, n)):
        if not unassigned:
            break
        seed = next(iter(unassigned))
        unassigned.remove(seed)
        routes.append([seed])

    # Greedily grow routes round-robin
    changed = True
    while unassigned and changed:
        changed = False
        for r in routes:
            if not unassigned:
                break
            if len(r) >= max_len:
                continue

            last = r[-1]
            cand = [c for c in knn[last] if c in unassigned]
            if not cand:
                c = min(unassigned, key=lambda i: ready_s[i])
            else:
                c = int(cand[0])

            r.append(int(c))
            unassigned.remove(int(c))
            changed = True

    # If still unassigned (because max_len too small), append them to routes anyway
    if unassigned:
        unassigned = list(unassigned)
        idx = 0
        for i in unassigned:
            routes[idx % len(routes)].append(int(i))
            idx += 1

    return [tuple(r) for r in routes]


def generate_initial_routes(
    df: pd.DataFrame,
    T0_s: int,
    cache: DFCache,         
    max_len: int = 7,
    k_nn: int = 15,
    max_pool: int = 2500,
    L_s: float = 2700,
    seed_cap: float = 100,
) -> List[RouteCol]:
    """
    Build a sparse pool:
    - all singletons
    - good pairs/triples from pickup KNN
    - some greedy-extended routes up to max_len
    """
    df = df.reset_index(drop=True)
    n = len(df)

    knn = build_knn_pickup(df, k=k_nn)

    cols: List[RouteCol] = []
    rid = 0

    # Singletons (always include)
    for i in range(n):
        cost, dur_sum = simulate_route([i], cache, T0_s, L_s)
        cols.append(RouteCol(rid, (i,), cost, dur_sum))
        rid += 1

    # Pairs from KNN
    for i in range(n):
        for k in knn[i]:
            k = int(k)
            if k == i:
                continue
            cost, dur_sum = simulate_route([i, k], cache, T0_s, L_s)
            cols.append(RouteCol(rid, (i, k), cost, dur_sum))
            rid += 1

    # Greedy extensions (dual-free initial heuristic)
    rng = np.random.default_rng(0)
    seeds = rng.choice(n, size=min(n, seed_cap), replace=False)

    for s in seeds:
        route = [int(s)]
        used = {int(s)}

        while len(route) < max_len:
            last = route[-1]
            cand = [int(c) for c in knn[last] if int(c) not in used]
            if not cand:
                break

            # STEP 5: compute cost(route) ONCE per extension step
            cost1, _ = simulate_route(route, cache, T0_s, L_s)

            best = None
            best_score = float("inf")
            for c in cand[:10]:
                cost2, _ = simulate_route(route + [c], cache, T0_s, L_s)
                inc = cost2 - cost1
                if inc < best_score:
                    best_score = inc
                    best = c

            if best is None:
                break
            route.append(best)
            used.add(best)

        if len(route) >= 2:
            cost, dur_sum = simulate_route(route, cache, T0_s, L_s)
            cols.append(RouteCol(rid, tuple(route), cost, dur_sum))
            rid += 1

    # Keep pool bounded
    cols.sort(key=lambda r: (len(r.deliveries), r.cost_s + 0.01 * r.dur_sum_s))
    cols = cols[:max_pool]

    return cols


def validate_pool(cols, n, max_drivers, label="pool"):
    import math
    covered = np.zeros(n, dtype=int)
    max_len = 0
    bad = 0
    for c in cols:
        if not (np.isfinite(c.cost_s) and np.isfinite(c.dur_sum_s)):
            bad += 1
            continue
        max_len = max(max_len, len(c.deliveries))
        for i in c.deliveries:
            if 0 <= i < n:
                covered[i] += 1

    missing = np.where(covered == 0)[0]
    lb_routes = math.ceil(n / max_len) if max_len > 0 else 10**9

    print(
        f"[{label}] n_cols={len(cols)} bad_cols={bad} max_len={max_len} "
        f"missing_coverage={len(missing)} LB_routes={lb_routes} cap={max_drivers}"
    )

    if len(missing) > 0:
        print(f"[{label}] first missing indices:", missing[:20])


# ---------------------------- Master problem (set partitioning) ---------------------------- #

def solve_master(
    n_deliveries: int,
    cols: List[RouteCol],
    L_s: float,
    max_routes: int,
    solver_name: str = "glpk",
    time_limit_s: int = 45,
    relax: bool = False,
    want_duals: bool = False,
):
    """
    Solve master MILP (or LP if relax=True).
    Returns (model, result, z_values, duals_or_none).
    """
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n_deliveries - 1)
    m.R = pyo.RangeSet(0, len(cols) - 1)

    routes_by_i = {i: [] for i in range(n_deliveries)}
    for r_idx, col in enumerate(cols):
        for ii in col.deliveries:
            ii = int(ii)
            if 0 <= ii < n_deliveries:
                routes_by_i[ii].append(r_idx)

    empty = [i for i, rs in routes_by_i.items() if len(rs) == 0]
    if empty:
        raise ValueError(f"Coverage missing for deliveries: {empty[:20]} (total {len(empty)})")

    m.cost = pyo.Param(m.R, initialize={r: cols[r].cost_s for r in range(len(cols))})
    m.dsum = pyo.Param(m.R, initialize={r: cols[r].dur_sum_s for r in range(len(cols))})

    if relax:
        m.z = pyo.Var(m.R, bounds=(0.0, 1.0))
    else:
        m.z = pyo.Var(m.R, within=pyo.Binary)

    def cover_rule(mdl, i):
        return sum(mdl.z[r] for r in routes_by_i[int(i)]) == 1

    m.cover = pyo.Constraint(m.I, rule=cover_rule)

    m.max_routes = pyo.Constraint(expr=sum(m.z[r] for r in m.R) <= max_routes)

    m.sla = pyo.Constraint(expr=sum(m.dsum[r] * m.z[r] for r in m.R) <= L_s * n_deliveries)

    m.obj = pyo.Objective(expr=sum(m.cost[r] * m.z[r] for r in m.R), sense=pyo.minimize)

    if want_duals:
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    opt = SolverFactory(solver_name)

    if solver_name.lower() in {"cbc"}:
        opt.options["seconds"] = time_limit_s
    elif solver_name.lower() in {"glpk"}:
        pass
    elif solver_name.lower() in {"gurobi"}:
        opt.options["TimeLimit"] = time_limit_s

    res = opt.solve(m, tee=False)

    status = res.solver.status
    term = res.solver.termination_condition
    ok = (status == pyo.SolverStatus.ok) and (
        term
        in {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
            pyo.TerminationCondition.locallyOptimal,
        }
    )

    if not ok:
        print(f"[solve_master] Solver failed. status={status}, termination={term}")
        z = np.zeros(len(cols), dtype=float)
        return m, res, z, None

    z = np.array([(pyo.value(m.z[r]) or 0.0) for r in m.R], dtype=float)

    duals = None
    if want_duals:
        cover_duals = np.array([m.dual[m.cover[i]] for i in m.I], dtype=float)
        sla_dual = float(m.dual[m.sla]) if m.sla in m.dual else 0.0
        routes_dual = float(m.dual[m.max_routes]) if m.max_routes in m.dual else 0.0
        duals = {"pi": cover_duals, "lambda": sla_dual, "sigma": routes_dual}

    return m, res, z, duals


# ---------------------------- Heuristic "pricing" (dual-guided route builder) - uses cache ---------------------------- #

def pricing_heuristic(
    df: pd.DataFrame,
    duals: Dict[str, np.ndarray],
    T0_s: int,
    cache: DFCache,          # <-- NEW
    L_s: Optional[float] = None,
    n_new: int = 50,
    max_len: int = 7,
    k_nn: int = 20,
    candidates_per_step: int = 10,
) -> List[Tuple[Tuple[int, ...], float]]:
    """
    Builds candidate routes attempting to minimize reduced cost:
        rc = cost - sum(pi_i) - lambda * dur_sum
    Returns list of (route_deliveries, reduced_cost).
    """
    df = df.reset_index(drop=True)
    n = len(df)

    pi = duals["pi"]
    lam = duals.get("lambda", 0.0)

    knn = build_knn_pickup(df, k=k_nn)

    seeds = np.argsort(-pi)[:min(n, 200)]

    candidates = []
    for s in seeds:
        route = [int(s)]
        used = {int(s)}
        for _ in range(max_len - 1):
            last = route[-1]
            nn = [int(c) for c in knn[last] if int(c) not in used]
            if not nn:
                break

            best = None
            best_rc = float("inf")
            for c in nn[:candidates_per_step]:
                test = route + [int(c)]
                cost, dsum = simulate_route(test, cache, T0_s, L_s)
                rc = cost - pi[test].sum() - lam * dsum
                if rc < best_rc:
                    best_rc = rc
                    best = int(c)

            if best is None:
                break
            route.append(best)
            used.add(best)

        cost, dsum = simulate_route(route, cache, T0_s, L_s)
        rc = cost - pi[route].sum() - lam * dsum
        if len(route) >= 2:
            candidates.append((tuple(route), float(rc)))

    candidates.sort(key=lambda x: x[1])
    return candidates[:n_new]


# ------------------------ Warm-start routes from greedy (to ensure SLA-feasible initial RMP) -------------------- #

def warmstart_routes_from_greedy(
    df: pd.DataFrame,
    cache: DFCache,
    T0_s: int,
    L_minutes: float,
    max_drivers: int,
) -> Tuple[List[Tuple[int, ...]], Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    """
    Build an initial set of routes from the greedy solver so the master problem
    is feasible:
      - set partitioning coverage
      - max_drivers constraint
      - average delivery duration SLA (global avg <= L_minutes)

    """
    greedy_mod = None
    for modname in ("modules.greedy", "greedy"):
        try:
            greedy_mod = __import__(modname, fromlist=["*"])
            break
        except Exception:
            continue

    if greedy_mod is None:
        print("[warmstart] greedy module not found; skipping warm-start.")
        return [], None, None

    # Prefer solve_with_min_drivers if available (it will try to satisfy SLA by increasing drivers)
    T0_utc = pd.Timestamp(int(T0_s), unit="s", tz="UTC").isoformat()

    assignment_df = None
    metrics = None
    try:
        if hasattr(greedy_mod, "solve_with_min_drivers"):
            assignment_df, metrics = greedy_mod.solve_with_min_drivers(
                deliveries_df=df,
                T0_utc=T0_utc,
                L_minutes=L_minutes,
                start_drivers=min(10, max_drivers),
                max_drivers=max_drivers,
                drivers_step=5,
                alpha=4.0,
                time_window_s=3600,
                use_region=("region_id" in df.columns),
                top_k=50,
                verbose=False,
            )
        elif hasattr(greedy_mod, "greedy_assign_deliveries"):
            assignment_df, metrics = greedy_mod.greedy_assign_deliveries(
                deliveries_df=df,
                num_drivers=max_drivers,
                T0_utc=T0_utc,
                L_minutes=L_minutes,
                alpha=4.0,
                time_window_s=3600,
                use_region=("region_id" in df.columns),
                top_k=50,
                verbose=False,
            )
    except Exception as e:
        print(f"[warmstart] greedy solver failed; skipping warm-start. err={e!r}")
        return [], None, None

    if assignment_df is None or metrics is None or assignment_df.empty:
        print("[warmstart] greedy produced empty assignment; skipping warm-start.")
        return [], assignment_df, metrics

    # Map delivery_id -> row index in df (cache uses reset_index(drop=True))
    delivery_id_to_idx = {cache.delivery_id[i]: i for i in range(len(cache.delivery_id))}

    routes: List[Tuple[int, ...]] = []
    # group by driver and order by pickup time to recover the route sequence
    for _, grp in assignment_df.groupby("driver"):
        grp2 = grp.sort_values("PickupTime_s", kind="mergesort")
        seq_idx: List[int] = []
        ok = True
        for did in grp2["delivery_id"].tolist():
            if did not in delivery_id_to_idx:
                ok = False
                break
            seq_idx.append(int(delivery_id_to_idx[did]))
        if ok and seq_idx:
            routes.append(tuple(seq_idx))

    # Diagnostics
    try:
        avg_min = float(metrics.get("avg_duration_min", float("nan")))
        print(f"[warmstart] greedy routes={len(routes)} avg_duration={avg_min:.3f} min")
        if np.isfinite(avg_min) and avg_min > L_minutes + 1e-6:
            print("[warmstart] WARNING: greedy warm-start does not meet SLA; RMP may still be infeasible.")
    except Exception:
        pass

    return routes, assignment_df, metrics

def column_generation_solve(
    delivery_df: pd.DataFrame,
    T0_s: int,
    L_minutes: float = 45.0,
    max_drivers: int = 50,
    solver_name: str = "glpk",
    overall_time_limit_s: int = 55,
    return_details: bool = False,


    # -------- tuning knobs (defaults are more aggressive/faster) --------
    init_max_pool: int = 3500,      
    init_seed_cap: int = 100,       
    knn_init: int = 15,             
    knn_feas: int = 20,             
    feas_max_len: int = 10,         
    lp_iters: int = 5,              
    pricing_new: int = 30,          
    pricing_candidates_per_step: int = 10,  
    pool_cap_after_lp: int = 3500,  
    rc_threshold: float = -1e-6,    
):
    """
    Restricted CG loop: build pool, iterate LP + heuristic pricing, then solve integer master.
    """
    timings: dict = {}
    t_start = time.perf_counter()
    start_wall = time.time()
    L_s = L_minutes * 60.0
    df0 = delivery_df.reset_index(drop=True)

    with timed("make_cache", timings):
        cache = make_cache(df0)

    # 1) initial pool
    with timed("generate_initial_routes", timings):
        cols = generate_initial_routes(
            df0, T0_s, cache=cache,
            max_len=7,
            k_nn=knn_init,
            max_pool=init_max_pool,
            L_s=L_s,
            seed_cap=init_seed_cap,
        )

    existing = set(c.deliveries for c in cols)
    print(f"[diag] init cols={len(cols)}")

    # feasibility cover
    with timed("build_feasible_routes_cover", timings):
        feas_routes = build_feasible_routes_cover(
            df0,
            cache=cache,
            T0_s=T0_s,
            max_drivers=max_drivers,
            max_len=feas_max_len,
            k_nn=knn_feas,
        )

    with timed("inject_feas_routes_simulate", timings):
        added_feas = 0
        for seq in feas_routes:
            if seq in existing:
                continue
            cost, dsum = simulate_route(list(seq), cache, T0_s, L_s)
            cols.append(RouteCol(len(cols), seq, cost, dsum))
            existing.add(seq)
            added_feas += 1

    print("[diag] added feasibility routes:", added_feas, "total cols:", len(cols))

    cover_set = set(feas_routes)
    cover_cols = [c for c in cols if c.deliveries in cover_set]
    dur_cover = sum(c.dur_sum_s for c in cover_cols)
    avg_cover_min = dur_cover / len(df0) / 60.0
    print("[diag] injected cover avg duration:", avg_cover_min, "minutes")
    # ---- Warm-start with greedy routes to ensure RMP feasibility w.r.t SLA ----
    warm_routes, warm_assignment_df, warm_metrics = warmstart_routes_from_greedy(
        df=df0, cache=cache, T0_s=T0_s, L_minutes=L_minutes, max_drivers=max_drivers
    )

    with timed("inject_warmstart_routes_simulate", timings):
        added_warm = 0
        for seq in warm_routes:
            if seq in existing:
                continue
            cost, dsum = simulate_route(list(seq), cache, T0_s, L_s)
            cols.append(RouteCol(len(cols), seq, cost, dsum))
            existing.add(seq)
            added_warm += 1

    if added_warm:
        print("[diag] added warm-start routes:", added_warm, "total cols:", len(cols))


    # 2) LP loop
    for it in range(1, lp_iters + 1):
        if time.time() - start_wall > overall_time_limit_s * 0.75:
            print("[CG] stopping LP loop due to overall time budget")
            break

        with timed(f"LP_solve_master_it{it}", timings):
            _, res_lp, z_lp, duals = solve_master(
                n_deliveries=len(df0),
                cols=cols,
                L_s=L_s,
                max_routes=max_drivers,
                solver_name=solver_name,
                time_limit_s=20,
                relax=True,
                want_duals=True,
            )

        if duals is None:
            print("[CG] No duals returned; finishing with current pool.")
            break

        with timed(f"pricing_heuristic_it{it}", timings):
            new = pricing_heuristic(
                df0, duals, T0_s,
                cache=cache, L_s=L_s,
                n_new=pricing_new,
                max_len=7,
                k_nn=knn_init,
                candidates_per_step=pricing_candidates_per_step,
            )

        with timed(f"add_new_cols_simulate_it{it}", timings):
            added = 0
            for seq, rc in new:
                if seq in existing:
                    continue
                if rc > rc_threshold:
                    continue
                cost, dsum = simulate_route(list(seq), cache, T0_s, L_s)
                cols.append(RouteCol(len(cols), seq, cost, dsum))
                existing.add(seq)
                added += 1

        print(f"[CG] it={it} added={added} pool={len(cols)}")
        if added == 0:
            break

        with timed(f"pool_prune_it{it}", timings):
            cols.sort(key=lambda r: (len(r.deliveries), r.cost_s + 0.01 * r.dur_sum_s))
            cols = cols[:pool_cap_after_lp]
            existing = set(c.deliveries for c in cols)

    # 3) integer master
    remaining = max(5, int(overall_time_limit_s - (time.time() - start_wall)))
    with timed("MILP_solve_master_integer", timings):
        m_int, res_int, z_int, _ = solve_master(
            n_deliveries=len(df0),
            cols=cols,
            L_s=L_s,
            max_routes=max_drivers,
            solver_name=solver_name,
            time_limit_s=remaining,
            relax=False,
            want_duals=False,
        )

    chosen = [cols[r] for r, val in enumerate(z_int) if val > 0.5]

    print("\n====== Timing Summary ======")
    total = sum(timings.values())
    for k, v in sorted(timings.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * v / total if total > 0 else 0.0
        print(f"[time] {k:28s} {v:8.3f}s  ({pct:5.1f}%)")
    print(f"[time] TOTAL (instrumented): {total:.3f}s")
    print(f"[time] WALL: {time.time() - start_wall:.3f}s")

    # If the integer master is infeasible (often happens if the initial column pool
    # cannot satisfy the global SLA constraint), fall back to the greedy warm-start
    # solution so we still get a valid, SLA-feasible output.
    if (len(chosen) == 0) and (warm_assignment_df is not None) and (warm_metrics is not None):
        print("[CG] No routes chosen (infeasible with current pool). Returning greedy warm-start solution.")
        if return_details:
            return warm_assignment_df, warm_metrics, {"fallback": "greedy_warmstart"}
        return warm_assignment_df, warm_metrics

    details = {"columns": cols, "chosen": chosen, "model": m_int, "result": res_int}


    assignment_rows: List[Dict[str, float]] = []
    total_route_time_s = 0.0

    for r, col in enumerate(chosen):
        recs, end_t = simulate_route_records(list(col.deliveries), cache, T0_s)
        total_route_time_s += max(0.0, end_t - float(T0_s))
        for rec in recs:
            rec["driver"] = str(r)
            assignment_rows.append(rec)

    assignment_df = pd.DataFrame(assignment_rows)

    avg_duration_s = float(assignment_df["Duration_s"].mean()) if len(assignment_df) else 0.0
    n = int(len(delivery_df))
    deliveries_per_hour = (n * 3600.0 / total_route_time_s) if total_route_time_s > 0 else 0.0

    metrics = {
        "num_deliveries": int(n),
        "num_drivers": int(len(chosen)),
        "avg_duration_min": avg_duration_s / 60.0,
        "avg_duration_s": avg_duration_s,
        "total_route_time_s": float(total_route_time_s),
        "deliveries_per_hour_proxy": float(deliveries_per_hour),
        "runtime_s": time.perf_counter() - t_start,
    }

    if return_details:
        return assignment_df, metrics, details
    return assignment_df, metrics