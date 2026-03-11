#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 13:03:03 2026

@author: anahitamolavi
"""

import math
import time
from contextlib import contextmanager
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# Modules
from modules import utilities

SPEED_M_PER_SEC = 4.5

# -------------------- Helpers -------------------- #
@contextmanager
def timed_step(name: str, *, enabled: bool = True, print_fn=print):
    """
    To print wall time
    """
    if not enabled:
        yield
        return
    t0 = time.perf_counter()
    print_fn(f"[TIMER] {name} ...")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print_fn(f"[TIMER] {name} done in {dt:.3f}s")

class Timer:
    """
    Optional finer-grain timer
    """
    def __init__(self, *, enabled: bool = True, print_fn=print):
        self.enabled = enabled
        self.print_fn = print_fn
        self._t0 = time.perf_counter()
        self._last = self._t0

    def lap(self, label: str):
        if not self.enabled:
            return
        now = time.perf_counter()
        self.print_fn(f"[TIMER]   - {label}: {now - self._last:.3f}s (since last), {now - self._t0:.3f}s (total)")
        self._last = now


# ----------------------------- Step 1: Lower-bound screen -------------------- #

def lower_bound_avg_duration_minutes(
    deliveries_df: pd.DataFrame,
    *,
    col_created="created_at",
    col_ready="food_ready_time",
    col_plat="pickup_lat",
    col_plon="pickup_long",
    col_dlat="dropoff_lat",
    col_dlon="dropoff_long",
    speed_m_per_sec=SPEED_M_PER_SEC,
) -> float:
    df = deliveries_df.copy()
    df[col_created] = pd.to_datetime(df[col_created], utc=True)
    df[col_ready] = pd.to_datetime(df[col_ready], utc=True)

    created_s = utilities.to_epoch_s(df[col_created])
    ready_s = utilities.to_epoch_s(df[col_ready])

    p_lat = df[col_plat].to_numpy()
    p_lon = df[col_plon].to_numpy()
    d_lat = df[col_dlat].to_numpy()
    d_lon = df[col_dlon].to_numpy()

    trip_s = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        trip_s[i] = utilities.travel_time_s(p_lat[i], p_lon[i], d_lat[i], d_lon[i], speed_m_per_sec)

    wait_s = np.maximum(0.0, ready_s - created_s).astype(float)
    lb_s = wait_s + trip_s
    return float(lb_s.mean() / 60.0)


# ----------------------------- Greedy feasibility ----------------------- #

def greedy_feasibility_schedule(
    deliveries_df: pd.DataFrame,
    *,
    K: int,
    T0_utc="2015-02-03 02:00:00+00:00",
    order_sort="slack",
    target_avg_min=45.0,
    col_id="delivery_id",
    col_created="created_at",
    col_ready="food_ready_time",
    col_plat="pickup_lat",
    col_plon="pickup_long",
    col_dlat="dropoff_lat",
    col_dlon="dropoff_long",
    speed_m_per_sec=SPEED_M_PER_SEC,
) -> Dict[str, Any]:
    df = deliveries_df.copy()
    df[col_created] = pd.to_datetime(df[col_created], utc=True)
    df[col_ready] = pd.to_datetime(df[col_ready], utc=True)
    T0_s = int(pd.Timestamp(T0_utc).timestamp())

    created_s = utilities.to_epoch_s(df[col_created])
    ready_s = utilities.to_epoch_s(df[col_ready])

    p_lat = df[col_plat].to_numpy(dtype=float)
    p_lon = df[col_plon].to_numpy(dtype=float)
    d_lat = df[col_dlat].to_numpy(dtype=float)
    d_lon = df[col_dlon].to_numpy(dtype=float)

    pd_trip_s = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        pd_trip_s[i] = utilities.travel_time_s(p_lat[i], p_lon[i], d_lat[i], d_lon[i], speed_m_per_sec)

    lb_s = np.maximum(0.0, ready_s - created_s).astype(float) + pd_trip_s
    slack_s = (target_avg_min * 60.0) - lb_s

    idx = np.arange(len(df))
    if order_sort == "slack":
        idx = idx[np.lexsort((created_s[idx], slack_s[idx]))]
    elif order_sort == "created":
        idx = idx[np.argsort(created_s)]
    elif order_sort == "ready":
        idx = idx[np.argsort(ready_s)]
    else:
        raise ValueError("order_sort must be one of: 'slack', 'created', 'ready'")

    driver_time = np.full(K, float(T0_s))
    driver_lat = np.full(K, np.nan)
    driver_lon = np.full(K, np.nan)
    driver_has_loc = np.zeros(K, dtype=bool)

    rows = []
    for i in idx:
        best_k = None
        best_dropoff = float("inf")

        for k in range(K):
            to_pickup_s = 0.0 if not driver_has_loc[k] else utilities.travel_time_s(
                driver_lat[k], driver_lon[k], p_lat[i], p_lon[i], speed_m_per_sec
            )
            pickup_arrival_s = driver_time[k] + to_pickup_s
            pickup_time_s = max(pickup_arrival_s, float(ready_s[i]))
            dropoff_time_s = pickup_time_s + float(pd_trip_s[i])

            if dropoff_time_s < best_dropoff:
                best_dropoff = dropoff_time_s
                best_k = k

        k = best_k
        to_pickup_s = 0.0 if not driver_has_loc[k] else utilities.travel_time_s(
            driver_lat[k], driver_lon[k], p_lat[i], p_lon[i], speed_m_per_sec
        )
        pickup_arrival_s = driver_time[k] + to_pickup_s
        pickup_time_s = max(pickup_arrival_s, float(ready_s[i]))
        dropoff_time_s = pickup_time_s + float(pd_trip_s[i])

        driver_time[k] = dropoff_time_s
        driver_lat[k] = d_lat[i]
        driver_lon[k] = d_lon[i]
        driver_has_loc[k] = True

        duration_s = dropoff_time_s - float(created_s[i])

        rows.append({
            "Delivery ID": df.iloc[i][col_id] if col_id in df.columns else int(i),
            "driver": int(k),
            "PickupTime_s": int(round(pickup_time_s)),
            "DropoffTime_s": int(round(dropoff_time_s)),
            "Duration_s": float(duration_s),
        })

    assignment_df = pd.DataFrame(rows)

    avg_delivery_duration_min = float(assignment_df["Duration_s"].mean() / 60.0)
    last_dropoff_by_driver = assignment_df.groupby("driver")["DropoffTime_s"].max()
    total_route_time_s = float((last_dropoff_by_driver - T0_s).clip(lower=0).sum())
    avg_deliveries_per_hour = float(len(assignment_df) * 3600.0 / total_route_time_s) if total_route_time_s > 0 else 0.0

    metrics = {
        "K_drivers": int(K),
        "avg_delivery_duration_min": avg_delivery_duration_min,
        "avg_deliveries_per_hour": avg_deliveries_per_hour,
        "total_route_time_hours": total_route_time_s / 3600.0,
    }

    feasible = (avg_delivery_duration_min <= target_avg_min)

    return {"assignment_df": assignment_df, "metrics": metrics, "feasible": feasible}


# ----------------------------- Step 3: Min-K search (closest to 45) ----------------------------- #

def _pick_best_close_to_target(candidates, target_avg_min=45.0):
    feas = [c for c in candidates if c["feasible"]]
    if not feas:
        return None
    return max(feas, key=lambda c: c["metrics"]["avg_delivery_duration_min"])

def eval_K_with_variants(deliveries_df, K, *, variants=5, target_avg_min=45.0, speed_m_per_sec=4.5, **kwargs):
    outs = []
    outs.append(greedy_feasibility_schedule(deliveries_df, K=K, order_sort="slack",
                                           target_avg_min=target_avg_min, speed_m_per_sec=speed_m_per_sec, **kwargs))
    outs.append(greedy_feasibility_schedule(deliveries_df, K=K, order_sort="created",
                                           target_avg_min=target_avg_min, speed_m_per_sec=speed_m_per_sec, **kwargs))
    outs.append(greedy_feasibility_schedule(deliveries_df, K=K, order_sort="ready",
                                           target_avg_min=target_avg_min, speed_m_per_sec=speed_m_per_sec, **kwargs))

    rng = np.random.default_rng(42 + K)
    for _ in range(max(0, variants - 3)):
        shuffled = deliveries_df.sample(frac=1.0, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)
        outs.append(greedy_feasibility_schedule(shuffled, K=K, order_sort="slack",
                                               target_avg_min=target_avg_min, speed_m_per_sec=speed_m_per_sec, **kwargs))

    best = _pick_best_close_to_target(outs, target_avg_min=target_avg_min)
    return {"best": best, "all": outs, "feasible": (best is not None)}

def find_min_K_feasible_close_target(
    deliveries_df,
    *,
    K_low=10,
    K_high=200,
    coarse_step=5,
    refine_window=10,
    variants_per_K=5,
    target_avg_min=45.0,
    speed_m_per_sec=4.5,
    **kwargs
) -> Dict[str, Any]:
    cache = {}

    def get(K):
        if K not in cache:
            cache[K] = eval_K_with_variants(deliveries_df, K, variants=variants_per_K,
                                            target_avg_min=target_avg_min, speed_m_per_sec=speed_m_per_sec, **kwargs)
        return cache[K]

    first_feasible = None
    for K in range(K_low, K_high + 1, coarse_step):
        if get(K)["feasible"]:
            first_feasible = K
            break

    if first_feasible is None:
        return {"status": "not_found_in_range", "cache": cache}

    start = max(K_low, first_feasible - refine_window)
    end = min(K_high, first_feasible + refine_window)

    for K in range(start, end + 1):
        r = get(K)
        if r["feasible"]:
            return {"status": "found", "min_K": int(K), "best_solution": r["best"], "cache": cache}

    r = get(first_feasible)
    return {"status": "found", "min_K": int(first_feasible), "best_solution": r["best"], "cache": cache}


# ----------------------------- Step 4: Fleet compression (52 -> 50 etc.) ----------------------------- #

def rebuild_schedule_for_one_driver(
    deliveries_df,
    delivery_ids,
    *,
    T0_utc="2015-02-03 02:00:00+00:00",
    col_id="delivery_id",
    col_created="created_at",
    col_ready="food_ready_time",
    col_plat="pickup_lat",
    col_plon="pickup_long",
    col_dlat="dropoff_lat",
    col_dlon="dropoff_long",
    speed_m_per_sec=4.5,
):
    df = deliveries_df.copy()
    df[col_created] = pd.to_datetime(df[col_created], utc=True)
    df[col_ready] = pd.to_datetime(df[col_ready], utc=True)
    T0_s = int(pd.Timestamp(T0_utc).timestamp())

    sub = df[df[col_id].isin(delivery_ids)].copy()
    if sub.empty:
        return pd.DataFrame(columns=[col_id, "PickupTime_s", "DropoffTime_s", "Duration_s"])

    created_s = utilities.to_epoch_s(sub[col_created])
    ready_s = utilities.to_epoch_s(sub[col_ready])

    p_lat = sub[col_plat].to_numpy()
    p_lon = sub[col_plon].to_numpy()
    d_lat = sub[col_dlat].to_numpy()
    d_lon = sub[col_dlon].to_numpy()

    pd_trip_s = np.empty(len(sub), dtype=float)
    for i in range(len(sub)):
        pd_trip_s[i] = utilities.travel_time_s(p_lat[i], p_lon[i], d_lat[i], d_lon[i], speed_m_per_sec)

    remaining = list(range(len(sub)))
    cur_t = float(T0_s)
    cur_has_loc = False
    cur_lat = cur_lon = np.nan

    out_rows = []
    while remaining:
        best_j = None
        best_drop = float("inf")
        best_pick = None

        for j in remaining:
            to_pick = 0.0 if not cur_has_loc else utilities.travel_time_s(cur_lat, cur_lon, p_lat[j], p_lon[j], speed_m_per_sec)
            pick_arr = cur_t + to_pick
            pick_t = max(pick_arr, float(ready_s[j]))
            drop_t = pick_t + float(pd_trip_s[j])
            if drop_t < best_drop:
                best_drop = drop_t
                best_j = j
                best_pick = pick_t

        j = best_j
        cur_t = best_drop
        cur_lat, cur_lon = d_lat[j], d_lon[j]
        cur_has_loc = True

        dur_s = cur_t - float(created_s[j])
        out_rows.append({
            col_id: sub.iloc[j][col_id],
            "PickupTime_s": int(round(best_pick)),
            "DropoffTime_s": int(round(cur_t)),
            "Duration_s": float(dur_s),
        })
        remaining.remove(j)

    return pd.DataFrame(out_rows)

def compress_fleet(
    deliveries_df,
    assignment_df,
    *,
    target_K=50,
    target_avg_min=45.0,
    col_id="delivery_id",
    speed_m_per_sec=4.5,
    rebuild_kwargs=None
):
    if rebuild_kwargs is None:
        rebuild_kwargs = {}

    work = assignment_df.copy()
    work = work.rename(columns={"Delivery ID": col_id})
    if col_id not in work.columns:
        raise ValueError(f"assignment_df must have 'Delivery ID' or '{col_id}'")

    def current_metrics(df_assign):
        return float(df_assign["Duration_s"].mean() / 60.0), int(df_assign["driver"].nunique())

    avg_min, used = current_metrics(work)
    if used <= target_K and avg_min <= target_avg_min:
        return work, {"status": "already_ok", "avg_duration_min": avg_min, "used_drivers": used, "eliminated_drivers": []}

    driver_to_delivs = work.groupby("driver")[col_id].apply(list).to_dict()
    drivers_sorted = sorted(driver_to_delivs.keys(), key=lambda d: len(driver_to_delivs[d]))

    eliminated = []
    for d_remove in drivers_sorted:
        if used <= target_K:
            break

        remove_delivs = driver_to_delivs.get(d_remove, [])
        if not remove_delivs:
            continue

        recipients = [d for d in driver_to_delivs.keys() if d != d_remove and len(driver_to_delivs[d]) > 0]
        if not recipients:
            continue

        tmp = work[work["driver"] != d_remove].copy()

        for deliv_id in remove_delivs:
            best_updated_tmp = None
            best_avg = float("inf")

            for d_rec in recipients:
                rec_delivs = tmp[tmp["driver"] == d_rec][col_id].tolist()
                new_list = rec_delivs + [deliv_id]

                rebuild_kwargs_clean = dict(rebuild_kwargs)
                rebuild_kwargs_clean.pop("col_id", None)

                rebuilt = rebuild_schedule_for_one_driver(
                    deliveries_df, new_list,
                    col_id=col_id,
                    speed_m_per_sec=speed_m_per_sec,
                    **rebuild_kwargs_clean
                )

                tmp2 = tmp[tmp["driver"] != d_rec].copy()
                rebuilt2 = rebuilt.copy()
                rebuilt2["driver"] = d_rec
                tmp2 = pd.concat([tmp2, rebuilt2], ignore_index=True)

                avg2, used2 = current_metrics(tmp2)
                if avg2 <= target_avg_min and used2 <= used:
                    if avg2 < best_avg:
                        best_avg = avg2
                        best_updated_tmp = tmp2

            if best_updated_tmp is None:
                tmp = None
                break

            tmp = best_updated_tmp

        if tmp is not None and len(tmp) == len(work):
            work = tmp
            eliminated.append(d_remove)
            avg_min, used = current_metrics(work)
            driver_to_delivs = work.groupby("driver")[col_id].apply(list).to_dict()

    avg_min, used = current_metrics(work)
    return work, {
        "status": "done",
        "avg_duration_min": avg_min,
        "used_drivers": used,
        "eliminated_drivers": eliminated,
    }


def compute_metrics(
    assignment_df: pd.DataFrame,
    T0_utc: str,
    *,
    driver_col: str = "driver",
    duration_s_col: str = "Duration_s",
    dropoff_time_s_col: str = "DropoffTime_s",
) -> Dict[str, float]:
    
    if assignment_df is None or len(assignment_df) == 0:
        return {"avg_duration_min": float("inf"), "deliveries_per_hour_proxy": 0.0}

    if duration_s_col not in assignment_df.columns:
        raise KeyError(f"Expected column '{duration_s_col}' in assignment_df.")
    avg_duration_min = float(assignment_df[duration_s_col].mean() / 60.0)

    for c in (driver_col, dropoff_time_s_col):
        if c not in assignment_df.columns:
            raise KeyError(f"Expected column '{c}' in assignment_df.")

    T0_s = int(pd.to_datetime(T0_utc, utc=True).timestamp())

    route_end_by_driver = assignment_df.groupby(driver_col)[dropoff_time_s_col].max()
    total_route_time_s = float((route_end_by_driver - T0_s).clip(lower=0).sum())

    deliveries_per_hour_proxy = (
        float(len(assignment_df) * 3600.0 / total_route_time_s)
        if total_route_time_s > 0
        else 0.0
    )

    return {
        "avg_duration_min": avg_duration_min,
        "deliveries_per_hour_proxy": deliveries_per_hour_proxy,
    }


# ----------------------------- Wrapper for the whole bonus flow ----------------------------- #

def run_bonus_feasibility_and_min_fleet(
    deliveries_df: pd.DataFrame,
    T0_utc: str,
    *,
    target_avg_min: float = 45.0,
    target_K: int = 50,
    K_low: int = 10,
    K_high: int = 200,
    coarse_step: int = 5,
    refine_window: int = 10,
    variants_per_K: int = 5,
    speed_m_per_sec: float = 4.5,
    verbose: bool = True,
    # timing
    timing: bool = True,
    # column config
    col_id: str = "delivery_id",
    col_created: str = "created_at",
    col_ready: str = "food_ready_time",
    col_plat: str = "pickup_lat",
    col_plon: str = "pickup_long",
    col_dlat: str = "dropoff_lat",
    col_dlon: str = "dropoff_long",
) -> Tuple[pd.DataFrame, Dict[str, float]]:

    col_kwargs = dict(
        col_id=col_id,
        col_created=col_created,
        col_ready=col_ready,
        col_plat=col_plat,
        col_plon=col_plon,
        col_dlat=col_dlat,
        col_dlon=col_dlon,
    )

    timer = Timer(enabled=timing and verbose)

    # ------------------------- Step 1: LB feasibility screen ------------------------- #
    
    with timed_step("Step 1 - Lower-bound feasibility screen", enabled=timing and verbose):
        lb_mean_min = lower_bound_avg_duration_minutes(
            deliveries_df,
            speed_m_per_sec=speed_m_per_sec,
            **{k: v for k, v in col_kwargs.items() if k != "col_id"}  # LB doesn't use id
        )
        if verbose:
            print(f"LB mean duration: {lb_mean_min:.2f} min")
        timer.lap("computed LB mean")

    if lb_mean_min >= target_avg_min:
        empty_assignment_df = deliveries_df.head(0).copy()
        metrics = {"avg_duration_min": float("inf"), "deliveries_per_hour_proxy": 0.0}
        return empty_assignment_df, metrics

    # ------------------------- Step 2+3: find min K by greedy variants ------------------------- #
    
    with timed_step("Step 2+3 - Min-K search via greedy variants", enabled=timing and verbose):
        search = find_min_K_feasible_close_target(
            deliveries_df,
            K_low=K_low,
            K_high=K_high,
            coarse_step=coarse_step,
            refine_window=refine_window,
            variants_per_K=variants_per_K,
            target_avg_min=target_avg_min,
            speed_m_per_sec=speed_m_per_sec,
            **col_kwargs
        )
        timer.lap("completed K-search")

    if search.get("status") != "found":
        empty_assignment_df = deliveries_df.head(0).copy()
        metrics = {"avg_duration_min": float("inf"), "deliveries_per_hour_proxy": 0.0}
        return empty_assignment_df, metrics

    sol = search["best_solution"]
    minK = search["min_K"]

    if verbose:
        print("found")
        print("min_K:", minK)
        try:
            print("avg_duration_min:", sol["metrics"]["avg_delivery_duration_min"])
            print("avg_deliveries_per_hour:", sol["metrics"]["avg_deliveries_per_hour"])
        except Exception:
            pass

    assignment_df = sol["assignment_df"]

    # ------------------------- Step 4: Compress fleet toward target_K ---------------------- #
    
    with timed_step("Step 4 - Fleet compression toward target_K", enabled=timing and verbose):
        used_drivers = int(assignment_df["driver"].nunique()) if len(assignment_df) else 0
        final_df = assignment_df

        if used_drivers > target_K:
            compressed_df, compression_info = compress_fleet(
                deliveries_df,
                assignment_df,
                target_K=target_K,
                target_avg_min=target_avg_min,
                col_id=col_id,
                speed_m_per_sec=speed_m_per_sec,
                rebuild_kwargs=col_kwargs,
            )

            if verbose:
                print("compression:", compression_info)

            # prefer compressed if it achieves BOTH target_K and SLA
            if compressed_df is not None and len(compressed_df) > 0:
                avg_min = float(compressed_df["Duration_s"].mean() / 60.0)
                used = int(compressed_df["driver"].nunique())
                if used <= target_K and avg_min <= target_avg_min:
                    final_df = compressed_df

        timer.lap("compression finished")

    # ------------------------- Compute metrics and return ------------------ #
    with timed_step("Final - Compute metrics", enabled=timing and verbose):
        metrics = compute_metrics(final_df, T0_utc=T0_utc)
        timer.lap("metrics computed")

    return final_df, metrics
