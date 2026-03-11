#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 14:34:10 2026

@author: anahitamolavi
"""

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

# Modules
from modules import utilities


@dataclass
class driverstate:
    """
    State of a single driver during greedy route construction.
    """
    driver_id: str
    t: float                      # current time (seconds)
    lat: Optional[float]          # latitude of last dropoff
    lon: Optional[float]          # longitude of last dropoff
    end_time: float               # time of last dropoff
    deliveries: List[int]         # indices of assigned deliveries (in order)



def greedy_assign_deliveries(
    deliveries_df: pd.DataFrame,
    num_drivers: int,
    T0_utc: str,
    L_minutes: float = 45.0,
    alpha: float = 1.0,
    speed_mps: float = 4.5,
    time_window_s: int = 3600,
    use_region: bool = True,
    top_k: Optional[int] = 50,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Greedy global assignment heuristic.

    At each iteration:
      - evaluate feasible (driver, delivery) pairs
      - compute incremental route cost + SLA penalty
      - assign the best-scoring pair
    """

    t_start = time.perf_counter()

    df = deliveries_df.copy()

    # ---------------------------- Validate required columns --------------------------- #
    required = [
        "delivery_id",
        "created_at",
        "food_ready_time",
        "pickup_lat",
        "pickup_long",
        "dropoff_lat",
        "dropoff_long",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Optional region constraint
    has_region = "region_id" in df.columns
    if use_region and not has_region:
        use_region = False

    n = len(df)

    created = utilities.to_epoch_s(df["created_at"])
    ready   = utilities.to_epoch_s(df["food_ready_time"])

    T0 = int(pd.to_datetime(T0_utc, utc=True).timestamp())
    L = float(L_minutes) * 60.0  # SLA target in seconds

    p_lat = df["pickup_lat"].astype(float).to_numpy()
    p_lon = df["pickup_long"].astype(float).to_numpy()
    d_lat = df["dropoff_lat"].astype(float).to_numpy()
    d_lon = df["dropoff_long"].astype(float).to_numpy()

    # --------------- Precompute pickup -> dropoff travel time for each delivery ----------------- #
    
    tau_intra = np.zeros(n, dtype=float)
    for i in range(n):
        tau_intra[i] = utilities.travel_time_s(
            p_lat[i], p_lon[i], d_lat[i], d_lon[i], speed_mps
        )

    region = df["region_id"].astype(str).to_numpy() if use_region else None

    # --------------------- Greedy assignment bookkeeping ---------------------------- #
    
    unassigned = set(range(n))
    deadhead_cache: Dict[Tuple[int, int], float] = {}

    def deadhead(i_last: Optional[int], i_next: int) -> float:
        """
        Travel time from last dropoff of a route to the pickup of the next delivery.
        Cached for efficiency.
        """
        if i_last is None:
            return 0.0
        key = (i_last, i_next)
        if key in deadhead_cache:
            return deadhead_cache[key]

        tt = utilities.travel_time_s(
            d_lat[i_last], d_lon[i_last],
            p_lat[i_next], p_lon[i_next],
            speed_mps
        )
        deadhead_cache[key] = tt
        return tt

    # Initialize drivers at global start time T0
    drivers: List[driverstate] = [
        driverstate(
            driver_id=str(j),
            t=float(T0),
            lat=None,
            lon=None,
            end_time=float(T0),
            deliveries=[]
        )
        for j in range(num_drivers)
    ]

    # ----------------------------- Candidate filtering to keep runtime low ------------------------------- #
    
    def candidate_indices_for_driver(ds: driverstate) -> List[int]:
        """
        Reduce candidate deliveries for a given driver:
          - time window filter
          - optional region consistency
          - optional top-K nearest pickups
        """
        if not unassigned:
            return []

        cutoff = ds.t + time_window_s
        cand = [i for i in unassigned if created[i] <= cutoff]

        # Fallback to earliest-created delivery to avoid deadlock
        if not cand:
            cand = [min(unassigned, key=lambda i: created[i])]

        if use_region and ds.deliveries:
            last_i = ds.deliveries[-1]
            same = [i for i in cand if region[i] == region[last_i]]
            if same:
                cand = same

        if top_k is not None and ds.deliveries and len(cand) > top_k:
            last_i = ds.deliveries[-1]
            lat0, lon0 = d_lat[last_i], d_lon[last_i]
            dist_proxy = (p_lat[cand] - lat0) ** 2 + (p_lon[cand] - lon0) ** 2
            idx = np.argpartition(dist_proxy, top_k)[:top_k]
            cand = [cand[k] for k in idx.tolist()]

        return cand

    # ------------------- Main greedy assignment loop -------------------------- #
    
    assignments = []

    while unassigned:
        best = None

        for dj, ds in enumerate(drivers):
            cand = candidate_indices_for_driver(ds)
            if not cand:
                continue

            last_i = ds.deliveries[-1] if ds.deliveries else None

            for i in cand:
                # Compute pickup and dropoff times
                s = max(ds.t + deadhead(last_i, i), ready[i])
                d = s + tau_intra[i]

                # Delivery duration
                u = d - created[i]

                # Incremental route time
                delta = d - ds.t

                # Objective: route efficiency + SLA penalty
                score = delta + alpha * max(0.0, u - L)

                if best is None or score < best[0]:
                    best = (score, dj, i, s, d, u)

        # Commit best assignment
        _, dj, i, s, d, u = best
        ds = drivers[dj]

        unassigned.remove(i)
        ds.deliveries.append(i)
        ds.t = float(d)
        ds.end_time = float(d)
        ds.lat = float(d_lat[i])
        ds.lon = float(d_lon[i])

        assignments.append({
            "delivery_id": df.loc[i, "delivery_id"],
            "driver": ds.driver_id,
            "PickupTime_s": float(s),
            "DropoffTime_s": float(d),
            "Duration_s": float(u),
        })

        if verbose and len(assignments) % 200 == 0:
            print(f"Assigned {len(assignments)}/{n} deliveries")

    # ------------------ Metrics computation ------------------------- #
    
    assignment_df = pd.DataFrame(assignments)

    total_route_time_s = sum(
        max(0.0, ds.end_time - T0)
        for ds in drivers if ds.deliveries
    )

    avg_duration_s = assignment_df["Duration_s"].mean()
    deliveries_per_hour = (
        n * 3600.0 / total_route_time_s if total_route_time_s > 0 else 0.0
    )

    metrics = {
        "num_deliveries": int(n),
        "num_drivers": int(num_drivers),
        "avg_duration_min": avg_duration_s / 60.0,
        "avg_duration_s": avg_duration_s,
        "total_route_time_s": float(total_route_time_s),
        "deliveries_per_hour_proxy": float(deliveries_per_hour),
        "runtime_s": time.perf_counter() - t_start,
    }

    return assignment_df, metrics


# =================================================
# Wrapper: increase drivers until SLA is satisfied
# =================================================

def solve_with_min_drivers(
    deliveries_df: pd.DataFrame,
    T0_utc: str,
    L_minutes: float = 45.0,
    start_drivers: int = 10,
    max_drivers: int = 60,
    drivers_step: int = 5,
    alpha: float = 2.0,
    time_window_s: int = 3600,
    use_region: bool = True,
    top_k: Optional[int] = 50,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run the greedy heuristic with increasing number of drivers
    until average delivery duration satisfies the SLA.
    """
    best_df, best_metrics = None, None

    for m in range(start_drivers, max_drivers + 1, drivers_step):
        df_sol, metrics = greedy_assign_deliveries(
            deliveries_df=deliveries_df,
            num_drivers=m,
            T0_utc=T0_utc,
            L_minutes=L_minutes,
            alpha=alpha,
            time_window_s=time_window_s,
            use_region=use_region,
            top_k=top_k,
            verbose=verbose,
        )
        best_df, best_metrics = df_sol, metrics

        if verbose:
            print(
                f"drivers={m} | "
                f"avg_dur={metrics['avg_duration_min']:.2f} min | "
                f"deliv/hr={metrics['deliveries_per_hour_proxy']:.2f}"
            )

        if metrics["avg_duration_min"] <= L_minutes:
            return best_df, best_metrics

    return best_df, best_metrics
