#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 14:33:34 2026

@author: anahitamolavi
"""

# Standard library
import sys
import time
import math
import logging
from contextlib import contextmanager
from typing import Dict, Tuple, Optional

# Third-party libraries
import numpy as np
import pandas as pd


# -------------------- Logging + Timing -------------------- #

import warnings
warnings.filterwarnings(
    "ignore",
    message="Could not infer format, so each element will be parsed individually",
    category=UserWarning,
)


def configure_logging(level: int = logging.INFO) -> None:
    """
    logging:
    - streams to notebook cell output (stdout)
    """
    root = logging.getLogger()
    root.setLevel(level)

    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    logging.info("Logging configured (level=%s).", logging.getLevelName(level))
    sys.stdout.flush()
    
@contextmanager
def log_step(step_name: str, level: int = logging.INFO):
    """
    Context manager that logs start/end and elapsed time, flushing stdout
    """
    start = time.perf_counter()
    logging.log(level, f"[START] {step_name}")
    sys.stdout.flush()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.log(level, f"[END]   {step_name} | elapsed={elapsed:.3f}s")
        sys.stdout.flush()

# -------------------- Geometry / time helpers -------------------- #

def haversine(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between (lat1, lon1) and (lat2, lon2) in meters.

    - Inputs can be scalars or NumPy arrays / pandas Series
    - Output is float or NumPy array
    """
    R_m = 6371000.0  # meters
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * R_m * np.arcsin(np.sqrt(a))


def to_epoch_s(x) -> np.ndarray | int:
    """
    Convert datetime-like input(s) to UNIX epoch seconds (UTC).

    Accepts:
      - pandas Series / Index / array-like of datetimes -> returns np.ndarray[int64]
      - scalar datetime / Timestamp / ISO string -> returns int

    Notes:
      - Naive datetimes are treated as UTC.
      - For arrays/Series, missing values become 0.
    """
    # Vectorized path (Series/Index/array)
    if isinstance(x, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        dt = pd.to_datetime(x, utc=True, errors="coerce")
        secs = (dt.astype("int64") // 10**9).to_numpy(dtype="int64", copy=False)
        # pandas uses NaT -> min int; coerce to 0 for safety
        secs = np.where(secs < 0, 0, secs).astype("int64", copy=False)
        return secs

    # Scalar path
    dt = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(dt):
        return 0
    return int(dt.value // 10**9)


def travel_time_s(lat1, lon1, lat2, lon2, speed_m_per_sec: float = 4.5) -> float:
    """Travel time in seconds using haversine distance (meters) and constant speed (m/s)."""
    if speed_m_per_sec <= 0:
        raise ValueError("speed_m_per_sec must be positive.")
    return float(haversine(lat1, lon1, lat2, lon2)) / float(speed_m_per_sec)



# -------------------- Output Creation ------------------- #

def compute_required_metrics(assignment_df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes the TWO required metrics:
      1) Average deliveries/hour across all drivers
      2) Average delivery duration across all deliveries (min)
    """

    # Average delivery duration across all deliveries
    avg_delivery_duration_min = assignment_df["Duration_s"].mean() / 60.0

    # Average deliveries/hour across all drivers
    T0_s = int(pd.to_datetime("2015-02-03 02:00:00", utc=True).timestamp())

    route_end_by_driver = assignment_df.groupby("driver")["DropoffTime_s"].max()

    total_route_time_s = float(
        (route_end_by_driver - T0_s)
        .clip(lower=0)
        .sum()
    )

    avg_deliveries_per_hour = (
        len(assignment_df) * 3600.0 / total_route_time_s
        if total_route_time_s > 0 else 0.0
    )

    return {
        "avg_deliveries_per_hour": float(avg_deliveries_per_hour),
        "avg_delivery_duration_min": float(avg_delivery_duration_min),
        "total_route_time_s": float(total_route_time_s),
    }


def print_required_metrics(metrics: Dict[str, float]) -> None:
    """Print the TWO REQUIRED METRICS"""
    print("\n=== Requested Output Metrics ===")
    print(f"Average deliveries/hour across all drivers: {metrics['avg_deliveries_per_hour']:.3f}")
    print(f"Average delivery duration across all deliveries (min): {metrics['avg_delivery_duration_min']:.3f}")


def build_submission_csv(assignment_df: pd.DataFrame) -> pd.DataFrame:
    # Route ID, Route Point Index, Delivery ID, Route Point Type, Route Point Time
    
    rows = []

    for route_id, g in assignment_df.groupby("driver", sort=True):
        g2 = g.sort_values(["PickupTime_s", "DropoffTime_s"]).reset_index(drop=True)

        rp_idx = 0
        for _, r in g2.iterrows():
            rows.append({
                "Route ID": route_id,
                "Route Point Index": rp_idx,
                "Delivery ID": r["delivery_id"],
                "Route Point Type": "Pickup",
                "Route Point Time": int(round(float(r["PickupTime_s"]))),
            })
            rp_idx += 1

            rows.append({
                "Route ID": route_id,
                "Route Point Index": rp_idx,
                "Delivery ID": r["delivery_id"],
                "Route Point Type": "DropOff",
                "Route Point Time": int(round(float(r["DropoffTime_s"]))),
            })
            rp_idx += 1

    submission_df = pd.DataFrame(rows, columns=[
        "Route ID", "Route Point Index", "Delivery ID", "Route Point Type", "Route Point Time"
    ])

    submission_df["Route Point Type"] = (
        submission_df["Route Point Type"]
        .astype(str)
        .str.strip()           
    )

    allowed = {"Pickup", "DropOff"}
    bad_vals = set(submission_df["Route Point Type"].unique()) - allowed
    assert not bad_vals, f"Invalid Route Point Type values found: {bad_vals}"

    submission_df["Route Point Type"] = submission_df["Route Point Type"].astype(str).str.strip()
    assert set(submission_df["Route Point Type"].unique()) <= {"Pickup", "DropOff"}

    return submission_df

def create_outputs(
    assignment_df: pd.DataFrame,
    output_csv_path: Optional[str] = None,
    print_metrics: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:


    # Basic sanity checks
    required_cols = {"Duration_s", "driver", "DropoffTime_s", "PickupTime_s", "delivery_id"}
    missing = required_cols - set(assignment_df.columns)
    if missing:
        raise ValueError(f"assignment_df is missing required columns: {sorted(missing)}")

    metrics = compute_required_metrics(assignment_df)

    if print_metrics:
        print_required_metrics(metrics)

    submission_df = build_submission_csv(assignment_df)

    if output_csv_path:
        submission_df.to_csv(output_csv_path, index=False)
        print(f"\nWrote submission CSV: {output_csv_path}")


    return submission_df, metrics

