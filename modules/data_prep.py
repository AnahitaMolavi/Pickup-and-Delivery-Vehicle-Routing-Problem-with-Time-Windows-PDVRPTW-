#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 14:31:43 2026

@author: anahitamolavi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Modules
from modules import utilities


def add_distance_and_time_columns(
    deliveries_df: pd.DataFrame,
    speed_m_per_sec: float,
) -> pd.DataFrame:
    """
    Adds haversine distance (km) and delivery time columns to deliveries_df
    """

    deliveries_df["haversine_km"] = (utilities.haversine(
        deliveries_df["pickup_lat"].values,
        deliveries_df["pickup_long"].values,
        deliveries_df["dropoff_lat"].values,
        deliveries_df["dropoff_long"].values,
    )) / 1000.0

    deliveries_df["delivery_time_sec"] = (
        deliveries_df["haversine_km"] * 1000.0 / speed_m_per_sec
    )

    deliveries_df["delivery_time_min"] = (
        deliveries_df["delivery_time_sec"] / 60.0
    )

    return deliveries_df


def run_initial_eda(
    deliveries_df: pd.DataFrame,
    speed_m_per_sec: float = 4.5,
    show_plots: bool = True,
    verbose: bool = True,
):
    """
    Run initial EDA, cleaning, sanity checks, outlier analysis, and plots
    """

    df = deliveries_df.copy()
    reports = {}

    # ---------------------------- Basic cleaning / types---------------------------- #
    for c in ["created_at", "food_ready_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    num_cols = [c for c in [
        "pickup_lat", "pickup_long", "dropoff_lat", "dropoff_long",
        "haversine_km", "delivery_time_sec", "delivery_time_min"
    ] if c in df.columns]

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "haversine_km" in df.columns and "delivery_time_sec" not in df.columns:
        df["delivery_time_sec"] = df["haversine_km"] * 1000 / speed_m_per_sec

    if "delivery_time_sec" in df.columns and "delivery_time_min" not in df.columns:
        df["delivery_time_min"] = df["delivery_time_sec"] / 60

    if "created_at" in df.columns and "food_ready_time" in df.columns:
        df["ready_gap_min"] = (
            (df["food_ready_time"] - df["created_at"])
            .dt.total_seconds() / 60
        )

    # ---------------------------- Overview ---------------------------- #
    if verbose:
        print("Shape:", df.shape)
        print("\nDtypes:\n", df.dtypes)

    missing_cnt = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing_cnt / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({
        "missing_count": missing_cnt,
        "missing_pct": missing_pct
    })

    reports["missing_summary"] = missing_summary

    if verbose:
        print(
            "\nMissingness (top):\n",
            missing_summary[missing_summary["missing_count"] > 0].head(20),
        )

    key_metrics = [c for c in ["haversine_km", "delivery_time_min", "ready_gap_min"] if c in df.columns]
    if key_metrics:
        stats_tbl = df[key_metrics].describe(
            percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
        ).T
        reports["key_metric_stats"] = stats_tbl
        if verbose:
            print("\nKey metric stats:\n", stats_tbl)

    # ---------------------------- Lat/Lon bounds checks ---------------------------- #
    def latlon_bounds_report(df):
        checks = {}
        if "pickup_lat" in df.columns:
            checks["pickup_lat_out_of_bounds"] = (
                (df["pickup_lat"] < -90) | (df["pickup_lat"] > 90)
            ).sum()
        if "dropoff_lat" in df.columns:
            checks["dropoff_lat_out_of_bounds"] = (
                (df["dropoff_lat"] < -90) | (df["dropoff_lat"] > 90)
            ).sum()
        if "pickup_long" in df.columns:
            checks["pickup_long_out_of_bounds"] = (
                (df["pickup_long"] < -180) | (df["pickup_long"] > 180)
            ).sum()
        if "dropoff_long" in df.columns:
            checks["dropoff_long_out_of_bounds"] = (
                (df["dropoff_long"] < -180) | (df["dropoff_long"] > 180)
            ).sum()
        return checks

    reports["latlon_bounds"] = latlon_bounds_report(df)
    if verbose:
        print("\nLat/Lon bounds issues:", reports["latlon_bounds"])

    # ---------------------------- Outlier analysis (IQR) ---------------------------- #
    def iqr_outlier_flags(s: pd.Series, k: float = 1.5):
        s = s.dropna()
        if s.empty:
            return None, None, None
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        idx = s[(s < lo) | (s > hi)].index
        return lo, hi, idx

    outlier_report = {}
    for c in ["haversine_km", "delivery_time_min", "ready_gap_min"]:
        if c in df.columns:
            lo, hi, idx = iqr_outlier_flags(df[c])
            if idx is not None:
                outlier_report[c] = {
                    "iqr_lo": lo,
                    "iqr_hi": hi,
                    "outlier_count": len(idx),
                }

    reports["outliers"] = outlier_report

    if verbose and outlier_report:
        print("\nIQR outlier report:")
        for k, v in outlier_report.items():
            print(
                f"  {k}: outliers={v['outlier_count']}, "
                f"bounds=({v['iqr_lo']:.3g}, {v['iqr_hi']:.3g})"
            )

    # ---------------------------- Plots ---------------------------- #
    if show_plots:
        def hist_bins(x):
            x = pd.Series(x).dropna()
            if len(x) < 2:
                return 10
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                return min(30, max(10, int(np.sqrt(len(x)))))
            bw = 2 * iqr / (len(x) ** (1 / 3))
            bins = int(np.ceil((x.max() - x.min()) / bw)) if bw > 0 else 10
            return max(10, min(60, bins))

        miss_plot = (
            missing_summary[missing_summary["missing_count"] > 0]
            .sort_values("missing_pct", ascending=False)
            .head(15)
        )

        if not miss_plot.empty:
            plt.figure()
            plt.bar(miss_plot.index.astype(str), miss_plot["missing_pct"])
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Missing (%)")
            plt.title("Top Missing Columns")
            plt.tight_layout()
            plt.show()

        if "haversine_km" in df.columns:
            x = df["haversine_km"]
            plt.figure()
            plt.hist(x.dropna(), bins=hist_bins(x), edgecolor="black")
            plt.xlabel("Haversine distance (km)")
            plt.ylabel("Count")
            plt.title("Distance Distribution")
            plt.tight_layout()
            plt.show()

        if "delivery_time_min" in df.columns:
            t = df["delivery_time_min"]
            plt.figure()
            plt.hist(t.dropna(), bins=hist_bins(t), edgecolor="black")
            plt.xlabel("Delivery time (min)")
            plt.ylabel("Count")
            plt.title("Delivery Time Distribution")
            plt.tight_layout()
            plt.show()


    suspicious = pd.DataFrame(index=df.index)

    if "haversine_km" in df.columns:
        suspicious["dist_neg_or_zero"] = df["haversine_km"] <= 0
        suspicious["dist_big_50km"] = df["haversine_km"] > 50

    if "delivery_time_min" in df.columns:
        suspicious["time_neg_or_zero"] = df["delivery_time_min"] <= 0
        suspicious["time_big_180min"] = df["delivery_time_min"] > 180

    if "ready_gap_min" in df.columns:
        suspicious["ready_gap_neg"] = df["ready_gap_min"] < 0
        suspicious["ready_gap_big_180min"] = df["ready_gap_min"] > 180

    flag_cols = [c for c in suspicious.columns if suspicious[c].any()]
    flagged = df.loc[suspicious[flag_cols].any(axis=1)] if flag_cols else pd.DataFrame()

    reports["flagged_rows"] = flagged

    if verbose:
        print(
            f"\nFlagged suspicious rows: {len(flagged)}"
            if not flagged.empty else
            "\nNo suspicious rows flagged by simple rules."
        )

    return df, reports

