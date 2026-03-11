import os
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_solution_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"Route ID", "Route Point Index", "Delivery ID", "Route Point Type", "Route Point Time"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def summarize_solution(sol: pd.DataFrame) -> Dict[str, Any]:
    sol = sol.copy()
    sol["Route Point Time"] = pd.to_numeric(sol["Route Point Time"], errors="coerce")
    sol = sol.dropna(subset=["Route Point Time"])
    sol["Route Point Time"] = sol["Route Point Time"].astype(int)

    piv = (
        sol.pivot_table(
            index=["Route ID", "Delivery ID"],
            columns="Route Point Type",
            values="Route Point Time",
            aggfunc="min",
        )
        .reset_index()
    )

    if "Pickup" not in piv.columns:
        piv["Pickup"] = np.nan
    if "Dropoff" not in piv.columns:
        piv["Dropoff"] = np.nan

    deliv = piv.dropna(subset=["Pickup", "Dropoff"]).copy()
    deliv["duration_s"] = (deliv["Dropoff"] - deliv["Pickup"]).clip(lower=0)

    route_span = (
        sol.groupby("Route ID")["Route Point Time"]
        .agg(start="min", end="max")
        .reset_index()
    )
    route_span["span_s"] = (route_span["end"] - route_span["start"]).clip(lower=0)

    deliveries_per_route = (
        deliv.groupby("Route ID")["Delivery ID"]
        .nunique()
        .rename("n_deliveries")
        .reset_index()
    )

    metrics = {
        "n_routes": int(sol["Route ID"].nunique()),
        "n_deliveries_in_solution": int(sol["Delivery ID"].nunique()),
        "avg_delivery_duration_min": float(deliv["duration_s"].mean() / 60.0) if len(deliv) else float("nan"),
        "p90_delivery_duration_min": float(np.percentile(deliv["duration_s"], 90) / 60.0) if len(deliv) else float("nan"),
        "avg_deliveries_per_route": float(deliveries_per_route["n_deliveries"].mean()) if len(deliveries_per_route) else float("nan"),
        "total_route_span_hr": float(route_span["span_s"].sum() / 3600.0) if len(route_span) else float("nan"),
    }

    return {
        "sol": sol,
        "deliveries": deliv,
        "route_span": route_span,
        "deliveries_per_route": deliveries_per_route,
        "metrics": metrics,
    }


def plot_solution_summary(
    deliv: pd.DataFrame,
    dpr: pd.DataFrame,
    title_prefix: str,
    *,
    show: bool = True,
    save_dir: Optional[str] = None,
):
    """Plots: duration histogram, duration CDF, deliveries/route histogram."""
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    def _savefig(name: str):
        if save_dir:
            plt.savefig(os.path.join(save_dir, name), bbox_inches="tight", dpi=180)

    # 1) Delivery duration distribution + CDF
    if len(deliv):
        plt.figure()
        plt.hist(deliv["duration_s"] / 60.0, bins=30)
        plt.xlabel("Delivery duration (min)")
        plt.ylabel("Count")
        plt.title(f"{title_prefix} — Delivery Duration Distribution")
        _savefig(f"{title_prefix}_duration_hist.png")
        if show:
            plt.show()
        else:
            plt.close()

        x = np.sort(deliv["duration_s"].values / 60.0)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.figure()
        plt.plot(x, y)
        plt.xlabel("Delivery duration (min)")
        plt.ylabel("CDF")
        plt.title(f"{title_prefix} — Delivery Duration CDF")
        plt.grid(True, alpha=0.3)
        _savefig(f"{title_prefix}_duration_cdf.png")
        if show:
            plt.show()
        else:
            plt.close()

    # 2) Deliveries per route
    if len(dpr):
        plt.figure()
        max_n = int(dpr["n_deliveries"].max())
        bins = np.arange(1, max_n + 2) - 0.5
        plt.hist(dpr["n_deliveries"], bins=bins)
        plt.xlabel("Deliveries per route")
        plt.ylabel("Count of routes")
        plt.title(f"{title_prefix} — Deliveries per Route")
        _savefig(f"{title_prefix}_deliveries_per_route.png")
        if show:
            plt.show()
        else:
            plt.close()


def show_metrics_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for k, v in results.items():
        m = v["metrics"]
        rows.append(
            {
                "Solution": k,
                "Routes (#)": m["n_routes"],
                "Deliveries (#)": m["n_deliveries_in_solution"],
                "Avg duration (min)": round(m["avg_delivery_duration_min"], 2) if pd.notnull(m["avg_delivery_duration_min"]) else np.nan,
                "P90 duration (min)": round(m["p90_delivery_duration_min"], 2) if pd.notnull(m["p90_delivery_duration_min"]) else np.nan,
                "Avg deliveries/route": round(m["avg_deliveries_per_route"], 2) if pd.notnull(m["avg_deliveries_per_route"]) else np.nan,
                "Total route span (hr)": round(m["total_route_span_hr"], 2) if pd.notnull(m["total_route_span_hr"]) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------- Wrapper ----------------------------- #

def run_solution_dashboard(
    outputs: Optional[Dict[str, str]] = None,
    *,
    base_dir: Optional[str] = None,
    show_plots: bool = True,
    save_plots_dir: Optional[str] = None,
    display_table: bool = True,
    return_results: bool = True,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    This function loads one or more solution CSVs, summarizes, displays a metrics table, and plots charts.
    """
    
    if outputs is None:
        outputs = {"main": "output.csv", "bonus": "bonus_output.csv"}
    if base_dir is None:
        base_dir = os.getcwd()

    loaded: Dict[str, Dict[str, Any]] = {}
    missing_files = []

    for name, fname in outputs.items():
        path = fname if os.path.isabs(fname) else os.path.join(base_dir, fname)
        if os.path.exists(path):
            sol = _load_solution_csv(path)
            loaded[name] = summarize_solution(sol)
        else:
            missing_files.append(path)

    if not loaded:
        msg = "No output CSVs found."
        if missing_files:
            msg += "\nSearched:\n- " + "\n- ".join(missing_files)
        print(msg)
        return None

    metrics_df = show_metrics_table(loaded)

    if display_table:
        try:
            display(metrics_df)  # noqa: F821 (works in notebooks)
        except NameError:
            print(metrics_df.to_string(index=False))

    for name, res in loaded.items():
        plot_solution_summary(
            res["deliveries"],
            res["deliveries_per_route"],
            title_prefix=name.upper(),
            show=show_plots,
            save_dir=save_plots_dir,
        )

    return loaded if return_results else None
