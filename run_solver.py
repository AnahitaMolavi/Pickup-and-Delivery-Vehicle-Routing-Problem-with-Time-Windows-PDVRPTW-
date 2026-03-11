#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production-style client entrypoint.

Goal:
- Provide a simple CLI wrapper around the existing notebook logic

Usage (from anywhere):
    python run_solver.py --mode greedy

Optional:
    python run_solver.py --config /abs/path/to/client_config.py --mode greedy
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import pandas as pd


REQUIRED_ASSIGNMENT_COLS = {"driver", "PickupTime_s", "DropoffTime_s", "Duration_s"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _add_repo_to_syspath(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _load_config_module(config_path: Path):
    # NOTE: We must register the module in sys.modules BEFORE exec_module.
    # Python 3.13 dataclasses may access sys.modules[cls.__module__] during decoration.
    module_name = f"client_config_{abs(hash(str(config_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(config_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config module from: {config_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _load_cfg(repo_root: Path, config_path: str | None):
    if config_path is None:
        config_file = repo_root / "client_config.py"
    else:
        config_file = Path(config_path).expanduser().resolve()

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    mod = _load_config_module(config_file)
    if not hasattr(mod, "load_config"):
        raise AttributeError(f"{config_file} must define load_config()")

    cfg = mod.load_config()
    return cfg, config_file


def _read_input_csv(repo_root: Path, input_path: Path) -> pd.DataFrame:
    in_path = (repo_root / input_path).resolve() if not input_path.is_absolute() else input_path
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")
    return pd.read_csv(in_path)


def _ensure_outputs_dir_writable(repo_root: Path, out_path: Path) -> Path:
    p = (repo_root / out_path).resolve() if not out_path.is_absolute() else out_path
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _validate_assignment_df(df: pd.DataFrame, label: str) -> None:
    missing = REQUIRED_ASSIGNMENT_COLS - set(df.columns)
    if missing:
        cols = sorted(df.columns)
        raise RuntimeError(
            f"{label} returned an assignment_df that is missing required columns: {sorted(missing)}\n"
            f"assignment_df shape={df.shape}, columns={cols}\n"
            "This indicates no feasible solution was produced (or the wrong object was returned).\n"
            "The notebook produces an assignment_df with these columns before calling create_outputs()."
        )


def run_greedy(deliveries_df: pd.DataFrame, cfg) -> Tuple[pd.DataFrame, dict]:
    from modules import greedy, utilities

    t0 = time.perf_counter()
    assignment_df, metrics = greedy.solve_with_min_drivers(
        deliveries_df=deliveries_df,
        T0_utc=cfg.T0_utc,
        start_drivers=cfg.greedy_start_drivers,
        max_drivers=cfg.greedy_max_drivers,
        drivers_step=cfg.greedy_drivers_step,
        alpha=cfg.greedy_alpha,
        time_window_s=cfg.greedy_time_window_s,
        use_region=cfg.greedy_use_region,
        top_k=cfg.greedy_top_k,
        verbose=cfg.greedy_verbose,
    )
    t1 = time.perf_counter()
    metrics = dict(metrics or {})
    metrics["runtime_sec"] = t1 - t0

    _validate_assignment_df(assignment_df, "Greedy")

    submission_df, req_metrics = utilities.create_outputs(
        assignment_df=assignment_df,
        output_csv_path=str(cfg.greedy_output_path),
        print_metrics=cfg.print_metrics,
    )
    return submission_df, dict(req_metrics or {})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DoorDash OR take-home client runner (notebook-aligned).")
    parser.add_argument("--mode", choices=["greedy"], default="greedy",
                        help="Which solver(s) to run.")
    parser.add_argument("--config", default=None,
                        help="Path to a client_config.py. Defaults to ./client_config.py next to run_solver.py.")
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    _add_repo_to_syspath(repo_root)

    os.chdir(repo_root)
    cfg, cfg_path = _load_cfg(repo_root, args.config)
    from modules import data_prep

    deliveries_df = _read_input_csv(repo_root, cfg.input_path)
    deliveries_df = data_prep.add_distance_and_time_columns(deliveries_df, speed_m_per_sec=cfg.speed_m_per_sec)

    greedy_out = _ensure_outputs_dir_writable(repo_root, cfg.greedy_output_path)
    object.__setattr__(cfg, "greedy_output_path", Path(greedy_out.name))

    print(f"[INFO] Repo root: {repo_root}")
    print(f"[INFO] Config: {cfg_path}")
    print(f"[INFO] Input: {(repo_root / cfg.input_path).resolve() if not cfg.input_path.is_absolute() else cfg.input_path}")

    if args.mode in ("greedy"):
        print("[INFO] Running GREEDY ...")
        run_greedy(deliveries_df, cfg)
        print(f"[INFO] Wrote: {cfg.greedy_output_path}")


    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
