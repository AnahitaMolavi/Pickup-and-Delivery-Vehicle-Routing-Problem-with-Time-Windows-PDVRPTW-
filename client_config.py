"""
Client-facing configuration
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClientConfig:
    # ----------- Paths (relative to repo root by default) ---------------- #
    input_path: Path = Path("data/input.csv")
    greedy_output_path: Path = Path("output.csv")

    # ----------- Shared parameters --------------------- #
    speed_m_per_sec: float = 4.5
    T0_utc: str = "2015-02-03 02:00:00"

    # ----------- Greedy parameters -------------- #
    greedy_start_drivers: int = 10
    greedy_max_drivers: int = 60
    greedy_drivers_step: int = 5
    greedy_alpha: float = 2.0
    greedy_time_window_s: int = 3600
    greedy_use_region: bool = True
    greedy_top_k: int = 50
    greedy_verbose: bool = False

    # ----------- Output behavior ------------- #
    print_metrics: bool = True


def load_config() -> ClientConfig:
    """Convenience loader so `run_solver.py` can import a single function."""
    return ClientConfig()
