#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 14:33:52 2026

@author: anahitamolavi
"""

# Standard library
import sys
import logging
from typing import Any, Dict, Tuple, List

# Third-party libraries
import pandas as pd

# Optimization/modeling
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

# Modules
from modules import utilities

utilities.configure_logging(logging.INFO)

# -------------------- Model Build -------------------- #

def build_mip_model() -> pyo.AbstractModel:
    """
    Abstract MIP model:
      - assign deliveries to drivers
      - sequence deliveries per driver (y variables)
      - schedule pickup times s[i,j]
      - define route end times t[j]
      - compute delivery durations u[i]
      - enforce average duration <= L
      - objective: minimize total route time sum_j (t[j] - T0)
    """
    with utilities.log_step("Build Pyomo AbstractModel"):
        M = pyo.AbstractModel()

        # ---- Sets ----
        M.deliveries = pyo.Set()    # i
        M.drivers = pyo.Set()       # j

        # ---- Parameters ----
        M.created = pyo.Param(M.deliveries, within=pyo.NonNegativeReals)  # c_i (sec)
        M.ready   = pyo.Param(M.deliveries, within=pyo.NonNegativeReals)  # r_i (sec)

        M.tau_intra    = pyo.Param(M.deliveries, within=pyo.NonNegativeReals)  # pickup->dropoff (sec)
        M.tau_deadhead = pyo.Param(M.deliveries, M.deliveries,
                                   within=pyo.NonNegativeReals, default=0.0)  # dropoff i -> pickup k (sec)

        M.T0   = pyo.Param(within=pyo.NonNegativeReals)  # global start (sec)
        M.L    = pyo.Param(within=pyo.NonNegativeReals)  # avg duration target (sec)
        M.BigM = pyo.Param(within=pyo.NonNegativeReals)  # big-M (sec)

        # ---- Decision Variables ----
        M.x  = pyo.Var(M.deliveries, M.drivers, within=pyo.Binary)  # assignment
        M.y  = pyo.Var(M.deliveries, M.deliveries, M.drivers, within=pyo.Binary)  # sequence i->k on j
        M.y0 = pyo.Var(M.deliveries, M.drivers, within=pyo.Binary)  # first delivery on j

        M.s = pyo.Var(M.deliveries, M.drivers, within=pyo.NonNegativeReals)  # pickup time
        M.t = pyo.Var(M.drivers, within=pyo.NonNegativeReals)               # route end time

        M.u = pyo.Var(M.deliveries, within=pyo.NonNegativeReals)            # delivery duration

        # -------------------- Constraints -------------------- #

        # (C1) each delivery assigned once
        def assign_once_rule(M, i):
            return sum(M.x[i, j] for j in M.drivers) == 1
        M.AssignOnce = pyo.Constraint(M.deliveries, rule=assign_once_rule)

        # (C2) pickup cannot be before ready time (if assigned)
        def ready_rule(M, i, j):
            return M.s[i, j] >= M.ready[i] - M.BigM * (1 - M.x[i, j])
        M.ReadyConstraint = pyo.Constraint(M.deliveries, M.drivers, rule=ready_rule)

        # (C3) pickup cannot be before global T0 (if assigned)
        def start_rule(M, i, j):
            return M.s[i, j] >= M.T0 - M.BigM * (1 - M.x[i, j])
        M.StartConstraint = pyo.Constraint(M.deliveries, M.drivers, rule=start_rule)

        # (C4) predecessor: if assigned then exactly one predecessor (either start or another delivery)
        def predecessor_rule(M, i, j):
            return M.y0[i, j] + sum(M.y[k, i, j] for k in M.deliveries if k != i) == M.x[i, j]
        M.Predecessor = pyo.Constraint(M.deliveries, M.drivers, rule=predecessor_rule)

        # (C5) successor: if assigned then at most one successor (open route allowed)
        def successor_rule(M, i, j):
            return sum(M.y[i, k, j] for k in M.deliveries if k != i) <= M.x[i, j]
        M.Successor = pyo.Constraint(M.deliveries, M.drivers, rule=successor_rule)

        # (C6) at most one first delivery per driver
        def one_first_rule(M, j):
            return sum(M.y0[i, j] for i in M.deliveries) <= 1
        M.OneFirst = pyo.Constraint(M.drivers, rule=one_first_rule)

        # (C7) no self loops
        def no_self_loop_rule(M, i, j):
            return M.y[i, i, j] == 0
        M.NoSelfLoop = pyo.Constraint(M.deliveries, M.drivers, rule=no_self_loop_rule)

        # (C8) time propagation: if i precedes k on j then s[k,j] >= s[i,j] + tau_intra[i] + tau_deadhead[i,k]
        def time_prop_rule(M, i, k, j):
            if i == k:
                return pyo.Constraint.Skip
            return M.s[k, j] >= (
                M.s[i, j] + M.tau_intra[i] + M.tau_deadhead[i, k] - M.BigM * (1 - M.y[i, k, j])
            )
        M.TimePropagation = pyo.Constraint(M.deliveries, M.deliveries, M.drivers, rule=time_prop_rule)

        # (C9) route end time t[j] >= dropoff time of any assigned delivery
        def route_end_rule(M, i, j):
            return M.t[j] >= M.s[i, j] + M.tau_intra[i] - M.BigM * (1 - M.x[i, j])
        M.RouteEnd = pyo.Constraint(M.deliveries, M.drivers, rule=route_end_rule)

        # (C10) delivery duration u[i] >= (dropoff - created) for whichever j serves i
        def duration_rule(M, i, j):
            return M.u[i] >= (M.s[i, j] + M.tau_intra[i] - M.created[i]) - M.BigM * (1 - M.x[i, j])
        M.Duration = pyo.Constraint(M.deliveries, M.drivers, rule=duration_rule)

        # (C11) average duration <= L
        def avg_duration_rule(M):
            n = len(list(M.deliveries))
            return sum(M.u[i] for i in M.deliveries) <= M.L * n
        M.AvgDuration = pyo.Constraint(rule=avg_duration_rule)

        # -------------------- Objective -------------------- #
        def objective_rule(M):
            return sum(M.t[j] - M.T0 for j in M.drivers)
        M.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        return M


# -------------------- Data + Solve -------------------- #

def build_instance_data(
    deliveries_df: pd.DataFrame,
    drivers: List[Any],
    T0_utc: str,
    L_minutes: float = 45.0,
    speed_mps: float = 4.5,
    big_m_seconds: float = 24 * 3600,
    log_every_n: int = 50000,
) -> Dict:
    """
    Builds data dict for AbstractModel.create_instance(data)
    """
    with utilities.log_step("Data prep: copy + validate columns"):
        df = deliveries_df.copy()
        required_cols = [
            "Delivery ID", "Created at", "Food ready time",
            "Pickup lat", "Pickup long", "Dropoff lat", "Dropoff long"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        logging.info("Input deliveries: rows=%d", len(df))
        sys.stdout.flush()

    with utilities.log_step("Data prep: IDs + drivers"):
        deliveries_list = df["Delivery ID"].astype(str).tolist()
        drivers_list = [str(x) for x in drivers]
        logging.info("Num deliveries=%d | Num drivers=%d", len(deliveries_list), len(drivers_list))
        sys.stdout.flush()

    with utilities.log_step("Data prep: timestamps -> epoch seconds"):
        created_s = utilities.to_epoch_s(df["Created at"])
        ready_s   = utilities.to_epoch_s(df["Food ready time"])
        T0_s = int(pd.to_datetime(T0_utc, utc=True).timestamp())
        L_s  = float(L_minutes) * 60.0
        logging.info("T0_utc=%s -> T0_s=%d | L=%.1f min", T0_utc, T0_s, L_minutes)
        sys.stdout.flush()

    with utilities.log_step("Data prep: compute tau_intra (pickup->dropoff)"):
        tau_intra: Dict[str, float] = {}
        for idx, row in df.iterrows():
            did = str(row["Delivery ID"])
            dist_m = utilities.haversine(row["Pickup lat"], row["Pickup long"], row["Dropoff lat"], row["Dropoff long"])
            tau_intra[did] = dist_m / speed_mps

        logging.info("tau_intra computed for %d deliveries", len(tau_intra))
        sys.stdout.flush()

    with utilities.log_step("Data prep: cache pickup/dropoff coords"):
        coords: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for _, row in df.iterrows():
            did = str(row["Delivery ID"])
            coords[did] = {
                "p": (float(row["Pickup lat"]), float(row["Pickup long"])),
                "d": (float(row["Dropoff lat"]), float(row["Dropoff long"])),
            }

    with utilities.log_step("Data prep: compute tau_deadhead (dropoff i -> pickup k)"):
        tau_deadhead: Dict[Tuple[str, str], float] = {}
        n = len(deliveries_list)
        total_pairs = n * (n - 1)
        processed = 0

        for i in deliveries_list:
            dlat, dlon = coords[i]["d"]
            for k in deliveries_list:
                if i == k:
                    continue
                plat, plon = coords[k]["p"]
                dist_m = utilities.haversine(dlat, dlon, plat, plon)
                tau_deadhead[(i, k)] = dist_m / speed_mps

                processed += 1
                if log_every_n and processed % log_every_n == 0:
                    logging.info("deadhead pairs computed %d/%d ...", processed, total_pairs)
                    sys.stdout.flush()

        logging.info("tau_deadhead computed: %d pairs (expected %d)", len(tau_deadhead), total_pairs)
        sys.stdout.flush()

    with utilities.log_step("Data prep: build Pyomo data dict"):
        created_dict = dict(zip(deliveries_list, created_s.astype(float).tolist()))
        ready_dict   = dict(zip(deliveries_list, ready_s.astype(float).tolist()))

        data = {
            None: {
                "deliveries": {None: deliveries_list},
                "drivers":    {None: drivers_list},

                "created": created_dict,
                "ready":   ready_dict,

                "tau_intra": tau_intra,
                "tau_deadhead": tau_deadhead,

                "T0":   {None: float(T0_s)},
                "L":    {None: float(L_s)},
                "BigM": {None: float(big_m_seconds)},
            }
        }
        logging.info("Pyomo data dict ready.")
        sys.stdout.flush()
        return data


def summarize_results(results: Any, instance: Any) -> Tuple[pd.DataFrame, Dict[str, float]]:
    with utilities.log_step("Summarize results"):
        logging.info("Solver status=%s | termination=%s",
                     results.solver.status, results.solver.termination_condition)
        sys.stdout.flush()

        ok = (
            results.solver.status == SolverStatus.ok
            and results.solver.termination_condition == TerminationCondition.optimal
        )
        if not ok:
            logging.warning("Non-optimal solve; returning empty outputs.")
            sys.stdout.flush()
            return pd.DataFrame(), {}

        deliveries = list(instance.deliveries)
        drivers = list(instance.drivers)

        rows = []
        for i in deliveries:
            j_star = None
            for j in drivers:
                if pyo.value(instance.x[i, j]) > 0.5:
                    j_star = j
                    break

            s = pyo.value(instance.s[i, j_star]) if j_star is not None else None
            drop = s + float(pyo.value(instance.tau_intra[i])) if s is not None else None
            dur = float(pyo.value(instance.u[i]))

            rows.append({
                "DeliveryID": i,
                "driver": j_star,
                "PickupTime_s": s,
                "DropoffTime_s": drop,
                "Duration_s": dur
            })

        out = pd.DataFrame(rows)

        avg_duration = float(out["Duration_s"].mean()) if len(out) else float("nan")
        total_route_time = sum(float(pyo.value(instance.t[j] - instance.T0)) for j in drivers)
        delivered = len(deliveries)
        deliveries_per_hour_proxy = (delivered * 3600.0 / total_route_time) if total_route_time > 0 else 0.0

        metrics = {
            "avg_duration_s": avg_duration,
            "avg_duration_min": avg_duration / 60.0 if avg_duration == avg_duration else float("nan"),
            "total_route_time_s": float(total_route_time),
            "deliveries_per_hour_proxy": float(deliveries_per_hour_proxy),
            "objective_value": float(pyo.value(instance.obj)),
        }

        logging.info(
            "Metrics: avg_duration=%.2f min | total_route_time=%.1f s | deliv/hr(proxy)=%.3f | obj=%.3f",
            metrics["avg_duration_min"],
            metrics["total_route_time_s"],
            metrics["deliveries_per_hour_proxy"],
            metrics["objective_value"],
        )
        sys.stdout.flush()

        return out, metrics


def run_mip(
    deliveries_df: pd.DataFrame,
    drivers: List[Any],
    T0_utc: str,
    L_minutes: float = 45.0,
    solver_name: str = "glpk",
    tee: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Main entrypoint: builds model, creates instance, solves, summarizes.
    """
    with utilities.log_step("TOTAL RUN: DoorDash MIP"):
        with utilities.log_step("Step: Build abstract model"):
            model = build_mip_model()

        with utilities.log_step("Step: Build data dict"):
            data = build_instance_data(
                deliveries_df=deliveries_df,
                drivers=drivers,
                T0_utc=T0_utc,
                L_minutes=L_minutes,
            )

        with utilities.log_step("Step: Create instance"):
            instance = model.create_instance(data)

        with utilities.log_step(f"Step: Solve (solver={solver_name})"):
            solver = SolverFactory(solver_name)
            results = solver.solve(instance, tee=tee)

        with utilities.log_step("Step: Summarize"):
            out, metrics = summarize_results(results, instance)

        return out, metrics