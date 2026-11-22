"""Matched-duration Mode 1 reference for JCMM manuscript 1084.

Place alongside jcmm_comment15_mappo_corrected.py, environment.py, agent.py,
trust_module.py, config.sumocfg and the associated SUMO XML files.

Mode 1 is the ORIGINAL wait-threshold speed rule. This script matches the
corrected M4/MAPPO frozen-evaluation structure: 300 controller decisions,
2 SUMO seconds per decision, 600 seconds total, same ten seeds and SUMO input.
It records the same per-iteration reward, wait, and speed definitions as the
corrected comparison. It does NOT retrain Mode 1 (a deterministic rule).
"""
import argparse
import csv
from pathlib import Path
import statistics

import numpy as np
import traci

from jcmm_comment15_mappo_corrected import (
    STEPS, SEEDS, SeededEnvironment, seed_all, observe, global_features,
)


def run_mode1(seed, sumo_binary):
    seed_all(seed)
    env = SeededEnvironment(seed, sumo_binary)
    env.reset()
    history = []
    arrivals = departures = 0
    try:
        start_time = float(traci.simulation.getTime())
        for _ in range(STEPS):
            env.step()
            arrivals += len(traci.simulation.getArrivedIDList())
            departures += len(traci.simulation.getDepartedIDList())

            obs = observe(env)
            _, avg_speed, avg_wait = global_features(obs)
            rewards = []
            for vehicle_id, (speed, wait) in obs.items():
                # Identical fixed-rule action logic to final_code.py.
                if wait > 20:
                    commanded_speed = 8
                elif wait > 10:
                    commanded_speed = 6
                elif wait > 5:
                    commanded_speed = 4
                else:
                    commanded_speed = 2
                env.apply_speed(vehicle_id, commanded_speed)
                # Same reported reward algebra as the original fixed-rule mode.
                rewards.append(speed / 8 - wait / 15)

            # Second simulation step: same physical time and action interval
            # as the corrected M4/MAPPO 600-second evaluation.
            env.step()
            arrivals += len(traci.simulation.getArrivedIDList())
            departures += len(traci.simulation.getDepartedIDList())
            history.append((float(np.mean(rewards)) if rewards else 0.0,
                            avg_wait, avg_speed))
        duration = float(traci.simulation.getTime()) - start_time
        if duration != STEPS * 2:
            raise RuntimeError(f"Expected {STEPS * 2}s; received {duration}s")
    finally:
        env.close()

    avg = np.mean(np.asarray(history), axis=0)
    return {
        "seed": seed,
        "algorithm": "Mode1_matched_frozen_reference",
        "reward": float(avg[0]),
        "wait": float(avg[1]),
        "speed": float(avg[2]),
        "throughput": float(arrivals * 3600.0 / duration),
        "arrivals": arrivals,
        "departures": departures,
        "duration": duration,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sumo", default="sumo", help="Path to SUMO executable")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output", default="jcmm_mode1_matched_600s.csv")
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        print(f"Running matched Mode 1, seed={seed} ...", flush=True)
        r = run_mode1(seed, args.sumo)
        rows.append(r)
        print("  reward={reward:.4f}, wait={wait:.4f}s, speed={speed:.4f}m/s, "
              "duration={duration:.0f}s".format(**r), flush=True)

    dest = Path(args.output)
    with dest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {dest.resolve()}")
    for metric in ("reward", "wait", "speed", "throughput"):
        vals = [float(r[metric]) for r in rows]
        print(f"Mode 1 {metric}: {statistics.mean(vals):.4f} ± "
              f"{statistics.stdev(vals):.4f} (n={len(vals)})")

    corrected = Path("jcmm_comment15_results_e6.csv")
    if corrected.exists() and len(rows) == 10:
        try:
            from scipy.stats import wilcoxon
            with corrected.open(encoding="utf-8", newline="") as f:
                m4 = {int(r["seed"]): r for r in csv.DictReader(f)
                      if r["algorithm"] == "M4_frozen"}
            paired = [(r, m4[r["seed"]]) for r in rows if r["seed"] in m4]
            if len(paired) == 10:
                print("\nM4 frozen vs Mode 1 matched 600s: paired two-sided Wilcoxon")
                for metric in ("reward", "wait", "speed"):
                    differences = [float(b[metric]) - float(a[metric]) for a, b in paired]
                    result = wilcoxon(differences)
                    print(f"  {metric}: W={result.statistic:.3f}; "
                          f"raw p={result.pvalue:.8f}; "
                          f"mean difference={statistics.mean(differences):.4f}")
                print("Raw p-values above; correct for multiple comparisons before a final claim.")
        except ImportError:
            print("SciPy not installed: paired significance test skipped.")

if __name__ == "__main__":
    main()

"""Matched 600-second Mode 1 evaluation for JCMM."""

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np
import traci

from jcmm_comment15_mappo_corrected import (
    STEPS,
    SEEDS,
    SeededEnvironment,
    seed_all,
    observe,
    global_features,
)


def run_mode1(seed, sumo_binary):
    # Match the held-out evaluation traffic used for M4 and MAPPO.
    eval_seed = seed + 100000

    seed_all(eval_seed)
    env = SeededEnvironment(eval_seed, sumo_binary)
    env.reset()

    history = []
    arrivals = 0
    departures = 0

    try:
        start_time = float(traci.simulation.getTime())

        for _ in range(STEPS):
            env.step()

            arrivals += len(
                traci.simulation.getArrivedIDList()
            )
            departures += len(
                traci.simulation.getDepartedIDList()
            )

            obs = observe(env)
            _, avg_speed, avg_wait = global_features(obs)

            rewards = []

            for vehicle_id, (speed, wait) in obs.items():

                # Original Mode 1 rule-based speed control.
                if wait > 20:
                    commanded_speed = 8
                elif wait > 10:
                    commanded_speed = 6
                elif wait > 5:
                    commanded_speed = 4
                else:
                    commanded_speed = 2

                env.apply_speed(
                    vehicle_id,
                    commanded_speed
                )

                rewards.append(
                    speed / 8 - wait / 15
                )

            # Match the two SUMO steps per controller decision.
            env.step()

            arrivals += len(
                traci.simulation.getArrivedIDList()
            )
            departures += len(
                traci.simulation.getDepartedIDList()
            )

            history.append((
                float(np.mean(rewards)) if rewards else 0.0,
                avg_wait,
                avg_speed
            ))

        duration = (
            float(traci.simulation.getTime()) - start_time
        )

        if duration != STEPS * 2:
            raise RuntimeError(
                f"Expected {STEPS * 2}s; received {duration}s"
            )

    finally:
        env.close()

    avg = np.mean(np.asarray(history), axis=0)

    return {
        "seed": seed,
        "algorithm": "Mode1_matched_frozen_reference",
        "reward": float(avg[0]),
        "wait": float(avg[1]),
        "speed": float(avg[2]),
        "throughput": float(
            arrivals * 3600.0 / duration
        ),
        "arrivals": arrivals,
        "departures": departures,
        "duration": duration,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sumo",
        default="sumo",
        help="Path to SUMO executable"
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS
    )

    parser.add_argument(
        "--output",
        default="jcmm_mode1_matched_eval_600s.csv"
    )

    args = parser.parse_args()

    rows = []

    for seed in args.seeds:
        print(
            f"Running matched Mode 1, seed={seed} ...",
            flush=True
        )

        result = run_mode1(
            seed,
            args.sumo
        )

        rows.append(result)

        print(
            "  reward={reward:.4f}, "
            "wait={wait:.4f}s, "
            "speed={speed:.4f}m/s, "
            "duration={duration:.0f}s".format(**result),
            flush=True
        )

    # Save results.
    destination = Path(args.output)

    with destination.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {destination.resolve()}")

    # Print mean and standard deviation.
    for metric in (
        "reward",
        "wait",
        "speed",
        "throughput"
    ):

        values = [
            float(row[metric])
            for row in rows
        ]

        print(
            f"Mode 1 {metric}: "
            f"{statistics.mean(values):.4f} ± "
            f"{statistics.stdev(values):.4f} "
            f"(n={len(values)})"
        )

    # Statistical comparison with corrected M4.
    corrected = Path(
        "jcmm_comment15_results_e6.csv"
    )

    if corrected.exists() and len(rows) == 10:

        try:
            from scipy.stats import wilcoxon

            with corrected.open(
                encoding="utf-8",
                newline=""
            ) as file:

                m4 = {
                    int(row["seed"]): row
                    for row in csv.DictReader(file)
                    if row["algorithm"] == "M4_frozen"
                }

            paired = [
                (row, m4[row["seed"]])
                for row in rows
                if row["seed"] in m4
            ]

            if len(paired) == 10:

                print(
                    "\nM4 frozen vs Mode 1 matched 600s:"
                    "\nPaired two-sided Wilcoxon"
                )

                for metric in (
                    "reward",
                    "wait",
                    "speed"
                ):

                    differences = [
                        float(m4_row[metric])
                        - float(mode1_row[metric])
                        for mode1_row, m4_row in paired
                    ]

                    test = wilcoxon(differences)

                    print(
                        f"  {metric}: "
                        f"W={test.statistic:.3f}; "
                        f"raw p={test.pvalue:.8f}; "
                        f"mean difference="
                        f"{statistics.mean(differences):.4f}"
                    )

                print(
                    "\nThese are raw p-values. "
                    "Apply multiple-comparison correction "
                    "before reporting significance."
                )

        except ImportError:
            print(
                "SciPy not installed: "
                "statistical tests skipped."
            )


if __name__ == "__main__":
    main()
