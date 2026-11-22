
import contextlib
import csv
import os

import numpy as np
from scipy.stats import wilcoxon

from experiment_smoothening import run_sensitivity as run_smoothing
from experiment_sensitivity import run_sensitivity as run_threshold

SEEDS = [
    42, 123, 256, 789, 1024,
    2048, 4096, 8192, 16384, 32768
]

CONFIGS = {
    "alpha_070": ("smoothing", 0.70),
    "default_alpha_090_low_05_high_15": ("smoothing", 0.90),
    "alpha_095": ("smoothing", 0.95),
    "threshold_03_17": ("threshold", (0.3, 1.7)),
    "threshold_07_13": ("threshold", (0.7, 1.3))
}

BASE = "default_alpha_090_low_05_high_15"

PROGRESS = "jcmm_comments13_14_progress.npy"
CSV_FILE = "jcmm_comments13_14_results.csv"

OLD_SOURCES = [
    (
        "smoothing_sensitivity_results.npy",
        {
            "A: alpha=0.70": "alpha_070",
            "B: alpha=0.90": BASE,
            "C: alpha=0.95": "alpha_095"
        }
    ),
    (
        "sensitivity_results.npy",
        {
            "A: low=0.3, high=1.7": "threshold_03_17",
            "B: low=0.5, high=1.5": BASE,
            "C: low=0.7, high=1.3": "threshold_07_13"
        }
    )
]


def row_from_history(hist):
    if len(hist) != 3 or any(len(arr) != 300 for arr in hist):
        raise ValueError(
            "Expected reward, wait, speed histories of length 300"
        )

    return tuple(
        float(np.mean(arr))
        for arr in hist
    )


def load_existing():
    results = {
        name: {}
        for name in CONFIGS
    }

    # Resume a previously interrupted combined experiment.
    if os.path.isfile(PROGRESS):
        prior = np.load(
            PROGRESS, allow_pickle=True
        ).item()

        for name in CONFIGS:
            for seed, values in prior.get(name, {}).items():
                if int(seed) in SEEDS:
                    results[name][int(seed)] = tuple(
                        map(float, values)
                    )

        print("Loaded previous Comment 13/14 progress.")

    # Reuse original sensitivity experiment results.
    for filename, mapping in OLD_SOURCES:

        if not os.path.isfile(filename):
            print(f"No previous cache: {filename}")
            continue

        try:
            old = np.load(
                filename, allow_pickle=True
            ).item()

            count = 0

            for old_name, new_name in mapping.items():
                histories = old.get(old_name, [])

                # Original scripts used the first five seeds.
                if len(histories) not in (5, 10):
                    continue

                for i, hist in enumerate(histories):
                    seed = SEEDS[i]

                    if seed not in results[new_name]:
                        results[new_name][seed] = (
                            row_from_history(hist)
                        )
                        count += 1

            print(
                f"Reused {count} prior runs from {filename}"
            )

        except Exception as exc:
            print(
                f"Could not reuse {filename}: {exc}"
            )

    return results


def save_results(results):
    np.save(PROGRESS, results)

    with open(
        CSV_FILE, "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "config", "seed",
            "reward", "wait_s", "speed_m_s"
        ])

        for name in CONFIGS:
            for seed in SEEDS:

                if seed in results[name]:
                    writer.writerow([
                        name,
                        seed,
                        *results[name][seed]
                    ])


def run_one(kind, param, seed):
    # Reuse the original experiment functions.
    # Suppress their per-step messages, not their calculations.
    try:
        with open(os.devnull, "w") as sink:
            with contextlib.redirect_stdout(sink):

                if kind == "smoothing":
                    histories = run_smoothing(
                        param, seed=seed
                    )
                else:
                    low, high = param
                    histories = run_threshold(
                        low, high, seed=seed
                    )

        return row_from_history(histories)

    except Exception:
        try:
            import traci

            if traci.isLoaded():
                traci.close()

        except Exception:
            pass

        raise


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(
        len(p_values), dtype=float
    )

    running_max = 0.0
    total = len(p_values)

    for index, pos in enumerate(order):
        running_max = max(
            running_max,
            min(1.0, (total - index) * p_values[pos])
        )

        adjusted[pos] = running_max

    return adjusted


def main():
    results = load_existing()
    save_results(results)

    # Complete only the missing runs.
    for name, (kind, param) in CONFIGS.items():

        for seed in SEEDS:

            if seed in results[name]:
                print(
                    f"REUSED {name} | seed {seed}"
                )
                continue

            print(
                f"RUNNING {name} | seed {seed}",
                flush=True
            )

            results[name][seed] = run_one(
                kind, param, seed
            )

            # Save after every run, so progress is retained.
            save_results(results)

            reward, wait, speed = results[name][seed]

            print(
                f"  Reward={reward:.4f} | "
                f"Wait={wait:.4f}s | "
                f"Speed={speed:.4f}m/s",
                flush=True
            )

    print("\n" + "=" * 72)
    print("COMMENTS 13 & 14: TEN-SEED SENSITIVITY RESULTS")
    print("Mean +/- sample standard deviation")
    print("=" * 72)

    for name in CONFIGS:

        arr = np.array([
            results[name][seed]
            for seed in SEEDS
        ], dtype=float)

        print(f"\n{name}")

        for i, metric in enumerate([
            "Reward", "Waiting time", "Speed"
        ]):
            values = arr[:, i]

            print(
                f"  {metric}: "
                f"{values.mean():.4f} +/- "
                f"{values.std(ddof=1):.4f}"
            )

    print("\n" + "=" * 72)
    print("PAIRED WILCOXON TESTS VS ORIGINAL CONFIGURATION")
    print("=" * 72)

    base = np.array([
        results[BASE][seed]
        for seed in SEEDS
    ], dtype=float)

    pvals = []
    labels = []

    for name in CONFIGS:

        if name == BASE:
            continue

        other = np.array([
            results[name][seed]
            for seed in SEEDS
        ], dtype=float)

        for i, metric in enumerate([
            "Reward", "Wait", "Speed"
        ]):

            differences = other[:, i] - base[:, i]

            if np.allclose(differences, 0):
                p = 1.0
            else:
                p = float(
                    wilcoxon(
                        differences,
                        alternative="two-sided"
                    ).pvalue
                )

            pvals.append(p)

            labels.append((
                name,
                metric,
                float(np.mean(differences))
            ))

    adjusted = holm_adjust(
        np.asarray(pvals)
    )

    for (name, metric, difference), p, padj in zip(
        labels, pvals, adjusted
    ):
        print(
            f"{name} | {metric} | "
            f"Difference={difference:+.4f} | "
            f"p={p:.5f} | "
            f"Holm p={padj:.5f}"
        )

    print("\nSaved:")
    print(CSV_FILE)
    print(PROGRESS)


if __name__ == "__main__":
    main()
