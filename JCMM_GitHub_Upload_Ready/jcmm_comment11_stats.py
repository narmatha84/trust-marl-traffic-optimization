
import os
import csv
import numpy as np
from scipy.stats import wilcoxon, rankdata

SEEDS = [42, 123, 256, 789, 1024,
         2048, 4096, 8192, 16384, 32768]

SCENARIOS = {
    "Symmetric": (
        "all_results_4mode.npy",
        "mappo_results.npy"
    ),
    "Asymmetric": (
        "asymmetric_results.npy",
        "mappo_results_asymmetric.npy"
    )
}

METRICS = ["Reward", "Wait", "Speed"]


def load_four_modes(filename):
    raw = np.load(filename, allow_pickle=True).item()
    result = {}

    for mode in [1, 2, 3, 4]:
        runs = raw[mode]

        if len(runs) != len(SEEDS):
            raise ValueError(
                f"{filename}: Mode {mode} has "
                f"{len(runs)} runs, expected 10"
            )

        result[mode] = np.array([
            [
                np.mean(np.asarray(run[i], dtype=float))
                for i in range(3)
            ]
            for run in runs
        ])

    return result


def load_mappo(filename):
    raw = np.load(filename, allow_pickle=True)

    if len(raw) != len(SEEDS):
        raise ValueError(
            f"{filename}: expected 10 MAPPO runs"
        )

    return np.array([
        [
            np.mean(np.asarray(run[i], dtype=float))
            for i in range(3)
        ]
        for run in raw
    ])


def compare(a, b):
    # a = proposed Mode 4, b = baseline
    diff = np.asarray(a) - np.asarray(b)

    if np.all(diff == 0):
        return 0.0, 1.0, 0.0, 0.0, 0.0, 0.0

    w, p = wilcoxon(
        diff,
        alternative="two-sided",
        method="auto"
    )

    nonzero = diff[diff != 0]
    ranks = rankdata(np.abs(nonzero))

    positive = np.sum(ranks[nonzero > 0])
    negative = np.sum(ranks[nonzero < 0])
    effect = (positive - negative) / (positive + negative)

    rng = np.random.default_rng(42)
    indices = rng.integers(
        0, len(diff), size=(10000, len(diff))
    )
    bootstrap_means = diff[indices].mean(axis=1)
    low, high = np.percentile(
        bootstrap_means, [2.5, 97.5]
    )

    return (
        float(w), float(p),
        float(effect), float(np.mean(diff)),
        float(low), float(high)
    )


def main():
    rows = []

    for scenario, (modes_file, mappo_file) in SCENARIOS.items():

        print("\n" + "=" * 65)
        print(f"{scenario.upper()} TRAFFIC")
        print("=" * 65)

        if not os.path.isfile(modes_file):
            print(f"MISSING: {modes_file}")
            continue

        modes = load_four_modes(modes_file)

        if os.path.isfile(mappo_file):
            modes["MAPPO"] = load_mappo(mappo_file)
        else:
            print(f"MISSING: {mappo_file}")

        for baseline in [1, 2, 3, "MAPPO"]:
            if baseline not in modes:
                continue

            # M1 and MAPPO use different simulation
            # durations from the original Mode 4.
            comparable = baseline in [2, 3]

            print(f"\nM4 vs {baseline}")
            print(
                "Matched simulation duration: "
                + ("YES" if comparable else "NO")
            )

            for i, metric in enumerate(METRICS):
                w, p, effect, delta, low, high = compare(
                    modes[4][:, i],
                    modes[baseline][:, i]
                )

                print(
                    f"  {metric:6s} | "
                    f"W={w:.2f} | p={p:.5f} | "
                    f"delta={delta:+.4f} | "
                    f"effect={effect:+.3f} | "
                    f"95% CI [{low:+.4f}, {high:+.4f}]"
                )

                rows.append({
                    "scenario": scenario,
                    "comparison": f"M4 vs {baseline}",
                    "metric": metric,
                    "matched_duration": comparable,
                    "W": w,
                    "p_value_unadjusted": p,
                    "mean_difference": delta,
                    "rank_biserial_effect": effect,
                    "ci_lower": low,
                    "ci_upper": high
                })

    if rows:
        with open(
            "jcmm_comment11_statistics.csv",
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(rows)

        print("\nSaved: jcmm_comment11_statistics.csv")
        print("P-values are unadjusted for multiple testing.")
        print("Unmatched-duration results are descriptive only.")


if __name__ == "__main__":
    main()
