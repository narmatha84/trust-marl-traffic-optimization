"""
Generate Per-Seed Output Tables for Journal Supplementary Files
===============================================================

Loads saved .npy result files and generates clean per-seed CSV tables
for submission as supplementary material to the journal.

Output files:
    table2_symmetric_per_seed.csv
    table3_asymmetric_per_seed.csv
    table4_sensitivity_per_seed.csv

Usage:
    python generate_per_seed_tables.py
"""

import numpy as np
import csv
import os


# ============================================================
# SEEDS
# ============================================================

SYMMETRIC_SEEDS  = [42, 123, 256, 789, 1024,
                    2048, 4096, 8192, 16384, 32768]

ASYMMETRIC_SEEDS = [42, 123, 256, 789, 1024,
                    2048, 4096, 8192, 16384, 32768]

SENSITIVITY_SEEDS = [42, 123, 256, 789, 1024]


# ============================================================
# HELPER: Print and write CSV
# ============================================================

def write_csv(filename, headers, rows):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Saved: {filename}")


# ============================================================
# TABLE 2 — SYMMETRIC TRAFFIC (all_results_4mode.npy)
# ============================================================

def generate_table2():

    file = "all_results_4mode.npy"

    if not os.path.exists(file):
        print(f"[SKIP] {file} not found — skipping Table 2")
        return

    data = np.load(file, allow_pickle=True).item()

    mode_labels = {
        1: "Mode 1 — Fixed Rule",
        2: "Mode 2 — Pure DQN",
        3: "Mode 3 — Reward Shaping",
        4: "Mode 4 — DQN + Trust (Proposed)"
    }

    print("\n" + "=" * 70)
    print("TABLE 2 — SYMMETRIC TRAFFIC — PER-SEED RESULTS")
    print("=" * 70)

    headers = ["Method", "Seed", "Avg Reward", "Avg Wait (s)", "Avg Speed (m/s)"]
    rows = []

    for mode in [1, 2, 3, 4]:
        runs = data[mode]
        label = mode_labels[mode]

        for i, (r_hist, w_hist, s_hist) in enumerate(runs):

            seed = SYMMETRIC_SEEDS[i] if i < len(SYMMETRIC_SEEDS) else f"Run {i+1}"

            avg_r = np.mean(r_hist)
            avg_w = np.mean(w_hist)
            avg_s = np.mean(s_hist)

            row = [label, seed, f"{avg_r:.4f}", f"{avg_w:.4f}", f"{avg_s:.4f}"]
            rows.append(row)

            print(
                f"{label:40s} | Seed {str(seed):6s} | "
                f"Reward {avg_r:.4f} | "
                f"Wait {avg_w:.4f} s | "
                f"Speed {avg_s:.4f} m/s"
            )

        # Summary row
        all_r = [np.mean(r) for r, w, s in runs]
        all_w = [np.mean(w) for r, w, s in runs]
        all_s = [np.mean(s) for r, w, s in runs]

        summary = [
            label,
            "MEAN ± STD",
            f"{np.mean(all_r):.4f} ± {np.std(all_r):.4f}",
            f"{np.mean(all_w):.4f} ± {np.std(all_w):.4f}",
            f"{np.mean(all_s):.4f} ± {np.std(all_s):.4f}"
        ]
        rows.append(summary)
        rows.append(["", "", "", "", ""])  # blank separator

        print(
            f"{label:40s} | MEAN ± STD | "
            f"Reward {np.mean(all_r):.4f} ± {np.std(all_r):.4f} | "
            f"Wait {np.mean(all_w):.4f} ± {np.std(all_w):.4f} | "
            f"Speed {np.mean(all_s):.4f} ± {np.std(all_s):.4f}"
        )
        print()

    write_csv("table2_symmetric_per_seed.csv", headers, rows)


# ============================================================
# TABLE 3 — ASYMMETRIC TRAFFIC (asymmetric_results.npy)
# ============================================================

def generate_table3():

    file = "asymmetric_results.npy"

    if not os.path.exists(file):
        print(f"[SKIP] {file} not found — skipping Table 3")
        return

    data = np.load(file, allow_pickle=True).item()

    mode_labels = {
        1: "Mode 1 — Fixed Rule",
        2: "Mode 2 — Pure DQN",
        3: "Mode 3 — Reward Shaping",
        4: "Mode 4 — DQN + Trust (Proposed)"
    }

    # Check if MAPPO asymmetric results exist separately
    mappo_file = "mappo_asymmetric_results.npy"
    mappo_data = None
    if os.path.exists(mappo_file):
        mappo_data = np.load(mappo_file, allow_pickle=True)
        print(f"Found {mappo_file} — including MAPPO asymmetric results")

    print("\n" + "=" * 70)
    print("TABLE 3 — ASYMMETRIC TRAFFIC — PER-SEED RESULTS")
    print("=" * 70)

    headers = ["Method", "Seed", "Avg Reward", "Avg Wait (s)", "Avg Speed (m/s)"]
    rows = []

    for mode in [1, 2, 3, 4]:

        if mode not in data:
            print(f"[SKIP] Mode {mode} not in asymmetric results")
            continue

        runs = data[mode]
        label = mode_labels[mode]

        for i, run in enumerate(runs):

            if isinstance(run, tuple):
                r_hist, w_hist, s_hist = run
            else:
                r_hist, w_hist, s_hist = run[0], run[1], run[2]

            seed = ASYMMETRIC_SEEDS[i] if i < len(ASYMMETRIC_SEEDS) else f"Run {i+1}"

            avg_r = np.mean(r_hist)
            avg_w = np.mean(w_hist)
            avg_s = np.mean(s_hist)

            row = [label, seed, f"{avg_r:.4f}", f"{avg_w:.4f}", f"{avg_s:.4f}"]
            rows.append(row)

            print(
                f"{label:40s} | Seed {str(seed):6s} | "
                f"Reward {avg_r:.4f} | "
                f"Wait {avg_w:.4f} s | "
                f"Speed {avg_s:.4f} m/s"
            )

        all_r = [np.mean(r if isinstance(run, tuple) else run[0])
                 for run in runs
                 for r in [run[0] if isinstance(run, tuple) else run[0]]]
        all_r = [np.mean(run[0]) for run in runs]
        all_w = [np.mean(run[1]) for run in runs]
        all_s = [np.mean(run[2]) for run in runs]

        summary = [
            label,
            "MEAN ± STD",
            f"{np.mean(all_r):.4f} ± {np.std(all_r):.4f}",
            f"{np.mean(all_w):.4f} ± {np.std(all_w):.4f}",
            f"{np.mean(all_s):.4f} ± {np.std(all_s):.4f}"
        ]
        rows.append(summary)
        rows.append(["", "", "", "", ""])
        print()

    # MAPPO asymmetric if available
    if mappo_data is not None:
        label = "MAPPO"
        for i, run in enumerate(mappo_data):
            r_hist, w_hist, s_hist = run
            seed = ASYMMETRIC_SEEDS[i] if i < len(ASYMMETRIC_SEEDS) else f"Run {i+1}"
            avg_r = np.mean(r_hist)
            avg_w = np.mean(w_hist)
            avg_s = np.mean(s_hist)
            rows.append([label, seed, f"{avg_r:.4f}", f"{avg_w:.4f}", f"{avg_s:.4f}"])
            print(
                f"{label:40s} | Seed {str(seed):6s} | "
                f"Reward {avg_r:.4f} | Wait {avg_w:.4f} | Speed {avg_s:.4f}"
            )

        all_r = [np.mean(r) for r, w, s in mappo_data]
        all_w = [np.mean(w) for r, w, s in mappo_data]
        all_s = [np.mean(s) for r, w, s in mappo_data]
        rows.append([
            label, "MEAN ± STD",
            f"{np.mean(all_r):.4f} ± {np.std(all_r):.4f}",
            f"{np.mean(all_w):.4f} ± {np.std(all_w):.4f}",
            f"{np.mean(all_s):.4f} ± {np.std(all_s):.4f}"
        ])

    write_csv("table3_asymmetric_per_seed.csv", headers, rows)


# ============================================================
# TABLE 4 — SENSITIVITY ANALYSIS (all three sensitivity files)
# ============================================================

def generate_table4():

    print("\n" + "=" * 70)
    print("TABLE 4 — SENSITIVITY ANALYSIS — PER-SEED RESULTS")
    print("=" * 70)

    headers = [
        "Parameter", "Configuration", "Setting",
        "Seed", "Avg Reward", "Avg Wait (s)", "Avg Speed (m/s)"
    ]
    rows = []

    # ── Threshold sensitivity ─────────────────────────────────────────────
    thresh_file = "sensitivity_results.npy"

    if os.path.exists(thresh_file):

        thresh_data = np.load(thresh_file, allow_pickle=True).item()

        config_labels = {
            "A: low=0.3, high=1.7": "0.3 / 1.7",
            "B: low=0.5, high=1.5": "0.5 / 1.5 (Default)",
            "C: low=0.7, high=1.3": "0.7 / 1.3"
        }

        print("\nThreshold Sensitivity:")

        for config_name, runs in thresh_data.items():
            setting = config_labels.get(config_name, config_name)

            for i, (r_hist, w_hist, s_hist) in enumerate(runs):
                seed = SENSITIVITY_SEEDS[i] if i < len(SENSITIVITY_SEEDS) else f"Run {i+1}"
                avg_r = np.mean(r_hist)
                avg_w = np.mean(w_hist)
                avg_s = np.mean(s_hist)

                rows.append([
                    "Threshold", config_name[0],
                    setting, seed,
                    f"{avg_r:.4f}", f"{avg_w:.4f}", f"{avg_s:.4f}"
                ])

                print(
                    f"Threshold | {config_name[0]} | {setting:20s} | "
                    f"Seed {str(seed):6s} | "
                    f"Reward {avg_r:.4f} | Wait {avg_w:.4f}"
                )

            all_r = [np.mean(r) for r, w, s in runs]
            all_w = [np.mean(w) for r, w, s in runs]
            rows.append([
                "Threshold", config_name[0], setting, "MEAN ± STD",
                f"{np.mean(all_r):.4f} ± {np.std(all_r):.4f}",
                f"{np.mean(all_w):.4f} ± {np.std(all_w):.4f}", ""
            ])
            rows.append(["", "", "", "", "", "", ""])

    else:
        print(f"[SKIP] {thresh_file} not found")

    # ── Smoothing sensitivity ─────────────────────────────────────────────
    smooth_file = "smoothing_sensitivity_results.npy"

    if os.path.exists(smooth_file):

        smooth_data = np.load(smooth_file, allow_pickle=True).item()

        print("\nSmoothing Factor Sensitivity:")

        for config_name, runs in smooth_data.items():

            for i, (r_hist, w_hist, s_hist) in enumerate(runs):
                seed = SENSITIVITY_SEEDS[i] if i < len(SENSITIVITY_SEEDS) else f"Run {i+1}"
                avg_r = np.mean(r_hist)
                avg_w = np.mean(w_hist)
                avg_s = np.mean(s_hist)

                rows.append([
                    "Smoothing", config_name[0],
                    config_name, seed,
                    f"{avg_r:.4f}", f"{avg_w:.4f}", f"{avg_s:.4f}"
                ])

                print(
                    f"Smoothing | {config_name[0]} | {config_name:20s} | "
                    f"Seed {str(seed):6s} | "
                    f"Reward {avg_r:.4f} | Wait {avg_w:.4f}"
                )

            all_r = [np.mean(r) for r, w, s in runs]
            all_w = [np.mean(w) for r, w, s in runs]
            rows.append([
                "Smoothing", config_name[0], config_name, "MEAN ± STD",
                f"{np.mean(all_r):.4f} ± {np.std(all_r):.4f}",
                f"{np.mean(all_w):.4f} ± {np.std(all_w):.4f}", ""
            ])
            rows.append(["", "", "", "", "", "", ""])

    else:
        print(f"[SKIP] {smooth_file} not found")

    # ── Clipping sensitivity ──────────────────────────────────────────────
    clip_file = "clipping_sensitivity_results.npy"

    if os.path.exists(clip_file):

        clip_data = np.load(clip_file, allow_pickle=True).item()

        print("\nClipping Range Sensitivity:")

        for config_name, runs in clip_data.items():

            for i, (r_hist, w_hist, s_hist) in enumerate(runs):
                seed = SENSITIVITY_SEEDS[i] if i < len(SENSITIVITY_SEEDS) else f"Run {i+1}"
                avg_r = np.mean(r_hist)
                avg_w = np.mean(w_hist)
                avg_s = np.mean(s_hist)

                rows.append([
                    "Clipping", config_name[0],
                    config_name, seed,
                    f"{avg_r:.4f}", f"{avg_w:.4f}", f"{avg_s:.4f}"
                ])

                print(
                    f"Clipping  | {config_name[0]} | {config_name:20s} | "
                    f"Seed {str(seed):6s} | "
                    f"Reward {avg_r:.4f} | Wait {avg_w:.4f}"
                )

            all_r = [np.mean(r) for r, w, s in runs]
            all_w = [np.mean(w) for r, w, s in runs]
            rows.append([
                "Clipping", config_name[0], config_name, "MEAN ± STD",
                f"{np.mean(all_r):.4f} ± {np.std(all_r):.4f}",
                f"{np.mean(all_w):.4f} ± {np.std(all_w):.4f}", ""
            ])
            rows.append(["", "", "", "", "", "", ""])

    else:
        print(f"[SKIP] {clip_file} not found")

    write_csv("table4_sensitivity_per_seed.csv", headers, rows)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("  GENERATING PER-SEED SUPPLEMENTARY TABLES")
    print("=" * 70)

    generate_table2()
    generate_table3()
    generate_table4()

    print("\n" + "=" * 70)
    print("COMPLETE — Files generated:")
    print("  1. table2_symmetric_per_seed.csv")
    print("  2. table3_asymmetric_per_seed.csv")
    print("  3. table4_sensitivity_per_seed.csv")
    print("\nSubmit these three CSV files as supplementary material.")
    print("=" * 70)