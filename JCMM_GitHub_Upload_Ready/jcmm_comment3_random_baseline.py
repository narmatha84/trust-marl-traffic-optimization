
import os
import csv
import random
import argparse

import numpy as np
import torch

from environment import SumoEnvironment
from agent import ACTIONS


# ============================================================
# REVIEWER COMMENT 3
# Uniform Random Speed Baseline
#
# Original first-revision SUMO environment.
# No DQN, no trust, no training.
# No original project files are modified.
# ============================================================

SEEDS = [
    42, 123, 256, 789, 1024,
    2048, 4096, 8192, 16384, 32768
]

MAX_STEPS = 300
OUTPUT_DIR = "comment3_random_results"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_random_baseline(seed):
    set_seed(seed)

    sumo = SumoEnvironment("config.sumocfg")

    reward_history = []
    wait_history = []
    speed_history = []
    action_counts = {int(a): 0 for a in ACTIONS}

    try:
        sumo.reset()

        for step in range(MAX_STEPS):

            # Original simulation structure:
            # one SUMO step, then observe and apply speeds.
            sumo.step()
            vehicles = sumo.get_vehicles()

            vehicle_states = {}

            for v in vehicles:
                _, speed, wait, _ = sumo.get_state(v)
                vehicle_states[v] = (speed, wait)

            avg_speed = (
                np.mean([s for s, w in vehicle_states.values()])
                if vehicles else 0.0
            )

            avg_wait = (
                np.mean([w for s, w in vehicle_states.values()])
                if vehicles else 0.0
            )

            rewards = []

            for v in vehicles:
                speed, wait = vehicle_states[v]

                # Uniform random selection:
                # Each action has probability 1 / 4.
                action = int(random.choice(ACTIONS))

                action_counts[action] += 1

                # Use original environment's driver behaviour.
                sumo.apply_speed(v, action)

                # Same shaped reward definition as Mode 4.
                individual = speed / 8.0 - wait / 15.0

                global_reward = (
                    avg_speed / 8.0 - avg_wait / 15.0
                )

                reward = (
                    0.7 * individual
                    + 0.3 * global_reward
                )

                rewards.append(reward)

            # Match the two-step loop of original
            # Modes 2, 3 and 4, without performing
            # any DQN updates.
            sumo.step()

            reward_history.append(
                float(np.mean(rewards)) if rewards else 0.0
            )

            wait_history.append(float(avg_wait))
            speed_history.append(float(avg_speed))

            if (step + 1) % 50 == 0:
                print(
                    f"Seed {seed} | "
                    f"Step {step + 1}/{MAX_STEPS} | "
                    f"Reward {reward_history[-1]:.4f} | "
                    f"Wait {avg_wait:.4f} | "
                    f"Speed {avg_speed:.4f}",
                    flush=True
                )

    finally:
        try:
            sumo.close()
        except Exception:
            pass

    total_actions = sum(action_counts.values())

    result = {
        "seed": seed,
        "reward": float(np.mean(reward_history)),
        "wait": float(np.mean(wait_history)),
        "speed": float(np.mean(speed_history)),
        "total_actions": total_actions
    }

    for action, count in action_counts.items():
        result[f"action_{action}_count"] = count
        result[f"action_{action}_pct"] = (
            100.0 * count / total_actions
            if total_actions else 0.0
        )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS
    )

    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []

    for seed in args.seeds:
        print(
            f"\nSEED {seed}: Running uniform random "
            f"speed baseline ...",
            flush=True
        )

        result = run_random_baseline(seed)
        results.append(result)

        print(
            f"   Random: "
            f"reward={result['reward']:.4f} "
            f"wait={result['wait']:.4f} "
            f"speed={result['speed']:.4f}",
            flush=True
        )

        print(
            "   Action distribution: "
            + " | ".join(
                f"{a} m/s: {result[f'action_{a}_pct']:.2f}%"
                for a in ACTIONS
            ),
            flush=True
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        "comment3_uniform_random_results.csv"
    )

    with open(
        output_path, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print(
        "\n=== COMMENT 3 SUMMARY "
        "(original symmetric protocol) ===",
        flush=True
    )

    print(f"Seeds: {len(results)}")

    for metric in ["reward", "wait", "speed"]:
        values = np.array(
            [r[metric] for r in results],
            dtype=float
        )

        print(
            f"{metric}: "
            f"{np.mean(values):.4f} "
            f"+/- {np.std(values, ddof=1) if len(values) > 1 else 0:.4f}",
            flush=True
        )

    print(f"Saved: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
