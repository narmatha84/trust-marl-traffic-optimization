
import random
import csv
from collections import defaultdict

import numpy as np
import torch

from environment import SumoEnvironment
from agent import Agent, ACTIONS
from trust_module import TrustModule

SEEDS = [42, 123, 256, 789, 1024,
         2048, 4096, 8192, 16384, 32768]

PROFILES = ["normal", "aggressive",
            "conservative", "non_compliant"]


def discretize(speed, wait):
    speed_bin = min(int(speed // 2), 4)
    if wait < 5:
        wait_bin = 0
    elif wait < 10:
        wait_bin = 1
    elif wait < 20:
        wait_bin = 2
    else:
        wait_bin = 3
    return speed_bin * 4 + wait_bin


def run(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sumo = SumoEnvironment("config.sumocfg")
    agent = Agent()
    trust = TrustModule()

    epsilon = 0.1

    data = {
        p: {
            "trust": [],
            "speed": [],
            "wait": [],
            "forced": 0,
            "normal_gate": 0,
            "high_gate": 0,
            "vehicles": set()
        }
        for p in PROFILES
    }

    sumo.reset()

    try:
        for step in range(300):
            sumo.step()
            vehicles = sumo.get_vehicles()

            states = {}
            for v in vehicles:
                _, speed, wait, _ = sumo.get_state(v)
                states[v] = (speed, wait)

            avg_speed = (
                np.mean([x[0] for x in states.values()])
                if states else 0
            )
            avg_wait = (
                np.mean([x[1] for x in states.values()])
                if states else 0
            )

            memory = []

            for v in vehicles:
                speed, wait = states[v]
                s = discretize(speed, wait)

                trust.update(v, speed, wait)
                t = float(trust.get(v))

                agent.init(v)
                action_idx = agent.choose(v, s, epsilon, t)
                sumo.apply_speed(v, ACTIONS[action_idx])

                # The original environment assigns the
                # profile during apply_speed().
                profile = sumo.driver_profiles[v]
                d = data[profile]

                d["vehicles"].add(v)
                d["trust"].append(t)
                d["speed"].append(speed)
                d["wait"].append(wait)

                if t < 0.5:
                    d["forced"] += 1
                elif t > 1.5:
                    d["high_gate"] += 1
                else:
                    d["normal_gate"] += 1

                individual = speed / 8 - wait / 15
                global_reward = (
                    avg_speed / 8 - avg_wait / 15
                )
                reward = (
                    0.7 * individual
                    + 0.3 * global_reward
                )

                memory.append((v, s, action_idx, reward))

            sumo.step()
            next_vehicles = set(sumo.get_vehicles())

            for v, s, a, r in memory:
                if v in next_vehicles:
                    _, speed, wait, _ = sumo.get_state(v)
                    ns = discretize(speed, wait)
                    agent.update(
                        v, s, a, r, ns, 0.05, 0.9
                    )

            epsilon = max(0.05, epsilon * 0.995)

    finally:
        sumo.close()

    rows = []

    for profile, d in data.items():
        n = len(d["trust"])

        if n == 0:
            continue

        rows.append({
            "seed": seed,
            "profile": profile,
            "unique_vehicles": len(d["vehicles"]),
            "decisions": n,
            "mean_trust": np.mean(d["trust"]),
            "mean_speed": np.mean(d["speed"]),
            "mean_wait": np.mean(d["wait"]),
            "forced_random_pct": 100 * d["forced"] / n,
            "epsilon_greedy_regime_pct":
                100 * d["normal_gate"] / n,
            "high_trust_regime_pct":
                100 * d["high_gate"] / n
        })

    return rows


def main():
    all_rows = []

    for seed in SEEDS:
        print(f"Running Mode 4 | Seed {seed}", flush=True)
        all_rows.extend(run(seed))

    filename = "jcmm_comment9_driver_profiles.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=all_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n" + "=" * 65)
    print("COMMENT 9: DRIVER PROFILE RESULTS")
    print("=" * 65)

    metrics = [
        "unique_vehicles",
        "decisions",
        "mean_trust",
        "forced_random_pct",
        "epsilon_greedy_regime_pct",
        "high_trust_regime_pct",
        "mean_speed",
        "mean_wait"
    ]

    for profile in PROFILES:
        rows = [
            r for r in all_rows
            if r["profile"] == profile
        ]

        print(f"\nPROFILE: {profile.upper()}")

        for metric in metrics:
            values = np.array(
                [r[metric] for r in rows],
                dtype=float
            )

            print(
                f"  {metric}: "
                f"{np.mean(values):.4f} "
                f"+/- {np.std(values):.4f}"
            )

    print(f"\nSaved: {filename}")


if __name__ == "__main__":
    main()
