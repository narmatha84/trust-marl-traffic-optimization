
import argparse
import csv
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import traci

from environment import SumoEnvironment
from agent import Agent, ACTIONS
from trust_module import TrustModule

ROOT = Path(__file__).resolve().parent

SEEDS = [
    42, 123, 256, 789, 1024,
    2048, 4096, 8192, 16384, 32768
]
RATES = [200, 400, 600]
MODES = [2, 3, 4]
MAX_ITERATIONS = 300

RESULTS = ROOT / "jcmm_comment18_high_demand.csv"


def discretize(speed, wait):
    s = min(int(speed // 2), 4)
    w = (
        0 if wait < 5 else
        1 if wait < 10 else
        2 if wait < 20 else 3
    )
    return s * 4 + w


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_config(rate):
    # Preserve original route and configuration files.
    source = ROOT / "routes.rou.xml"

    if not source.exists():
        raise FileNotFoundError(source)

    tree = ET.parse(source)
    flows = tree.findall(".//flow")

    if len(flows) != 4:
        raise ValueError(
            f"Expected 4 traffic flows, found {len(flows)}"
        )

    for flow in flows:
        flow.set("vehsPerHour", str(rate))

    routes_name = f"jcmm_comment18_routes_{rate}.rou.xml"
    config_name = f"jcmm_comment18_config_{rate}.sumocfg"

    tree.write(
        ROOT / routes_name,
        encoding="utf-8",
        xml_declaration=True
    )

    config_tree = ET.parse(ROOT / "config.sumocfg")
    route_node = config_tree.find(".//route-files")

    if route_node is None:
        raise ValueError("Missing route-files configuration")

    route_node.set("value", routes_name)

    end_node = config_tree.find(".//time/end")

    if end_node is None:
        time_node = config_tree.find(".//time")
        if time_node is None:
            time_node = ET.SubElement(
                config_tree.getroot(), "time"
            )
        end_node = ET.SubElement(time_node, "end")

    end_node.set("value", "1000")

    config_tree.write(
        ROOT / config_name,
        encoding="utf-8",
        xml_declaration=True
    )

    return ROOT / config_name


def run_one(rate, mode, seed, binary, config):
    seed_all(seed)

    env = SumoEnvironment(str(config))
    agent = Agent()
    trust = TrustModule() if mode == 4 else None

    epsilon = 0.1
    departed = 0
    arrived = 0
    teleports = 0
    started = False

    rewards = []
    waits = []
    speeds = []

    def advance():
        nonlocal departed, arrived, teleports

        env.step()

        departed += len(
            traci.simulation.getDepartedIDList()
        )
        arrived += len(
            traci.simulation.getArrivedIDList()
        )
        teleports += (
            traci.simulation.getStartingTeleportNumber()
        )

    try:
        traci.start([
            binary,
            "-c", str(config),
            "--seed", str(seed),
            "--no-step-log", "true"
        ])

        started = True
        env.driver_profiles = {}

        start_time = traci.simulation.getTime()

        for _ in range(MAX_ITERATIONS):
            advance()

            vehicles = env.get_vehicles()
            states = {}

            for vid in vehicles:
                _, speed, wait, _ = env.get_state(vid)
                states[vid] = (
                    float(speed), float(wait)
                )

            avg_speed = (
                float(np.mean([
                    s for s, w in states.values()
                ]))
                if states else 0.0
            )
            avg_wait = (
                float(np.mean([
                    w for s, w in states.values()
                ]))
                if states else 0.0
            )

            experiences = []
            step_rewards = []

            for vid, (speed, wait) in states.items():
                state = discretize(speed, wait)

                if mode == 4:
                    trust.update(vid, speed, wait)
                    t = trust.get(vid)
                else:
                    t = 1.0

                agent.init(vid)

                action_idx = agent.choose(
                    vid, state, epsilon, t
                )

                env.apply_speed(
                    vid, ACTIONS[action_idx]
                )

                individual = speed / 8 - wait / 15
                global_r = (
                    avg_speed / 8 - avg_wait / 15
                )

                reward = (
                    individual if mode == 2
                    else 0.7 * individual
                    + 0.3 * global_r
                )

                experiences.append(
                    (vid, state, action_idx, reward)
                )
                step_rewards.append(reward)

            # Original M2/M3/M4 structure:
            # two SUMO steps per iteration.
            advance()

            next_vehicles = set(env.get_vehicles())

            for vid, state, action_idx, reward in experiences:
                if vid in next_vehicles:
                    _, speed, wait, _ = env.get_state(vid)
                    next_state = discretize(speed, wait)

                    agent.update(
                        vid, state, action_idx,
                        reward, next_state,
                        0.05, 0.9
                    )

            rewards.append(
                float(np.mean(step_rewards))
                if step_rewards else 0.0
            )
            waits.append(avg_wait)
            speeds.append(avg_speed)

            epsilon = max(
                0.05, epsilon * 0.995
            )

        elapsed = (
            traci.simulation.getTime() - start_time
        )
        remaining = len(env.get_vehicles())

        return {
            "rate_per_direction": rate,
            "total_demand_per_hour": 4 * rate,
            "mode": mode,
            "seed": seed,
            "reward": float(np.mean(rewards)),
            "wait_s": float(np.mean(waits)),
            "speed_m_s": float(np.mean(speeds)),
            "departed": departed,
            "arrived": arrived,
            "remaining": remaining,
            "teleport_starts": teleports,
            "sim_seconds": elapsed,
            "throughput_veh_h": (
                arrived * 3600 / elapsed
            ),
            "arrival_fraction_of_departed": (
                arrived / departed
                if departed else float("nan")
            ),
            "departure_fraction_of_nominal_demand": (
                departed / (4 * rate * elapsed / 3600)
            )
        }

    finally:
        if started:
            env.close()


def load_rows():
    if not RESULTS.exists():
        return []

    with RESULTS.open(
        newline="", encoding="utf-8"
    ) as f:
        return list(csv.DictReader(f))


def append_row(row):
    exists = (
        RESULTS.exists()
        and RESULTS.stat().st_size > 0
    )

    with RESULTS.open(
        "a", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(row)
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rates", nargs="+", type=int,
        default=RATES
    )
    parser.add_argument(
        "--modes", nargs="+", type=int,
        choices=MODES, default=MODES
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=SEEDS
    )
    parser.add_argument(
        "--sumo", default="sumo-gui"
    )

    args = parser.parse_args()
    torch.set_num_threads(1)

    configs = {
        rate: prepare_config(rate)
        for rate in args.rates
    }

    completed = {
        (
            int(row["rate_per_direction"]),
            int(row["mode"]),
            int(row["seed"])
        )
        for row in load_rows()
    }

    for rate in args.rates:
        for mode in args.modes:
            for seed in args.seeds:
                key = (rate, mode, seed)

                if key in completed:
                    print(
                        f"REUSED rate={rate} "
                        f"mode={mode} seed={seed}"
                    )
                    continue

                print(
                    f"RUNNING rate={rate} "
                    f"mode={mode} seed={seed}",
                    flush=True
                )

                result = run_one(
                    rate, mode, seed,
                    args.sumo, configs[rate]
                )

                append_row(result)
                completed.add(key)

                print(
                    f"  Reward={result['reward']:.4f} | "
                    f"Wait={result['wait_s']:.3f}s | "
                    f"Speed={result['speed_m_s']:.3f}m/s | "
                    f"Throughput={result['throughput_veh_h']:.1f} | "
                    f"Departed={result['departed']} | "
                    f"Arrived={result['arrived']} | "
                    f"Teleports={result['teleport_starts']}",
                    flush=True
                )

    print("\n" + "=" * 65)
    print("COMMENT 18: HIGH-DEMAND SUMMARY")
    print("=" * 65)

    rows = load_rows()

    metrics = [
        "reward", "wait_s", "speed_m_s",
        "throughput_veh_h", "departed", "arrived",
        "remaining", "teleport_starts",
        "departure_fraction_of_nominal_demand"
    ]

    for rate in args.rates:
        for mode in args.modes:
            matches = [
                row for row in rows
                if int(row["rate_per_direction"]) == rate
                and int(row["mode"]) == mode
                and int(row["seed"]) in args.seeds
            ]

            if not matches:
                continue

            print(
                f"\nRate={rate} veh/h/direction | "
                f"Mode={mode} | n={len(matches)}"
            )

            for metric in metrics:
                values = np.array([
                    float(row[metric])
                    for row in matches
                ])

                std = (
                    values.std(ddof=1)
                    if len(values) > 1 else 0.0
                )

                print(
                    f"  {metric}: "
                    f"{values.mean():.4f} "
                    f"+/- {std:.4f}"
                )

    print(f"\nSaved: {RESULTS.name}")


if __name__ == "__main__":
    main()
