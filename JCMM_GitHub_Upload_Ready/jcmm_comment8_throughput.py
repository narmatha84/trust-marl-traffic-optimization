
import random
import numpy as np
import torch
import traci

from environment import SumoEnvironment
from agent import Agent, ACTIONS
from trust_module import TrustModule

SEEDS = [42, 123, 256, 789, 1024,
         2048, 4096, 8192, 16384, 32768]

MAX_STEPS = 300
MODES = [1, 2, 3, 4]


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(mode, seed):
    set_seed(seed)

    agent = Agent()
    trust = TrustModule() if mode == 4 else None
    sumo = SumoEnvironment("config.sumocfg")

    epsilon = 0.1
    alpha = 0.05
    gamma = 0.9

    completed = 0
    departed = 0
    collisions = 0
    arrived_ids = set()
    departure_ids = set()
    completed_travel_times = []

    started = False

    def record_traffic_events():
        nonlocal completed, departed, collisions

        new_departures = traci.simulation.getDepartedIDList()
        new_arrivals = traci.simulation.getArrivedIDList()

        departed += len(new_departures)
        completed += len(new_arrivals)
        collisions += traci.simulation.getCollidingVehiclesNumber()

        departure_ids.update(new_departures)

        for vid in new_arrivals:
            arrived_ids.add(vid)

            # Vehicle is no longer accessible through
            # traci.vehicle after arriving, so use
            # recorded departure time where available.
            # This experiment's primary metric is throughput.

    try:
        sumo.reset()
        started = True
        start_time = traci.simulation.getTime()

        for step in range(MAX_STEPS):

            # First SUMO step: same as original code
            sumo.step()
            record_traffic_events()

            vehicles = sumo.get_vehicles()

            vehicle_states = {}
            for v in vehicles:
                _, speed, wait, _ = sumo.get_state(v)
                vehicle_states[v] = (speed, wait)

            avg_speed = (
                np.mean([s for s, w in vehicle_states.values()])
                if vehicles else 0
            )
            avg_wait = (
                np.mean([w for s, w in vehicle_states.values()])
                if vehicles else 0
            )

            states_memory = []

            for v in vehicles:
                speed, wait = vehicle_states[v]
                s = discretize(speed, wait)

                if mode == 1:
                    if wait > 20:
                        action = 8
                    elif wait > 10:
                        action = 6
                    elif wait > 5:
                        action = 4
                    else:
                        action = 2

                    sumo.apply_speed(v, action)

                elif mode == 2:
                    agent.init(v)
                    action_idx = agent.choose(
                        v, s, epsilon, trust=1.0
                    )
                    sumo.apply_speed(v, ACTIONS[action_idx])

                    reward = speed / 8 - wait / 15
                    states_memory.append(
                        (v, s, action_idx, reward)
                    )

                elif mode == 3:
                    agent.init(v)
                    action_idx = agent.choose(
                        v, s, epsilon, trust=1.0
                    )
                    sumo.apply_speed(v, ACTIONS[action_idx])

                    individual = speed / 8 - wait / 15
                    global_reward = (
                        avg_speed / 8 - avg_wait / 15
                    )
                    reward = (
                        0.7 * individual + 0.3 * global_reward
                    )
                    states_memory.append(
                        (v, s, action_idx, reward)
                    )

                elif mode == 4:
                    trust.update(v, speed, wait)
                    t = trust.get(v)

                    agent.init(v)
                    action_idx = agent.choose(
                        v, s, epsilon, t
                    )
                    sumo.apply_speed(v, ACTIONS[action_idx])

                    individual = speed / 8 - wait / 15
                    global_reward = (
                        avg_speed / 8 - avg_wait / 15
                    )
                    reward = (
                        0.7 * individual + 0.3 * global_reward
                    )
                    states_memory.append(
                        (v, s, action_idx, reward)
                    )

            # Preserve original simulation structure:
            # modes 2-4 advance twice per iteration.
            if mode != 1:
                sumo.step()
                record_traffic_events()

                next_vehicles = set(sumo.get_vehicles())

                for v, s, a, r in states_memory:
                    if v in next_vehicles:
                        _, speed, wait, _ = sumo.get_state(v)
                        ns = discretize(speed, wait)
                        agent.update(
                            v, s, a, r, ns, alpha, gamma
                        )

            epsilon = max(0.05, epsilon * 0.995)

        elapsed = traci.simulation.getTime() - start_time
        remaining = len(sumo.get_vehicles())

        return {
            "mode": mode,
            "seed": seed,
            "completed": completed,
            "departed": departed,
            "remaining": remaining,
            "collisions": collisions,
            "elapsed": elapsed,
            "throughput_per_hour": (
                completed * 3600 / elapsed
                if elapsed > 0 else float("nan")
            ),
            "completion_fraction": (
                completed / departed
                if departed > 0 else float("nan")
            ),
        }

    finally:
        if started:
            sumo.close()


def main():
    results = {mode: [] for mode in MODES}

    print("\nCOMMENT 8: INDEPENDENT THROUGHPUT EVALUATION")
    print("Original algorithms; no reward modifications\n")

    for mode in MODES:
        for seed in SEEDS:
            print(f"Running Mode {mode}, Seed {seed} ...",
                  flush=True)

            result = run_experiment(mode, seed)
            results[mode].append(result)

            print(
                f"  Completed={result['completed']} | "
                f"Departed={result['departed']} | "
                f"Remaining={result['remaining']} | "
                f"Duration={result['elapsed']:.0f}s | "
                f"Throughput={result['throughput_per_hour']:.2f} veh/h | "
                f"Collisions={result['collisions']}"
            )

    print("\n" + "=" * 65)
    print("COMMENT 8: FINAL SUMMARY (10 SEEDS)")
    print("=" * 65)

    for mode in MODES:
        rows = results[mode]

        for key, label in [
            ("completed", "Completed trips"),
            ("departed", "Departed vehicles"),
            ("remaining", "Remaining vehicles"),
            ("elapsed", "Simulation duration (s)"),
            ("throughput_per_hour", "Throughput (veh/h)"),
            ("completion_fraction", "Completion fraction"),
            ("collisions", "Collision events"),
        ]:
            values = np.array(
                [row[key] for row in rows], dtype=float
            )
            print(
                f"Mode {mode} | {label}: "
                f"{np.mean(values):.4f} "
                f"+/- {np.std(values):.4f}"
            )

        print("-" * 65)

    # Preserve per-seed results for the paper.
    import csv

    with open(
        "jcmm_comment8_throughput.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[1][0].keys())
        )
        writer.writeheader()

        for mode in MODES:
            writer.writerows(results[mode])

    print("\nSaved: jcmm_comment8_throughput.csv")


if __name__ == "__main__":
    main()
