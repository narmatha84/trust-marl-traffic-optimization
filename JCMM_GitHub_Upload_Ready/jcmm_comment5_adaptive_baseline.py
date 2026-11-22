
"""
JCMM Comment 5
Pressure-Based Adaptive Speed Control (PBASC)

Uses the EXISTING MaxPressureController without changing its
pressure calculation or speed-selection rules.

Evaluation aligned with original Mode 4:
- 300 control iterations
- 2 SUMO simulation steps per iteration
- Same driver heterogeneity via SumoEnvironment.apply_speed()
- Same individual + global reward expression
- 10 independent Python random seeds

Run:
    python jcmm_comment5_pressure_baseline.py
"""

import random
import numpy as np
import torch
import traci

from environment import SumoEnvironment

# Your original archive names this file "max_pressue.py".
# If yours is named "max_pressure.py", change this import only.
from max_pressue import MaxPressureController


SEEDS = [42, 123, 256, 789, 1024, 2025, 2026, 314, 777, 999]
MAX_STEPS = 300
CONFIG_FILE = "config.sumocfg"

RESULT_FILE = "jcmm_comment5_pressure_results.npz"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one_seed(seed):
    set_seed(seed)

    controller = MaxPressureController()
    sumo = SumoEnvironment(CONFIG_FILE)

    reward_history = []
    wait_history = []
    speed_history = []

    action_counts = {2.0: 0, 4.0: 0, 8.0: 0}
    unknown_routes = set()

    try:
        sumo.reset()

        for step in range(MAX_STEPS):

            # First simulation step: identical position
            # in the control loop to original Mode 4.
            sumo.step()

            vehicles = list(sumo.get_vehicles())
            vehicle_states = {}

            for vehicle_id in vehicles:
                _, speed, wait, _ = sumo.get_state(vehicle_id)
                vehicle_states[vehicle_id] = (speed, wait)

            if vehicle_states:
                avg_speed = float(np.mean([
                    s for s, w in vehicle_states.values()
                ]))
                avg_wait = float(np.mean([
                    w for s, w in vehicle_states.values()
                ]))
            else:
                avg_speed = 0.0
                avg_wait = 0.0

            # Original pressure calculation, unchanged.
            pressure = controller.compute_pressure(vehicles)

            global_reward = avg_speed / 8.0 - avg_wait / 15.0

            rewards = []

            for vehicle_id in vehicles:
                speed, wait = vehicle_states[vehicle_id]

                approach = controller.get_approach(vehicle_id)

                if approach == "unknown":
                    unknown_routes.add(
                        traci.vehicle.getRouteID(vehicle_id)
                    )

                # Original controller action, unchanged.
                action = float(
                    controller.get_speed_command(
                        vehicle_id, pressure
                    )
                )

                action_counts[action] = (
                    action_counts.get(action, 0) + 1
                )

                # Original driver heterogeneity, unchanged.
                sumo.apply_speed(vehicle_id, action)

                # Explicitly match Mode 4 reward definition.
                individual_reward = (
                    speed / 8.0 - wait / 15.0
                )

                reward = (
                    0.7 * individual_reward
                    + 0.3 * global_reward
                )

                rewards.append(reward)

            # Second SUMO step: aligns with Mode 4's
            # online-update transition step.
            # PBASC has no neural network to update.
            sumo.step()

            mean_reward = (
                float(np.mean(rewards)) if rewards else 0.0
            )

            reward_history.append(mean_reward)
            wait_history.append(avg_wait)
            speed_history.append(avg_speed)

            if step % 50 == 0 or step == MAX_STEPS - 1:
                print(
                    f"Seed {seed} | Step {step:3d} | "
                    f"Reward {mean_reward:.4f} | "
                    f"Wait {avg_wait:.4f} | "
                    f"Speed {avg_speed:.4f}"
                )

    finally:
        sumo.close()

    return {
        "reward": np.array(reward_history),
        "wait": np.array(wait_history),
        "speed": np.array(speed_history),
        "actions": action_counts,
        "unknown_routes": sorted(unknown_routes),
    }


def main():
    print("=" * 65)
    print("JCMM COMMENT 5 — PRESSURE-BASED ADAPTIVE SPEED CONTROL")
    print("=" * 65)

    results = []

    for seed in SEEDS:
        print(f"\nRunning seed {seed}")
        results.append(run_one_seed(seed))

    rewards = np.array([r["reward"] for r in results])
    waits = np.array([r["wait"] for r in results])
    speeds = np.array([r["speed"] for r in results])

    # One aggregate value per seed.
    seed_rewards = rewards.mean(axis=1)
    seed_waits = waits.mean(axis=1)
    seed_speeds = speeds.mean(axis=1)

    print("\n" + "=" * 65)
    print("FINAL RESULTS — 10 SEEDS")
    print("=" * 65)

    for label, values, unit in [
        ("Average Reward", seed_rewards, ""),
        ("Average Waiting Time", seed_waits, "s"),
        ("Average Speed", seed_speeds, "m/s"),
    ]:
        print(
            f"{label}: "
            f"{values.mean():.4f} ± "
            f"{values.std(ddof=1):.4f} {unit}"
        )

    total_actions = {}

    for result in results:
        for action, count in result["actions"].items():
            total_actions[action] = (
                total_actions.get(action, 0) + count
            )

    total = sum(total_actions.values())

    print("\nAction distribution:")
    for action in sorted(total_actions):
        percentage = (
            100 * total_actions[action] / total
            if total else 0
        )
        print(
            f"  {action:g} m/s: "
            f"{total_actions[action]} "
            f"({percentage:.2f}%)"
        )

    unknown = sorted(set(
        route
        for result in results
        for route in result["unknown_routes"]
    ))

    print("\nUnknown route IDs:", unknown if unknown else "None")

    np.savez(
        RESULT_FILE,
        seeds=np.array(SEEDS),
        rewards=rewards,
        waits=waits,
        speeds=speeds,
        seed_rewards=seed_rewards,
        seed_waits=seed_waits,
        seed_speeds=seed_speeds,
    )

    print(f"\nSaved: {RESULT_FILE}")
    print("Original files were not modified.")


if __name__ == "__main__":
    main()
