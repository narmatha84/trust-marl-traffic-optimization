"""
Max-Pressure Baseline for Single Four-Way Intersection

Max-pressure is a well-established adaptive traffic control algorithm.
At each step, it measures the number of waiting vehicles on each
approach and assigns the highest speed command to vehicles on the
most congested approach.

Unlike Mode 1 (Fixed Rule), max-pressure adapts to actual traffic
conditions at each step. Unlike Modes 2-4, it requires no learning.

Reference:
    Varaiya, P. (2013). Max pressure control of a network of
    signalized intersections. Transportation Research Part C:
    Emerging Technologies, 36, 177-195.
    DOI: 10.1016/j.trc.2013.08.014

Usage:
    python max_pressure.py

Output:
    max_pressure_results.npy
    max_pressure_reward.png
    max_pressure_wait.png
    max_pressure_speed.png
"""

import numpy as np
import time
import random
import torch
import matplotlib.pyplot as plt

from environment import SumoEnvironment
import traci


# ============================================================
# CONFIGURATION
# ============================================================

SEEDS       = [42, 123, 256, 789, 1024]
MAX_STEPS   = 300
CONFIG_FILE = "config.sumocfg"

# Speed levels available to max-pressure controller
# Same action space as learning agents for fair comparison
SPEED_HIGH   = 8.0   # m/s — given to high-pressure approach
SPEED_MEDIUM = 4.0   # m/s — given to medium-pressure approach
SPEED_LOW    = 2.0   # m/s — given to low-pressure approach


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# MAX-PRESSURE CONTROLLER
# ============================================================

class MaxPressureController:
    """
    Max-Pressure adaptive traffic controller.

    At each simulation step:
        1. Count waiting vehicles on each approach (N, S, E, W)
        2. Compute pressure = number of waiting vehicles per approach
        3. Assign speed commands inversely proportional to pressure
           (high pressure approach gets high speed to clear faster)

    This is a purely reactive, rule-based controller.
    No learning, no neural network, no trust mechanism.
    """

    def __init__(self):
        # Approach direction labels
        self.approaches = ["north", "south", "east", "west"]

    def get_approach(self, veh_id):
        """
        Determine which approach a vehicle is on based on
        its current route ID or edge.
        """
        
        try:
           route_id = traci.vehicle.getRouteID(veh_id)
        
           if "r_tb" in route_id:
             return "north"
           elif "r_bt" in route_id:
             return "south"
           elif "r_lr" in route_id:
             return "east"
           elif "r_rl" in route_id:
             return "west"
           else:
             return "unknown"
        except:
           return "unknown"

    def compute_pressure(self, vehicles):
        """
        Compute pressure for each approach.
        Pressure = number of waiting vehicles on that approach.
        """
        pressure = {
            "north":   0,
            "south":   0,
            "east":    0,
            "west":    0,
            "unknown": 0
        }

        for veh_id in vehicles:
            try:
                wait = traci.vehicle.getWaitingTime(veh_id)
                # Count vehicle as waiting if wait > 0
                if wait > 0:
                    approach = self.get_approach(veh_id)
                    pressure[approach] += 1
            except:
                continue

        return pressure

    def get_speed_command(self, veh_id, pressure):
        """
        Assign speed based on approach pressure.

        High pressure approach → high speed (clear the queue)
        Low pressure approach  → low speed (yield to congested)
        """
        approach = self.get_approach(veh_id)

        if approach == "unknown":
            return SPEED_MEDIUM

        # Get pressure values for all known approaches
        known_pressures = {
            k: v for k, v in pressure.items()
            if k != "unknown"
        }

        if not known_pressures:
            return SPEED_MEDIUM

        max_pressure = max(known_pressures.values())
        min_pressure = min(known_pressures.values())
        veh_pressure = pressure.get(approach, 0)

        # Assign speed tier based on pressure rank
        if max_pressure == min_pressure:
            # All approaches equal — medium speed
            return SPEED_MEDIUM
        elif veh_pressure == max_pressure:
            # Highest pressure → highest speed (priority clearance)
            return SPEED_HIGH
        elif veh_pressure == min_pressure:
            # Lowest pressure → lowest speed (yield)
            return SPEED_LOW
        else:
            # Medium pressure → medium speed
            return SPEED_MEDIUM


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_max_pressure(seed=42):
    """
    Run one max-pressure experiment.

    Returns reward, wait, and speed histories.
    """

    set_seed(seed)

    controller = MaxPressureController()
    sumo       = SumoEnvironment(CONFIG_FILE)

    sumo.reset()

    reward_history = []
    wait_history   = []
    speed_history  = []

    for step in range(MAX_STEPS):

        sumo.step()

        vehicles = sumo.get_vehicles()

        # ----------------------------------------------------
        # Collect vehicle states
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Compute approach pressures
        # ----------------------------------------------------

        pressure = controller.compute_pressure(vehicles)

        # ----------------------------------------------------
        # Apply max-pressure speed commands
        # ----------------------------------------------------

        rewards = []

        for v in vehicles:
            speed, wait = vehicle_states[v]

            # Get speed command from max-pressure controller
            action = controller.get_speed_command(v, pressure)

            # Apply speed (uses driver profiles from environment)
            sumo.apply_speed(v, action)

            # Same reward function as other modes for fair comparison
            reward = speed / 8.0 - wait / 15.0
            rewards.append(reward)

        # ----------------------------------------------------
        # Record metrics
        # ----------------------------------------------------

        mean_reward = np.mean(rewards) if rewards else 0.0

        reward_history.append(mean_reward)
        wait_history.append(avg_wait)
        speed_history.append(avg_speed)

        print(
            f"[MAX-PRESSURE] Seed {seed} | "
            f"Step {step:3d} | "
            f"Reward {mean_reward:.3f} | "
            f"Speed {avg_speed:.2f} | "
            f"Wait {avg_wait:.2f} | "
            f"Pressure N={pressure['north']} "
            f"S={pressure['south']} "
            f"E={pressure['east']} "
            f"W={pressure['west']}"
        )

        time.sleep(0.02)

    sumo.close()

    return reward_history, wait_history, speed_history


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  MAX-PRESSURE BASELINE EXPERIMENT")
    print("="*60 + "\n")

    all_results = []

    # --------------------------------------------------------
    # Run 5 seeds
    # Note: max-pressure is deterministic given fixed traffic
    # but driver profiles introduce stochasticity so we run
    # multiple seeds for fair comparison with learning modes
    # --------------------------------------------------------

    for seed in SEEDS:

        print(f"\n{'='*60}")
        print(f"  MAX-PRESSURE | SEED {seed}")
        print(f"{'='*60}\n")

        r, w, s = run_max_pressure(seed)
        all_results.append((r, w, s))

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n\n================ MAX-PRESSURE RESULTS ================\n")

    all_r = [np.mean(r) for r, w, s in all_results]
    all_w = [np.mean(w) for r, w, s in all_results]
    all_s = [np.mean(s) for r, w, s in all_results]

    r_mean, r_std = np.mean(all_r), np.std(all_r)
    w_mean, w_std = np.mean(all_w), np.std(all_w)
    s_mean, s_std = np.mean(all_s), np.std(all_s)

    print(f"Mode MP — Max-Pressure Control")
    print(f"  Avg Reward : {r_mean:.3f} ± {r_std:.3f}")
    print(f"  Avg Wait   : {w_mean:.3f} ± {w_std:.3f} s")
    print(f"  Avg Speed  : {s_mean:.3f} ± {s_std:.3f} m/s")
    print()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save("max_pressure_results.npy", all_results)
    print("Saved: max_pressure_results.npy")

    # --------------------------------------------------------
    # Plot reward curve
    # --------------------------------------------------------

    histories    = [r for r, w, s in all_results]
    mean_curve   = np.mean(histories, axis=0)
    std_curve    = np.std(histories, axis=0)
    steps        = np.arange(len(mean_curve))

    plt.figure(figsize=(10, 5))

    plt.plot(
        steps, mean_curve,
        label="Max-Pressure",
        color="#9C27B0",
        linewidth=2
    )

    plt.fill_between(
        steps,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.2,
        color="#9C27B0"
    )

    plt.title(
        "Max-Pressure Baseline — Average Reward "
        "(mean ± std, 5 seeds)"
    )
    plt.xlabel("Simulation Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("max_pressure_reward.png", dpi=300)
    plt.show()
    print("Saved: max_pressure_reward.png")

    # --------------------------------------------------------
    # Summary comparison reminder
    # --------------------------------------------------------

    print("\n" + "="*60)
    print("Add these results to your comparison table:")
    print("="*60)
    print(f"  Max-Pressure | {r_mean:.3f} ± {r_std:.3f} | "
          f"{w_mean:.3f} ± {w_std:.3f} s | "
          f"{s_mean:.3f} ± {s_std:.3f} m/s")
    print("="*60)