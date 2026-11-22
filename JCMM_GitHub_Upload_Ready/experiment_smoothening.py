"""
Experiment: Trust Smoothing-Factor Sensitivity Analysis

Tests whether TA-MARL performance is robust to different
trust smoothing factors.

Trust update:
    T_new = alpha * T_old + (1-alpha) * score

Configurations:
    A: alpha = 0.70
    B: alpha = 0.90  <-- default
    C: alpha = 0.95

All other trust parameters remain unchanged:
    clipping range = [0.1, 2.0]
    low threshold  = 0.5
    high threshold = 1.5

3 seeds each.
"""

import numpy as np
import time
import random
import torch

from environment import SumoEnvironment
from agent import Agent, ACTIONS


# ============================================================
# Trust module with configurable smoothing factor
# ============================================================

class SensitivityTrustModule:

    def __init__(self, alpha=0.9):
        self.trust = {}
        self.alpha = alpha

    def init(self, veh_id):
        if veh_id not in self.trust:
            self.trust[veh_id] = 1.0

    def update(self, veh_id, speed, wait_time):

        self.init(veh_id)

        efficiency = speed / 10
        penalty = wait_time / 50

        score = efficiency - penalty

        self.trust[veh_id] = (
            self.alpha * self.trust[veh_id]
            + (1 - self.alpha) * score
        )

        self.trust[veh_id] = np.clip(
            self.trust[veh_id],
            0.1,
            2.0
        )

    def get(self, veh_id):
        self.init(veh_id)
        return self.trust[veh_id]


# ============================================================
# State discretization
# ============================================================

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


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Run experiment
# ============================================================

def run_sensitivity(alpha, seed=42):

    set_seed(seed)

    trust = SensitivityTrustModule(alpha=alpha)

    sumo = SumoEnvironment("config.sumocfg")

    agent = Agent()

    epsilon = 0.1

    sumo.reset()

    MAX_STEPS = 300

    reward_history = []
    wait_history = []
    speed_history = []

    for step in range(MAX_STEPS):

        sumo.step()

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
        rewards = []

        # ----------------------------------------------------
        # Vehicle decisions
        # ----------------------------------------------------

        for v in vehicles:

            speed, wait = vehicle_states[v]

            s = discretize(speed, wait)

            # Update trust using selected smoothing factor
            trust.update(v, speed, wait)

            t = trust.get(v)

            agent.init(v)

            # ------------------------------------------------
            # Same gating thresholds as original model
            # ------------------------------------------------

            if t < 0.5:

                action_idx = np.random.randint(len(ACTIONS))

            elif t > 1.5:

                with torch.no_grad():

                    from agent import state_to_tensor

                    q_vals = agent.networks[v](
                        state_to_tensor(s)
                    )

                action_idx = int(
                    torch.argmax(q_vals).item()
                )

            else:

                action_idx = agent.choose(
                    v,
                    s,
                    epsilon,
                    t
                )

            action = ACTIONS[action_idx]

            sumo.apply_speed(v, action)

            # ------------------------------------------------
            # Reward
            # ------------------------------------------------

            individual = speed / 8 - wait / 15

            global_reward = (
                avg_speed / 8
                - avg_wait / 15
            )

            reward = (
                0.7 * individual
                + 0.3 * global_reward
            )

            states_memory.append(
                (v, s, action_idx, reward)
            )

            rewards.append(reward)

        # ----------------------------------------------------
        # Learning step
        # ----------------------------------------------------

        sumo.step()

        next_vehicles = sumo.get_vehicles()

        for v, s, a, r in states_memory:

            if v in next_vehicles:

                _, speed, wait, _ = sumo.get_state(v)

                ns = discretize(speed, wait)

                agent.update(
                    v,
                    s,
                    a,
                    r,
                    ns,
                    alpha=0.05,
                    gamma=0.9
                )

        epsilon = max(
            0.05,
            epsilon * 0.995
        )

        # ----------------------------------------------------
        # Record metrics
        # ----------------------------------------------------

        reward_history.append(
            np.mean(rewards) if rewards else 0
        )

        wait_history.append(avg_wait)

        speed_history.append(avg_speed)

        print(
            f"[SMOOTHING] alpha={alpha} | "
            f"Seed {seed} | "
            f"Step {step:3d} | "
            f"Reward {np.mean(rewards):.3f} | "
            f"Speed {avg_speed:.2f} | "
            f"Wait {avg_wait:.2f}"
        )

        time.sleep(0.02)

    sumo.close()

    return (
        reward_history,
        wait_history,
        speed_history
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    SEEDS = [42, 123, 256, 789, 1024]

    # --------------------------------------------------------
    # Smoothing-factor configurations
    # --------------------------------------------------------

    configs = {

        "A: alpha=0.70": 0.70,

        "B: alpha=0.90": 0.90,   # DEFAULT

        "C: alpha=0.95": 0.95,
    }

    all_results = {
        name: []
        for name in configs
    }

    # --------------------------------------------------------
    # Run all configurations
    # --------------------------------------------------------

    for config_name, alpha in configs.items():

        for seed in SEEDS:

            print("\n" + "=" * 60)

            print(
                f"  CONFIG {config_name} | SEED {seed}"
            )

            print("=" * 60 + "\n")

            r, w, s = run_sensitivity(
                alpha,
                seed
            )

            all_results[config_name].append(
                (r, w, s)
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n\n===== SMOOTHING-FACTOR "
        "SENSITIVITY RESULTS (mean ± std) =====\n"
    )

    for config_name in configs:

        all_r = [
            np.mean(r)
            for r, w, s in all_results[config_name]
        ]

        all_w = [
            np.mean(w)
            for r, w, s in all_results[config_name]
        ]

        all_s = [
            np.mean(s)
            for r, w, s in all_results[config_name]
        ]

        r_m = np.mean(all_r)
        r_s = np.std(all_r)

        w_m = np.mean(all_w)
        w_s = np.std(all_w)

        s_m = np.mean(all_s)
        s_s = np.std(all_s)

        print(config_name)

        print(
            f"  Avg Reward : "
            f"{r_m:.3f} ± {r_s:.3f}"
        )

        print(
            f"  Avg Wait   : "
            f"{w_m:.3f} ± {w_s:.3f} s"
        )

        print(
            f"  Avg Speed  : "
            f"{s_m:.3f} ± {s_s:.3f} m/s"
        )

        print()

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    np.save(
        "smoothing_sensitivity_results.npy",
        all_results
    )

    print(
        "Saved: smoothing_sensitivity_results.npy"
    )

    # --------------------------------------------------------
    # Plot reward
    # --------------------------------------------------------

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    for config_name in configs:

        histories = [
            r
            for r, w, s
            in all_results[config_name]
        ]

        mean_curve = np.mean(
            histories,
            axis=0
        )

        std_curve = np.std(
            histories,
            axis=0
        )

        steps = np.arange(
            len(mean_curve)
        )

        plt.plot(
            steps,
            mean_curve,
            label=config_name,
            linewidth=2
        )

        plt.fill_between(
            steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2
        )

    plt.title(
        "Trust Smoothing-Factor Sensitivity"
    )

    plt.xlabel(
        "Simulation Steps"
    )

    plt.ylabel(
        "Average Reward"
    )

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "smoothing_sensitivity_reward.png",
        dpi=150
    )

    plt.show()

    print(
        "Saved: smoothing_sensitivity_reward.png"
    )