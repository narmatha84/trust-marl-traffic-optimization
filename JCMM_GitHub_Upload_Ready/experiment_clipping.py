"""
Experiment: Trust Clipping-Range Sensitivity Analysis

Tests robustness to different trust clipping ranges.

Default trust update:
    T_new = 0.9*T_old + 0.1*score

Default gating:
    low threshold  = 0.5
    high threshold = 1.5

The experiment additionally tracks:
- Minimum raw trust value before clipping
- Maximum raw trust value before clipping
- Number of lower-bound clipping events
- Number of upper-bound clipping events

This helps determine whether the clipping bounds were actually
activated during the experiment.
"""

import numpy as np
import time
import random
import torch

from environment import SumoEnvironment
from agent import Agent, ACTIONS


# ============================================================
# Trust module with configurable clipping range and tracking
# ============================================================

class SensitivityTrustModule:

    def __init__(self, clip_low=0.1, clip_high=2.0):

        self.trust = {}

        self.clip_low = clip_low
        self.clip_high = clip_high

        # Track trust values BEFORE clipping
        self.min_raw_trust = float("inf")
        self.max_raw_trust = float("-inf")

        # Count how often clipping is actually applied
        self.lower_clip_count = 0
        self.upper_clip_count = 0

    def init(self, veh_id):

        if veh_id not in self.trust:

            self.trust[veh_id] = 1.0

    def update(self, veh_id, speed, wait_time):

        self.init(veh_id)

        # ----------------------------------------------------
        # Behavioural score
        # B(v) = speed / 10 - wait_time / 50
        # ----------------------------------------------------

        efficiency = speed / 10.0
        penalty = wait_time / 50.0

        score = efficiency - penalty

        # ----------------------------------------------------
        # Trust update before clipping
        # T_new = 0.9 * T_old + 0.1 * score
        # ----------------------------------------------------

        raw_trust = (
            0.9 * self.trust[veh_id]
            + 0.1 * score
        )

        # Track raw trust range across the experiment
        self.min_raw_trust = min(
            self.min_raw_trust,
            raw_trust
        )

        self.max_raw_trust = max(
            self.max_raw_trust,
            raw_trust
        )

        # ----------------------------------------------------
        # Check whether clipping is activated
        # ----------------------------------------------------

        if raw_trust < self.clip_low:

            self.lower_clip_count += 1

        elif raw_trust > self.clip_high:

            self.upper_clip_count += 1

        # Apply clipping
        self.trust[veh_id] = np.clip(
            raw_trust,
            self.clip_low,
            self.clip_high
        )

    def get(self, veh_id):

        self.init(veh_id)

        return self.trust[veh_id]

    def get_statistics(self):

        return {
            "min_raw_trust": self.min_raw_trust,
            "max_raw_trust": self.max_raw_trust,
            "lower_clip_count": self.lower_clip_count,
            "upper_clip_count": self.upper_clip_count
        }


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

def run_sensitivity(
    clip_low,
    clip_high,
    seed=42
):

    set_seed(seed)

    trust = SensitivityTrustModule(
        clip_low=clip_low,
        clip_high=clip_high
    )

    sumo = SumoEnvironment(
        "config.sumocfg"
    )

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

            _, speed, wait, _ = (
                sumo.get_state(v)
            )

            vehicle_states[v] = (
                speed,
                wait
            )

        avg_speed = (
            np.mean(
                [s for s, w in vehicle_states.values()]
            )
            if vehicles
            else 0
        )

        avg_wait = (
            np.mean(
                [w for s, w in vehicle_states.values()]
            )
            if vehicles
            else 0
        )

        states_memory = []
        rewards = []

        # ----------------------------------------------------
        # Vehicle decisions
        # ----------------------------------------------------

        for v in vehicles:

            speed, wait = vehicle_states[v]

            s = discretize(
                speed,
                wait
            )

            # Update trust using the SAME formula as Mode 4
            trust.update(
                v,
                speed,
                wait
            )

            t = trust.get(v)

            agent.init(v)

            # ------------------------------------------------
            # Trust-based action selection
            # ------------------------------------------------

            if t < 0.5:

                # Low trust: forced exploration
                action_idx = np.random.randint(
                    len(ACTIONS)
                )

            elif t > 1.5:

                # High trust: full exploitation
                with torch.no_grad():

                    from agent import state_to_tensor

                    q_vals = agent.networks[v](
                        state_to_tensor(s)
                    )

                action_idx = int(
                    torch.argmax(q_vals).item()
                )

            else:

                # Normal trust: epsilon-greedy
                action_idx = agent.choose(
                    v,
                    s,
                    epsilon,
                    t
                )

            action = ACTIONS[action_idx]

            sumo.apply_speed(
                v,
                action
            )

            # ------------------------------------------------
            # Reward
            # ------------------------------------------------

            individual = (
                speed / 8.0
                - wait / 15.0
            )

            global_reward = (
                avg_speed / 8.0
                - avg_wait / 15.0
            )

            reward = (
                0.7 * individual
                + 0.3 * global_reward
            )

            states_memory.append(
                (
                    v,
                    s,
                    action_idx,
                    reward
                )
            )

            rewards.append(reward)

        # ----------------------------------------------------
        # Learning
        # ----------------------------------------------------

        sumo.step()

        next_vehicles = sumo.get_vehicles()

        for v, s, a, r in states_memory:

            if v in next_vehicles:

                _, speed, wait, _ = (
                    sumo.get_state(v)
                )

                ns = discretize(
                    speed,
                    wait
                )

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

        reward_history.append(
            np.mean(rewards)
            if rewards
            else 0
        )

        wait_history.append(avg_wait)
        speed_history.append(avg_speed)

        avg_reward = (
            np.mean(rewards)
            if rewards
            else 0
        )

        print(
            f"[CLIPPING] "
            f"range=({clip_low},{clip_high}) | "
            f"Seed {seed} | "
            f"Step {step:3d} | "
            f"Reward {avg_reward:.3f} | "
            f"Speed {avg_speed:.2f} | "
            f"Wait {avg_wait:.2f}"
        )

        time.sleep(0.02)

    # Get trust statistics BEFORE closing
    trust_stats = trust.get_statistics()

    sumo.close()

    return (
        reward_history,
        wait_history,
        speed_history,
        trust_stats
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    SEEDS = [
        42,
        123,
        256,
        789,
        1024
    ]

    # --------------------------------------------------------
    # Clipping configurations
    # --------------------------------------------------------

    configs = {

        # Lower upper bound
        "A: [0.0, 2.0]":
            (0.0, 2.0),

        # Default configuration
        "B: [0.1, 2.0]":
            (0.1, 2.0),

        # Higher upper bound
        "C: [0.2, 2.0]":
            (0.2, 2.0),
    }

    all_results = {
        name: []
        for name in configs
    }

    all_trust_stats = {
        name: []
        for name in configs
    }

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    for config_name, (
        clip_low,
        clip_high
    ) in configs.items():

        for seed in SEEDS:

            print(
                "\n" + "=" * 60
            )

            print(
                f"  CONFIG {config_name} "
                f"| SEED {seed}"
            )

            print(
                "=" * 60 + "\n"
            )

            r, w, s, trust_stats = run_sensitivity(
                clip_low,
                clip_high,
                seed
            )

            all_results[
                config_name
            ].append(
                (r, w, s)
            )

            all_trust_stats[
                config_name
            ].append(
                trust_stats
            )

    # --------------------------------------------------------
    # Performance summary
    # --------------------------------------------------------

    print(
        "\n\n===== CLIPPING-RANGE "
        "SENSITIVITY RESULTS (mean ± std) =====\n"
    )

    for config_name in configs:

        all_r = [
            np.mean(r)
            for r, w, s
            in all_results[config_name]
        ]

        all_w = [
            np.mean(w)
            for r, w, s
            in all_results[config_name]
        ]

        all_s = [
            np.mean(s)
            for r, w, s
            in all_results[config_name]
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
    # Trust range and clipping statistics
    # --------------------------------------------------------

    print(
        "\n===== TRUST RANGE AND CLIPPING STATISTICS =====\n"
    )

    for config_name in configs:

        stats_list = all_trust_stats[config_name]

        min_values = [
            x["min_raw_trust"]
            for x in stats_list
        ]

        max_values = [
            x["max_raw_trust"]
            for x in stats_list
        ]

        lower_clips = [
            x["lower_clip_count"]
            for x in stats_list
        ]

        upper_clips = [
            x["upper_clip_count"]
            for x in stats_list
        ]

        print(config_name)

        print(
            f"  Overall minimum raw trust: "
            f"{min(min_values):.4f}"
        )

        print(
            f"  Overall maximum raw trust: "
            f"{max(max_values):.4f}"
        )

        print(
            f"  Total lower clipping events: "
            f"{sum(lower_clips)}"
        )

        print(
            f"  Total upper clipping events: "
            f"{sum(upper_clips)}"
        )

        print()

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    np.save(
        "clipping_sensitivity_results.npy",
        all_results
    )

    np.save(
        "clipping_trust_statistics.npy",
        all_trust_stats
    )

    print(
        "Saved: clipping_sensitivity_results.npy"
    )

    print(
        "Saved: clipping_trust_statistics.npy"
    )

    # --------------------------------------------------------
    # Reward plot
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
        "Trust Clipping-Range Sensitivity"
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
        "clipping_sensitivity_reward.png",
        dpi=150
    )

    plt.show()

    print(
        "Saved: clipping_sensitivity_reward.png"
    )