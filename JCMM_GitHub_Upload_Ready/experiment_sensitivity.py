
"""
Experiment 5: Trust Threshold Sensitivity Analysis

Tests whether TA-MARL performance is robust to different trust
threshold values.

Configurations:

    A: low=0.3, high=1.7
    B: low=0.5, high=1.5  <-- DEFAULT / ORIGINAL MODE 4
    C: low=0.7, high=1.3

IMPORTANT:
    - agent.py is NOT modified.
    - trust_module.py is NOT modified.
    - Same SUMO environment.
    - Same DQN Agent.
    - Same TrustModule.
    - Same reward function.
    - Same DQN learning procedure.
    - Same five random seeds.

The only thing changed between A/B/C is the trust gating thresholds.
"""


import numpy as np
import time
import random
import torch

from environment import SumoEnvironment
from agent import Agent, ACTIONS, state_to_tensor
from trust_module import TrustModule


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
# Trust-gated action selection
#
# This is the SAME logic as Agent.choose() in agent.py.
#
# The ONLY difference is that low_thresh and high_thresh
# are supplied as parameters.
#
# agent.py itself is NOT modified.
# ============================================================

def choose_with_thresholds(
    agent,
    veh_id,
    state,
    epsilon,
    trust,
    low_thresh,
    high_thresh
):

    # --------------------------------------------------------
    # Low trust -> forced exploration
    # Same behavior as Agent.choose()
    # --------------------------------------------------------

    if trust < low_thresh:

        return np.random.randint(len(ACTIONS))


    # --------------------------------------------------------
    # High trust -> pure exploitation
    # Same behavior as Agent.choose()
    # --------------------------------------------------------

    if trust > high_thresh:

        with torch.no_grad():

            q_vals = agent.networks[veh_id](
                state_to_tensor(state)
            )

        return int(
            torch.argmax(q_vals).item()
        )


    # --------------------------------------------------------
    # Normal trust range -> epsilon-greedy
    # Same behavior as Agent.choose()
    # --------------------------------------------------------

    if np.random.rand() < epsilon:

        return np.random.randint(len(ACTIONS))


    with torch.no_grad():

        q_vals = agent.networks[veh_id](
            state_to_tensor(state)
        )

    return int(
        torch.argmax(q_vals).item()
    )


# ============================================================
# Run one sensitivity experiment
# ============================================================

def run_sensitivity(
    low_thresh,
    high_thresh,
    seed=42
):

    set_seed(seed)

    trust = TrustModule()

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


    # ========================================================
    # Simulation loop
    # ========================================================

    for step in range(MAX_STEPS):


        # ----------------------------------------------------
        # Advance SUMO
        # ----------------------------------------------------

        sumo.step()

        vehicles = sumo.get_vehicles()


        # ----------------------------------------------------
        # Collect current vehicle states
        # ----------------------------------------------------

        vehicle_states = {}

        for v in vehicles:

            _, speed, wait, _ = sumo.get_state(v)

            vehicle_states[v] = (
                speed,
                wait
            )


        # ----------------------------------------------------
        # Global metrics
        # ----------------------------------------------------

        avg_speed = (
            np.mean(
                [
                    speed
                    for speed, wait
                    in vehicle_states.values()
                ]
            )
            if vehicles
            else 0
        )

        avg_wait = (
            np.mean(
                [
                    wait
                    for speed, wait
                    in vehicle_states.values()
                ]
            )
            if vehicles
            else 0
        )


        states_memory = []
        rewards = []


        # ====================================================
        # Vehicle decisions
        # ====================================================

        for v in vehicles:

            speed, wait = vehicle_states[v]

            s = discretize(
                speed,
                wait
            )


            # ------------------------------------------------
            # Update trust
            # EXACTLY the same TrustModule as Mode 4
            # ------------------------------------------------

            trust.update(
                v,
                speed,
                wait
            )

            t = trust.get(v)


            # ------------------------------------------------
            # Initialize vehicle's DQN
            # EXACTLY as in Agent.choose()
            # ------------------------------------------------

            agent.init(v)


            # ------------------------------------------------
            # Trust-gated action selection
            #
            # A: 0.3 / 1.7
            # B: 0.5 / 1.5
            # C: 0.7 / 1.3
            #
            # Same logic in every configuration.
            # Only thresholds change.
            # ------------------------------------------------

            action_idx = choose_with_thresholds(
                agent=agent,
                veh_id=v,
                state=s,
                epsilon=epsilon,
                trust=t,
                low_thresh=low_thresh,
                high_thresh=high_thresh
            )


            # ------------------------------------------------
            # Convert action index to speed
            # ------------------------------------------------

            action = ACTIONS[action_idx]

            sumo.apply_speed(
                v,
                action
            )


            # ------------------------------------------------
            # Reward
            #
            # EXACTLY the same reward as Mode 4
            # ------------------------------------------------

            individual = (
                speed / 8
                - wait / 15
            )

            global_reward = (
                avg_speed / 8
                - avg_wait / 15
            )

            reward = (
                0.7 * individual
                + 0.3 * global_reward
            )


            # ------------------------------------------------
            # Store transition
            # ------------------------------------------------

            states_memory.append(
                (
                    v,
                    s,
                    action_idx,
                    reward
                )
            )

            rewards.append(reward)


        # ====================================================
        # DQN learning
        #
        # EXACTLY the same procedure as Mode 4
        # ====================================================

        sumo.step()

        next_vehicles = sumo.get_vehicles()


        for v, s, a, r in states_memory:

            if v in next_vehicles:

                _, speed, wait, _ = sumo.get_state(v)

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


        # ----------------------------------------------------
        # Epsilon decay
        # ----------------------------------------------------

        epsilon = max(
            0.05,
            epsilon * 0.995
        )


        # ====================================================
        # Record metrics
        # ====================================================

        reward_history.append(
            np.mean(rewards)
            if rewards
            else 0
        )

        wait_history.append(
            avg_wait
        )

        speed_history.append(
            avg_speed
        )


        # ----------------------------------------------------
        # Progress output
        # ----------------------------------------------------

        avg_reward = (
            np.mean(rewards)
            if rewards
            else 0
        )

        print(
            f"[SENSITIVITY] "
            f"thresh=({low_thresh},{high_thresh}) | "
            f"Seed {seed} | "
            f"Step {step:3d} | "
            f"Reward {avg_reward:.3f} | "
            f"Speed {avg_speed:.2f} | "
            f"Wait {avg_wait:.2f}"
        )

        time.sleep(0.02)


    # ========================================================
    # Close SUMO
    # ========================================================

    sumo.close()


    return (
        reward_history,
        wait_history,
        speed_history
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":


    # --------------------------------------------------------
    # Same five seeds as the main experiment
    # --------------------------------------------------------

    SEEDS = [
        42,
        123,
        256,
        789,
        1024
    ]


    # --------------------------------------------------------
    # Threshold configurations
    # --------------------------------------------------------

    configs = {

        "A: low=0.3, high=1.7":
            (0.3, 1.7),

        "B: low=0.5, high=1.5":
            (0.5, 1.5),       # ORIGINAL MODE 4

        "C: low=0.7, high=1.3":
            (0.7, 1.3),
    }


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    all_results = {
        name: []
        for name in configs
    }


    # ========================================================
    # Run all configurations
    # ========================================================

    for config_name, (low, high) in configs.items():

        for seed in SEEDS:

            print(
                "\n" + "=" * 60
            )

            print(
                f"  CONFIG {config_name} | "
                f"SEED {seed}"
            )

            print(
                "=" * 60 + "\n"
            )


            r, w, s = run_sensitivity(
                low_thresh=low,
                high_thresh=high,
                seed=seed
            )


            all_results[
                config_name
            ].append(
                (
                    r,
                    w,
                    s
                )
            )


    # ========================================================
    # Summary
    # ========================================================

    print(
        "\n\n"
        "===== TRUST THRESHOLD SENSITIVITY RESULTS "
        "(mean ± std) =====\n"
    )


    for config_name in configs:


        # ----------------------------------------------------
        # Average each seed first
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Mean ± std across seeds
        # ----------------------------------------------------

        r_m = np.mean(all_r)
        r_s = np.std(all_r)

        w_m = np.mean(all_w)
        w_s = np.std(all_w)

        s_m = np.mean(all_s)
        s_s = np.std(all_s)


        print(
            f"{config_name}"
        )

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


    # ========================================================
    # Save results
    # ========================================================

    np.save(
        "sensitivity_results.npy",
        all_results
    )

    print(
        "Saved: sensitivity_results.npy"
    )


    # ========================================================
    # Plot reward comparison
    # ========================================================

    import matplotlib.pyplot as plt


    colors = {

        "A: low=0.3, high=1.7":
            "#5B8DB8",

        "B: low=0.5, high=1.5":
            "#E05C4B",

        "C: low=0.7, high=1.3":
            "#4CAF50",
    }


    plt.figure(
        figsize=(10, 5)
    )


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
            color=colors[config_name],
            linewidth=2
        )


        plt.fill_between(
            steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
            color=colors[config_name]
        )


    plt.title(
        "Trust Threshold Sensitivity — "
        "Avg Reward (mean ± std, 5 seeds)"
    )

    plt.xlabel(
        "Simulation Steps"
    )

    plt.ylabel(
        "Avg Reward"
    )

    plt.legend(
        fontsize=8
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(
        "sensitivity_reward.png",
        dpi=150
    )

    plt.show()


    print(
        "Saved: sensitivity_reward.png"
    )
