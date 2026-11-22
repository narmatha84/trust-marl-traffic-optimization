"""
Experiment 4: Asymmetric Traffic Load

Purpose:
    Test the robustness of the proposed trust-augmented DQN approach
    under asymmetric traffic demand.

Modes:
    Mode 1 - Fixed Rule
    Mode 2 - Pure DQN
    Mode 3 - DQN + Global Reward
    Mode 4 - DQN + Global Reward + Trust (Proposed)

Traffic demand:
    Symmetric experiment total demand = 800 veh/hr

    Asymmetric experiment:
        East-West:
            100 veh/hr
            100 veh/hr

        North-South:
            300 veh/hr
            300 veh/hr

        Total = 800 veh/hr

Seeds:
    10 seeds per mode

This protocol is designed to address Reviewer Comment 8:
    - Add Mode 2
    - Match total demand with symmetric experiment
    - Compare all four modes under the same protocol
"""

import numpy as np
import torch
import time
import random
import matplotlib.pyplot as plt

from environment import SumoEnvironment
from agent import Agent, ACTIONS
from trust_module import TrustModule


# ============================================================
# CONFIGURATION
# ============================================================
SEEDS = [42, 123, 256, 789, 1024,
         2048, 4096, 8192, 16384, 32768]

MAX_STEPS = 300

ALPHA = 0.05
GAMMA = 0.9

INITIAL_EPSILON = 0.1
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995


# ============================================================
# STATE DISCRETIZATION
# ============================================================

def discretize(speed, wait):
    """
    Convert continuous speed/wait values into a discrete state.

    States:
        0 -> Very high waiting time
        1 -> Low speed
        2 -> Moderate speed
        3 -> High speed
    """

    if wait > 20:
        return 0
    elif speed < 1:
        return 1
    elif speed < 5:
        return 2
    else:
        return 3


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):
    """
    Set all relevant random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# WRITE ASYMMETRIC ROUTES
# ============================================================

def write_asymmetric_routes():

    """
    Creates the asymmetric SUMO route file.

    Total demand is intentionally kept at 800 veh/hr,
    matching the balanced/symmetric experiment.

    East-West:
        100 + 100 = 200 veh/hr

    North-South:
        300 + 300 = 600 veh/hr

    Total:
        800 veh/hr
    """

    xml = """<routes>

    <vType id="car"
           accel="1.0"
           decel="4.5"
           sigma="0.6"
           length="5"
           minGap="2.5"
           tau="1.2"
           maxSpeed="8"
           lcStrategic="0.3"
           lcCooperative="0.2"
           lcSpeedGain="0.5"
           lcKeepRight="0.5"/>

    <!-- ROUTES -->

    <route id="r_lr" edges="e0 e1"/>
    <route id="r_rl" edges="e1_rev e0_rev"/>
    <route id="r_tb" edges="e3_rev e2_rev"/>
    <route id="r_bt" edges="e2 e3"/>


    <!--
        ASYMMETRIC TRAFFIC

        East-West:
            LR = 100 veh/hr
            RL = 100 veh/hr

        North-South:
            TB = 300 veh/hr
            BT = 300 veh/hr

        TOTAL = 800 veh/hr
    -->

    <flow id="f_lr"
          route="r_lr"
          type="car"
          begin="0"
          end="1000"
          vehsPerHour="100"/>

    <flow id="f_rl"
          route="r_rl"
          type="car"
          begin="0"
          end="1000"
          vehsPerHour="100"/>

    <flow id="f_tb"
          route="r_tb"
          type="car"
          begin="0"
          end="1000"
          vehsPerHour="300"/>

    <flow id="f_bt"
          route="r_bt"
          type="car"
          begin="0"
          end="1000"
          vehsPerHour="300"/>

</routes>
"""

    with open("routes_asymmetric.rou.xml", "w") as f:
        f.write(xml)

    print("Written: routes_asymmetric.rou.xml")


# ============================================================
# WRITE ASYMMETRIC SUMO CONFIG
# ============================================================

def write_asymmetric_config():

    """
    Creates a SUMO configuration file pointing to
    the asymmetric route file.

    IMPORTANT:
    The network file name is kept as network.net.xml.
    Change it only if your existing network uses another name.
    """

    xml = """<configuration>

    <input>
        <net-file value="map.net.xml"/>
        <route-files value="routes_asymmetric.rou.xml"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="1000"/>
    </time>

</configuration>
"""

    with open("config_asymmetric.sumocfg", "w") as f:
        f.write(xml)

    print("Written: config_asymmetric.sumocfg")


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(mode, seed):

    """
    Run one asymmetric traffic experiment.

    Modes:

        Mode 1:
            Fixed Rule

        Mode 2:
            Pure DQN
            Individual reward only

        Mode 3:
            DQN + Global Reward
            No trust mechanism

        Mode 4:
            DQN + Global Reward + Trust
            Proposed method
    """

    set_seed(seed)

    print(
        f"\nStarting Mode {mode} | Seed {seed}"
    )

    # --------------------------------------------------------
    # Initialize components
    # --------------------------------------------------------

    agent = Agent()

    # Trust is used ONLY in Mode 4
    trust = TrustModule() if mode == 4 else None

    sumo = SumoEnvironment(
        "config_asymmetric.sumocfg"
    )

    epsilon = INITIAL_EPSILON

    sumo.reset()

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    reward_history = []
    wait_history = []
    speed_history = []

    # ========================================================
    # SIMULATION LOOP
    # ========================================================

    for step in range(MAX_STEPS):

        # ----------------------------------------------------
        # Advance simulation
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
        # Global traffic statistics
        # ----------------------------------------------------

        if vehicles:

            avg_speed = np.mean(
                [
                    speed
                    for speed, wait in vehicle_states.values()
                ]
            )

            avg_wait = np.mean(
                [
                    wait
                    for speed, wait in vehicle_states.values()
                ]
            )

        else:

            avg_speed = 0.0
            avg_wait = 0.0

        # ----------------------------------------------------
        # Storage for DQN transitions
        # ----------------------------------------------------

        transitions = []

        step_rewards = []

        # ====================================================
        # VEHICLE DECISION LOOP
        # ====================================================

        for v in vehicles:

            speed, wait = vehicle_states[v]

            state = discretize(
                speed,
                wait
            )

            # =================================================
            # MODE 1 — FIXED RULE
            # =================================================

            if mode == 1:

                if wait > 20:
                    action = 8

                elif wait > 10:
                    action = 6

                elif wait > 5:
                    action = 4

                else:
                    action = 2

                sumo.apply_speed(
                    v,
                    action
                )

                # Individual reward
                reward = (
                    speed / 8.0
                    - wait / 15.0
                )

                step_rewards.append(
                    reward
                )

            # =================================================
            # MODE 2 — PURE DQN
            # =================================================

            elif mode == 2:

                agent.init(v)

                action_idx = agent.choose(
                    v,
                    state,
                    epsilon,
                    1.0
                )

                action = ACTIONS[action_idx]

                sumo.apply_speed(
                    v,
                    action
                )

                # Pure individual reward
                reward = (
                    speed / 8.0
                    - wait / 15.0
                )

                transitions.append(
                    (
                        v,
                        state,
                        action_idx,
                        reward
                    )
                )

                step_rewards.append(
                    reward
                )

            # =================================================
            # MODE 3 — DQN + GLOBAL REWARD
            # =================================================

            elif mode == 3:

                agent.init(v)

                # No trust gating.
                # trust = 1.0 means normal epsilon-greedy.
                action_idx = agent.choose(
                    v,
                    state,
                    epsilon,
                    1.0
                )

                action = ACTIONS[action_idx]

                sumo.apply_speed(
                    v,
                    action
                )

                # Individual component
                individual_reward = (
                    speed / 8.0
                    - wait / 15.0
                )

                # Global component
                global_reward = (
                    avg_speed / 8.0
                    - avg_wait / 15.0
                )

                # Same reward formulation used
                # in the main experiment
                reward = (
                    0.7 * individual_reward
                    + 0.3 * global_reward
                )

                transitions.append(
                    (
                        v,
                        state,
                        action_idx,
                        reward
                    )
                )

                step_rewards.append(
                    reward
                )

            # =================================================
            # MODE 4 — DQN + GLOBAL REWARD + TRUST
            # =================================================

            elif mode == 4:

                # Update trust using current behavior
                trust.update(
                    v,
                    speed,
                    wait
                )

                current_trust = trust.get(v)

                agent.init(v)

                # Trust-gated DQN action selection
                action_idx = agent.choose(
                    v,
                    state,
                    epsilon,
                    current_trust
                )

                action = ACTIONS[action_idx]

                sumo.apply_speed(
                    v,
                    action
                )

                # Individual component
                individual_reward = (
                    speed / 8.0
                    - wait / 15.0
                )

                # Global component
                global_reward = (
                    avg_speed / 8.0
                    - avg_wait / 15.0
                )

                # Combined reward
                reward = (
                    0.7 * individual_reward
                    + 0.3 * global_reward
                )

                transitions.append(
                    (
                        v,
                        state,
                        action_idx,
                        reward
                    )
                )

                step_rewards.append(
                    reward
                )

        # ====================================================
        # ADVANCE ONE STEP FOR NEXT STATE
        # ====================================================

        if mode in [2, 3, 4] and transitions:

            sumo.step()

            next_vehicles = sumo.get_vehicles()

            for (
                v,
                state,
                action_idx,
                reward
            ) in transitions:

                if v in next_vehicles:

                    _, next_speed, next_wait, _ = (
                        sumo.get_state(v)
                    )

                    next_state = discretize(
                        next_speed,
                        next_wait
                    )

                    agent.update(
                        v,
                        state,
                        action_idx,
                        reward,
                        next_state,
                        alpha=ALPHA,
                        gamma=GAMMA
                    )

        # ====================================================
        # EPSILON DECAY
        # ====================================================

        epsilon = max(
            MIN_EPSILON,
            epsilon * EPSILON_DECAY
        )

        # ====================================================
        # STORE METRICS
        # ====================================================

        if step_rewards:

            mean_reward = np.mean(
                step_rewards
            )

        else:

            mean_reward = 0.0

        reward_history.append(
            mean_reward
        )

        wait_history.append(
            avg_wait
        )

        speed_history.append(
            avg_speed
        )

        # ====================================================
        # PROGRESS OUTPUT
        # ====================================================

        print(
            f"[ASYMMETRIC] "
            f"Mode {mode} | "
            f"Seed {seed} | "
            f"Step {step:3d} | "
            f"Reward {mean_reward:.3f} | "
            f"Speed {avg_speed:.2f} | "
            f"Wait {avg_wait:.2f}"
        )

        # Small delay so terminal output remains readable
        time.sleep(0.02)

    # ========================================================
    # CLOSE SUMO
    # ========================================================

    sumo.close()

    return (
        reward_history,
        wait_history,
        speed_history
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Write configuration files
    # --------------------------------------------------------

    write_asymmetric_routes()
    write_asymmetric_config()

    # --------------------------------------------------------
    # Experiment storage
    # --------------------------------------------------------

    all_results = {
        1: [],
        2: [],
        3: [],
        4: []
    }

    # --------------------------------------------------------
    # Run all four modes with five seeds each
    # --------------------------------------------------------

    for mode in [1, 2, 3, 4]:

        print("\n")
        print("=" * 70)

        if mode == 1:
            print(
                "MODE 1 — FIXED RULE"
            )

        elif mode == 2:
            print(
                "MODE 2 — PURE DQN"
            )

        elif mode == 3:
            print(
                "MODE 3 — DQN + GLOBAL REWARD"
            )

        elif mode == 4:
            print(
                "MODE 4 — DQN + GLOBAL REWARD + TRUST"
            )

        print(
            f"Running {len(SEEDS)} seeds: {SEEDS}"
        )

        print("=" * 70)

        for seed in SEEDS:

            print("\n" + "-" * 60)
            print(
                f"Mode {mode} | Seed {seed}"
            )
            print("-" * 60)

            result = run_experiment(
                mode=mode,
                seed=seed
            )

            all_results[mode].append(
                result
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print(
        "ASYMMETRIC TRAFFIC RESULTS "
        "(mean ± std across 10 seeds)"
    )
    print("=" * 70)

    labels = {
        1: "Mode 1 — Fixed Rule",
        2: "Mode 2 — Pure DQN",
        3: "Mode 3 — DQN + Global Reward",
        4: "Mode 4 — DQN + Global Reward + Trust"
    }

    summary_results = {}

    for mode in [1, 2, 3, 4]:

        rewards = [
            np.mean(r)
            for r, w, s in all_results[mode]
        ]

        waits = [
            np.mean(w)
            for r, w, s in all_results[mode]
        ]

        speeds = [
            np.mean(s)
            for r, w, s in all_results[mode]
        ]

        reward_mean = np.mean(rewards)
        reward_std = np.std(
            rewards,
            ddof=1
        )

        wait_mean = np.mean(waits)
        wait_std = np.std(
            waits,
            ddof=1
        )

        speed_mean = np.mean(speeds)
        speed_std = np.std(
            speeds,
            ddof=1
        )

        summary_results[mode] = {
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "wait_mean": wait_mean,
            "wait_std": wait_std,
            "speed_mean": speed_mean,
            "speed_std": speed_std
        }

        print(
            f"\n{labels[mode]}"
        )

        print(
            f"  Avg Reward : "
            f"{reward_mean:.3f} ± "
            f"{reward_std:.3f}"
        )

        print(
            f"  Avg Wait   : "
            f"{wait_mean:.3f} ± "
            f"{wait_std:.3f} s"
        )

        print(
            f"  Avg Speed  : "
            f"{speed_mean:.3f} ± "
            f"{speed_std:.3f} m/s"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    np.save(
        "asymmetric_results.npy",
        all_results,
        allow_pickle=True
    )

    print(
        "\nSaved: asymmetric_results.npy"
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    np.save(
        "asymmetric_summary.npy",
        summary_results,
        allow_pickle=True
    )

    print(
        "Saved: asymmetric_summary.npy"
    )

    # ========================================================
    # PLOTS
    # ========================================================

    mode_labels = {
        1: "Mode 1 (Rule)",
        2: "Mode 2 (DQN)",
        3: "Mode 3 (DQN + Global)",
        4: "Mode 4 (DQN + Global + Trust)"
    }

    # --------------------------------------------------------
    # Reward plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    for mode in [1, 2, 3, 4]:

        histories = [
            r
            for r, w, s in all_results[mode]
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
            label=mode_labels[mode],
            linewidth=2
        )

        plt.fill_between(
            steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.15
        )

    plt.title(
        "Asymmetric Traffic — Average Reward\n"
        "Total Demand = 800 veh/hr"
    )

    plt.xlabel(
        "Simulation Steps"
    )

    plt.ylabel(
        "Average Reward"
    )

    plt.legend()
    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "asymmetric_reward.png",
        dpi=150
    )

    plt.show()

    print(
        "Saved: asymmetric_reward.png"
    )

    # --------------------------------------------------------
    # Waiting time plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    for mode in [1, 2, 3, 4]:

        histories = [
            w
            for r, w, s in all_results[mode]
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
            label=mode_labels[mode],
            linewidth=2
        )

        plt.fill_between(
            steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.15
        )

    plt.title(
        "Asymmetric Traffic — Average Waiting Time\n"
        "Total Demand = 800 veh/hr"
    )

    plt.xlabel(
        "Simulation Steps"
    )

    plt.ylabel(
        "Average Wait (s)"
    )

    plt.legend()
    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "asymmetric_wait.png",
        dpi=150
    )

    plt.show()

    print(
        "Saved: asymmetric_wait.png"
    )

    # --------------------------------------------------------
    # Speed plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    for mode in [1, 2, 3, 4]:

        histories = [
            s
            for r, w, s in all_results[mode]
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
            label=mode_labels[mode],
            linewidth=2
        )

        plt.fill_between(
            steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.15
        )

    plt.title(
        "Asymmetric Traffic — Average Speed\n"
        "Total Demand = 800 veh/hr"
    )

    plt.xlabel(
        "Simulation Steps"
    )

    plt.ylabel(
        "Average Speed (m/s)"
    )

    plt.legend()
    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "asymmetric_speed.png",
        dpi=150
    )

    plt.show()

    print(
        "Saved: asymmetric_speed.png"
    )

    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ASYMMETRIC EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        "\nTraffic configuration:"
    )

    print(
        "  East-West  : 100 + 100 = 200 veh/hr"
    )

    print(
        "  North-South: 300 + 300 = 600 veh/hr"
    )

    print(
        "  TOTAL      : 800 veh/hr"
    )

    print(
        "\nModes:"
    )

    print(
        "  Mode 1 = Fixed Rule"
    )

    print(
        "  Mode 2 = Pure DQN"
    )

    print(
        "  Mode 3 = DQN + Global Reward"
    )

    print(
        "  Mode 4 = DQN + Global Reward + Trust"
    )

    print(
        "\nSeeds:"
    )

    print(
        f"  {SEEDS}"
    )

    print(
        "\nOutput files:"
    )

    print(
        "  asymmetric_results.npy"
    )

    print(
        "  asymmetric_summary.npy"
    )

    print(
        "  asymmetric_reward.png"
    )

    print(
        "  asymmetric_wait.png"
    )

    print(
        "  asymmetric_speed.png"
    )

    print("=" * 70)