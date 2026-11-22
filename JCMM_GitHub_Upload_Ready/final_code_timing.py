import numpy as np
import time
import random
import torch

from environment import SumoEnvironment
from agent import Agent, ACTIONS
from trust_module import TrustModule


print("RUNNING COMPUTATIONAL COST + CONVERGENCE EXPERIMENT")
print("MAX_STEPS = 300")
print("SEEDS = 10")
print("=" * 70)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_STEPS = 300

SEEDS = [
    42,
    123,
    256,
    789,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768
]


# ============================================================
# STATE DISCRETIZATION
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
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(mode, seed=42):

    set_seed(seed)

    print(
        f"\nStarting Mode {mode} | Seed {seed}"
    )

    # --------------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------------

    total_start = time.perf_counter()

    agent = Agent()

    trust = (
        TrustModule()
        if mode == 4
        else None
    )

    sumo = SumoEnvironment(
        "config.sumocfg"
    )

    epsilon = 0.1
    alpha = 0.05
    gamma = 0.9

    sumo.reset()

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    reward_history = []
    wait_history = []
    speed_history = []

    # --------------------------------------------------------
    # TIMING ACCUMULATORS
    # --------------------------------------------------------

    inference_time_total = 0.0

    training_time_total = 0.0

    trust_time_total = 0.0

    # Number of individual vehicle action selections
    inference_calls = 0

    # Number of trust calculations
    trust_calls = 0

    # Number of DQN updates
    training_calls = 0

    # --------------------------------------------------------
    # TRUST HISTORY
    # --------------------------------------------------------

    # Only Mode 4 has trust.
    #
    # For every simulation step we store the mean trust
    # across active vehicles.
    #
    # We also store the standard deviation.
    #
    trust_mean_history = []
    trust_std_history = []

    # --------------------------------------------------------
    # SIMULATION LOOP
    # --------------------------------------------------------

    for step in range(MAX_STEPS):

        sumo.step()

        vehicles = sumo.get_vehicles()

        # ----------------------------------------------------
        # COLLECT VEHICLE STATES
        # ----------------------------------------------------

        vehicle_states = {}

        for v in vehicles:

            _, speed, wait, _ = (
                sumo.get_state(v)
            )

            vehicle_states[v] = (
                speed,
                wait
            )

        # ----------------------------------------------------
        # GLOBAL TRAFFIC STATISTICS
        # ----------------------------------------------------

        if vehicles:

            avg_speed = np.mean(
                [
                    speed
                    for speed, wait
                    in vehicle_states.values()
                ]
            )

            avg_wait = np.mean(
                [
                    wait
                    for speed, wait
                    in vehicle_states.values()
                ]
            )

        else:

            avg_speed = 0.0
            avg_wait = 0.0

        # ----------------------------------------------------
        # PER-STEP MEMORY
        # ----------------------------------------------------

        states_memory = []

        rewards = []

        current_trust_values = []

        # ====================================================
        # VEHICLE LOOP
        # ====================================================

        for v in vehicles:

            speed, wait = vehicle_states[v]

            state = discretize(
                speed,
                wait
            )

            # ------------------------------------------------
            # MODE 1 — FIXED RULE
            # ------------------------------------------------

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

                reward = (
                    speed / 8
                    - wait / 15
                )

            # ------------------------------------------------
            # MODE 2 — PURE DQN
            # ------------------------------------------------

            elif mode == 2:

                agent.init(v)

                # --------------------------------------------
                # INFERENCE TIMING
                # --------------------------------------------

                inference_start = (
                    time.perf_counter()
                )

                action_idx = agent.choose(
                    v,
                    state,
                    epsilon,
                    trust=1.0
                )

                inference_time_total += (
                    time.perf_counter()
                    - inference_start
                )

                inference_calls += 1

                # --------------------------------------------

                action = ACTIONS[
                    action_idx
                ]

                sumo.apply_speed(
                    v,
                    action
                )

                reward = (
                    speed / 8
                    - wait / 15
                )

                states_memory.append(
                    (
                        v,
                        state,
                        action_idx,
                        reward
                    )
                )

            # ------------------------------------------------
            # MODE 3 — DQN + REWARD SHAPING
            # ------------------------------------------------

            elif mode == 3:

                agent.init(v)

                # --------------------------------------------
                # INFERENCE TIMING
                # --------------------------------------------

                inference_start = (
                    time.perf_counter()
                )

                action_idx = agent.choose(
                    v,
                    state,
                    epsilon,
                    trust=1.0
                )

                inference_time_total += (
                    time.perf_counter()
                    - inference_start
                )

                inference_calls += 1

                # --------------------------------------------

                action = ACTIONS[
                    action_idx
                ]

                sumo.apply_speed(
                    v,
                    action
                )

                individual_reward = (
                    speed / 8
                    - wait / 15
                )

                global_reward = (
                    avg_speed / 8
                    - avg_wait / 15
                )

                reward = (
                    0.7 * individual_reward
                    + 0.3 * global_reward
                )

                states_memory.append(
                    (
                        v,
                        state,
                        action_idx,
                        reward
                    )
                )

            # ------------------------------------------------
            # MODE 4 — DQN + REWARD SHAPING + TRUST
            # ------------------------------------------------

            else:

                # --------------------------------------------
                # TRUST COMPUTATION TIMING
                # --------------------------------------------

                trust_start = (
                    time.perf_counter()
                )

                trust.update(
                    v,
                    speed,
                    wait
                )

                t = trust.get(v)

                trust_time_total += (
                    time.perf_counter()
                    - trust_start
                )

                trust_calls += 1

                current_trust_values.append(
                    float(t)
                )

                # --------------------------------------------

                agent.init(v)

                # --------------------------------------------
                # INFERENCE TIMING
                # --------------------------------------------

                inference_start = (
                    time.perf_counter()
                )

                action_idx = agent.choose(
                    v,
                    state,
                    epsilon,
                    t
                )

                inference_time_total += (
                    time.perf_counter()
                    - inference_start
                )

                inference_calls += 1

                # --------------------------------------------

                action = ACTIONS[
                    action_idx
                ]

                sumo.apply_speed(
                    v,
                    action
                )

                individual_reward = (
                    speed / 8
                    - wait / 15
                )

                global_reward = (
                    avg_speed / 8
                    - avg_wait / 15
                )

                reward = (
                    0.7 * individual_reward
                    + 0.3 * global_reward
                )

                states_memory.append(
                    (
                        v,
                        state,
                        action_idx,
                        reward
                    )
                )

            rewards.append(
                reward
            )

        # ====================================================
        # RECORD TRUST EVOLUTION
        # ====================================================

        if mode == 4:

            if current_trust_values:

                trust_mean_history.append(
                    float(
                        np.mean(
                            current_trust_values
                        )
                    )
                )

                trust_std_history.append(
                    float(
                        np.std(
                            current_trust_values
                        )
                    )
                )

            else:

                trust_mean_history.append(
                    np.nan
                )

                trust_std_history.append(
                    np.nan
                )

        # ====================================================
        # DQN UPDATE
        # ====================================================

        if mode != 1:

            sumo.step()

            next_vehicles = (
                sumo.get_vehicles()
            )

            for (
                v,
                state,
                action_idx,
                reward
            ) in states_memory:

                if v in next_vehicles:

                    _, speed, wait, _ = (
                        sumo.get_state(v)
                    )

                    next_state = (
                        discretize(
                            speed,
                            wait
                        )
                    )

                    # ----------------------------------------
                    # TRAINING TIMING
                    # ----------------------------------------

                    training_start = (
                        time.perf_counter()
                    )

                    agent.update(
                        v,
                        state,
                        action_idx,
                        reward,
                        next_state,
                        alpha,
                        gamma
                    )

                    training_time_total += (
                        time.perf_counter()
                        - training_start
                    )

                    training_calls += 1

        # ====================================================
        # EPSILON DECAY
        # ====================================================

        epsilon = max(
            0.05,
            epsilon * 0.995
        )

        # ====================================================
        # METRICS
        # ====================================================

        mean_reward = (
            np.mean(rewards)
            if rewards
            else 0.0
        )

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
        # PROGRESS
        # ====================================================

        print(
            f"Mode {mode} | "
            f"Seed {seed} | "
            f"Step {step:3d} | "
            f"Reward {mean_reward:.3f} | "
            f"Speed {avg_speed:.2f} | "
            f"Wait {avg_wait:.2f}"
        )

    # ========================================================
    # CLOSE SUMO
    # ========================================================

    sumo.close()

    # ========================================================
    # TOTAL RUNTIME
    # ========================================================

    total_runtime = (
        time.perf_counter()
        - total_start
    )

    # ========================================================
    # AVERAGE TIMINGS
    # ========================================================

    if inference_calls > 0:

        mean_inference_time = (
            inference_time_total
            / inference_calls
        )

    else:

        mean_inference_time = 0.0

    if trust_calls > 0:

        mean_trust_time = (
            trust_time_total
            / trust_calls
        )

    else:

        mean_trust_time = 0.0

    if training_calls > 0:

        mean_training_time = (
            training_time_total
            / training_calls
        )

    else:

        mean_training_time = 0.0

    # ========================================================
    # PRINT TIMING RESULTS
    # ========================================================

    print(
        "\n"
        + "-" * 70
    )

    print(
        f"COMPUTATIONAL COST | "
        f"Mode {mode} | Seed {seed}"
    )

    print(
        "-" * 70
    )

    print(
        f"Total runtime        : "
        f"{total_runtime:.4f} s"
    )

    print(
        f"Inference calls      : "
        f"{inference_calls}"
    )

    print(
        f"Mean inference time  : "
        f"{mean_inference_time * 1000:.6f} ms"
    )

    print(
        f"Training calls       : "
        f"{training_calls}"
    )

    print(
        f"Mean training update : "
        f"{mean_training_time * 1000:.6f} ms"
    )

    if mode == 4:

        print(
            f"Trust calls          : "
            f"{trust_calls}"
        )

        print(
            f"Mean trust time      : "
            f"{mean_trust_time * 1000:.6f} ms"
        )

    print(
        "-" * 70
    )

    # ========================================================
    # RETURN ALL RESULTS
    # ========================================================

    result = {

        "reward_history":
            reward_history,

        "wait_history":
            wait_history,

        "speed_history":
            speed_history,

        "total_runtime":
            total_runtime,

        "inference_time_total":
            inference_time_total,

        "inference_calls":
            inference_calls,

        "mean_inference_time":
            mean_inference_time,

        "training_time_total":
            training_time_total,

        "training_calls":
            training_calls,

        "mean_training_time":
            mean_training_time,

        "trust_time_total":
            trust_time_total,

        "trust_calls":
            trust_calls,

        "mean_trust_time":
            mean_trust_time,

        "trust_mean_history":
            trust_mean_history,

        "trust_std_history":
            trust_std_history,

        "seed":
            seed,

        "mode":
            mode
    }

    return result


# ============================================================
# MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 70
    )

    print(
        "COMPUTATIONAL COST EXPERIMENT"
    )

    print(
        "=" * 70
    )

    print(
        "\nModes      : 1, 2, 3, 4"
    )

    print(
        "Seeds      : 10"
    )

    print(
        "Steps      : 300 per run"
    )

    print(
        "Measurement: runtime + inference + training + trust"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Store all results
    # --------------------------------------------------------

    computational_results = {
        1: [],
        2: [],
        3: [],
        4: []
    }

    # --------------------------------------------------------
    # Run all modes and seeds
    # --------------------------------------------------------

    experiment_start = (
        time.perf_counter()
    )

    for mode in [1, 2, 3, 4]:

        for seed in SEEDS:

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"MODE {mode} | SEED {seed}"
            )

            print(
                "=" * 70
            )

            result = run_experiment(
                mode,
                seed
            )

            computational_results[
                mode
            ].append(
                result
            )

    experiment_total_runtime = (
        time.perf_counter()
        - experiment_start
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "COMPUTATIONAL COST SUMMARY"
    )

    print(
        "=" * 70
    )

    for mode in [1, 2, 3, 4]:

        results = (
            computational_results[mode]
        )

        runtimes = [
            x["total_runtime"]
            for x in results
        ]

        inference_times = [
            x["mean_inference_time"]
            for x in results
        ]

        training_times = [
            x["mean_training_time"]
            for x in results
        ]

        print(
            f"\nMode {mode}"
        )

        print(
            f"  Runtime : "
            f"{np.mean(runtimes):.4f} "
            f"± {np.std(runtimes):.4f} s"
        )

        if mode != 1:

            print(
                f"  Inference : "
                f"{np.mean(inference_times) * 1000:.6f} "
                f"± "
                f"{np.std(inference_times) * 1000:.6f} ms"
            )

            print(
                f"  Training update : "
                f"{np.mean(training_times) * 1000:.6f} "
                f"± "
                f"{np.std(training_times) * 1000:.6f} ms"
            )

        if mode == 4:

            trust_times = [
                x["mean_trust_time"]
                for x in results
            ]

            print(
                f"  Trust computation : "
                f"{np.mean(trust_times) * 1000:.6f} "
                f"± "
                f"{np.std(trust_times) * 1000:.6f} ms"
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"TOTAL EXPERIMENT TIME: "
        f"{experiment_total_runtime:.2f} s"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    np.save(
        "computational_results.npy",
        computational_results,
        allow_pickle=True
    )

    print(
        "\nSaved:"
    )

    print(
        "  computational_results.npy"
    )

    # ========================================================
    # CONVERGENCE DATA
    # ========================================================

    convergence_data = {}

    for mode in [1, 2, 3, 4]:

        results = (
            computational_results[mode]
        )

        reward_histories = np.array(
            [
                x["reward_history"]
                for x in results
            ],
            dtype=np.float64
        )

        wait_histories = np.array(
            [
                x["wait_history"]
                for x in results
            ],
            dtype=np.float64
        )

        speed_histories = np.array(
            [
                x["speed_history"]
                for x in results
            ],
            dtype=np.float64
        )

        convergence_data[mode] = {

            "reward_mean":
                np.mean(
                    reward_histories,
                    axis=0
                ),

            "reward_std":
                np.std(
                    reward_histories,
                    axis=0
                ),

            "wait_mean":
                np.mean(
                    wait_histories,
                    axis=0
                ),

            "wait_std":
                np.std(
                    wait_histories,
                    axis=0
                ),

            "speed_mean":
                np.mean(
                    speed_histories,
                    axis=0
                ),

            "speed_std":
                np.std(
                    speed_histories,
                    axis=0
                )
        }

    # ========================================================
    # TRUST EVOLUTION DATA
    # ========================================================

    trust_histories = []

    for result in computational_results[4]:

        trust_histories.append(
            result["trust_mean_history"]
        )

    trust_histories = np.array(
        trust_histories,
        dtype=np.float64
    )

    trust_evolution = {

        "mean":
            np.nanmean(
                trust_histories,
                axis=0
            ),

        "std":
            np.nanstd(
                trust_histories,
                axis=0
            ),

        "individual":
            trust_histories
    }

    # ========================================================
    # SAVE ANALYSIS DATA
    # ========================================================

    np.save(
        "convergence_data.npy",
        convergence_data,
        allow_pickle=True
    )

    np.save(
        "trust_evolution.npy",
        trust_evolution,
        allow_pickle=True
    )

    print(
        "  convergence_data.npy"
    )

    print(
        "  trust_evolution.npy"
    )
    

    # ========================================================
# COMBINED FIGURE — CONVERGENCE + TRUST EVOLUTION
# ========================================================

    import matplotlib.pyplot as plt

    mode_labels = {
        1: "Mode 1 (Rule)",
        2: "Mode 2 (DQN)",
        3: "Mode 3 (Reward Shaping)",
        4: "Mode 4 (DQN + Trust)"
    }

    colors = {
        1: "#888888",
        2: "#5B8DB8",
        3: "#F5A623",
        4: "#E05C4B"
    }

    steps = np.arange(MAX_STEPS)

# Create one figure with two panels
    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 6)
    )

# ========================================================
# PANEL (a) — CONVERGENCE
# ========================================================

    ax = axes[0]

    for mode in [1, 2, 3, 4]:

        mean_curve = convergence_data[mode]["reward_mean"]
        std_curve = convergence_data[mode]["reward_std"]

        ax.plot(
            steps,
            mean_curve,
            label=mode_labels[mode],
            color=colors[mode],
            linewidth=2
        )

        ax.fill_between(
            steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=colors[mode],
            alpha=0.2
        )

        ax.set_xlabel("Simulation Steps", fontsize=13)
        ax.set_ylabel("Average Reward",fontsize=13)
        ax.set_title("(a) Reward Convergence Across 10 Seeds",fontsize=14)
        ax.tick_params(axis="both",labelsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)


# ========================================================
# PANEL (b) — TRUST EVOLUTION
# ========================================================

    ax = axes[1]

    trust_mean = trust_evolution["mean"]
    trust_std = trust_evolution["std"]

    ax.plot(
        steps,
        trust_mean,
        color=colors[4],
        linewidth=2,
        label="Mode 4 Trust"
    )

    ax.fill_between(
        steps,
        trust_mean - trust_std,
        trust_mean + trust_std,
        color=colors[4],
        alpha=0.2
    )

    ax.set_xlabel("Simulation Steps")
    ax.set_ylabel("Mean Trust Value")
    ax.set_title("(b) Trust Evolution Across 10 Seeds")
    ax.tick_params(axis="both",labelsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)


# ========================================================
# SAVE COMBINED FIGURE
# ========================================================

    plt.tight_layout()

    plt.savefig(
       "convergence_and_trust.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()

    print("Saved: convergence_and_trust.png")
    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "COMPUTATIONAL COST EXPERIMENT COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nGenerated files:"
    )

    print(
        "  1. computational_results.npy"
    )

    print(
        "  2. convergence_data.npy"
    )

    print(
        "  3. trust_evolution.npy"
    )

    print(
        "  4. convergence_reward.png"
    )

    print(
        "  5. trust_evolution.png"
    )

    print(
        "\nUse these results for Reviewer Comment:"
    )

    print(
        "Computational cost, convergence, and trust evolution."
    )

    print(
        "=" * 70
    )