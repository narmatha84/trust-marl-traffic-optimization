import numpy as np
import time
import random
import torch

from environment import SumoEnvironment
from agent import Agent, ACTIONS
from trust_module import TrustModule

print("RUNNING UPDATED COMPARISON FILE — 4 MODES")
print("MAX_STEPS = 300")


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


def run_experiment(mode, seed=42):
    set_seed(seed)

    agent = Agent()
    trust = TrustModule() if mode == 4 else None
    sumo  = SumoEnvironment("config.sumocfg")

    epsilon = 0.1
    alpha   = 0.05
    gamma   = 0.9

    sumo.reset()

    MAX_STEPS = 300

    reward_history = []
    wait_history   = []
    speed_history  = []

    for step in range(MAX_STEPS):

        sumo.step()
        vehicles = sumo.get_vehicles()

        vehicle_states = {}
        for v in vehicles:
            _, speed, wait, _ = sumo.get_state(v)
            vehicle_states[v] = (speed, wait)

        avg_speed = np.mean([s for s, w in vehicle_states.values()]) if vehicles else 0
        avg_wait  = np.mean([w for s, w in vehicle_states.values()]) if vehicles else 0

        states_memory = []
        rewards       = []

        for v in vehicles:
            speed, wait = vehicle_states[v]
            s = discretize(speed, wait)

            # ── MODE 1: Fixed Rule ──────────────────────────────────────────
            if mode == 1:
                if   wait > 20: action = 8
                elif wait > 10: action = 6
                elif wait > 5:  action = 4
                else:           action = 2

                sumo.apply_speed(v, action)
                reward = speed / 8 - wait / 15

            # ── MODE 2: Pure DQN ────────────────────────────────────────────
            elif mode == 2:
                agent.init(v)
                action_idx = agent.choose(v, s, epsilon, trust=1.0)
                action = ACTIONS[action_idx]

                sumo.apply_speed(v, action)
                reward = speed / 8 - wait / 15
                states_memory.append((v, s, action_idx, reward))

            # ── MODE 3: DQN + Reward Shaping only (no trust gating) ─────────
            # Ablation control: isolates reward shaping contribution
            # Uses combined individual + global reward but NO trust mechanism
            elif mode == 3:
                agent.init(v)
                # Standard epsilon-greedy, trust fixed at 1.0 (no gating)
                action_idx = agent.choose(v, s, epsilon, trust=1.0)
                action = ACTIONS[action_idx]

                sumo.apply_speed(v, action)

                individual    = speed / 8 - wait / 15
                global_reward = avg_speed / 8 - avg_wait / 15
                reward        = 0.7 * individual + 0.3 * global_reward
                states_memory.append((v, s, action_idx, reward))

            # ── MODE 4: DQN + Reward Shaping + Trust (Proposed) ─────────────
            # Full proposed framework: trust gates exploration-exploitation
            else:
                trust.update(v, speed, wait)
                t = trust.get(v)

                agent.init(v)
                action_idx = agent.choose(v, s, epsilon, t)
                action = ACTIONS[action_idx]

                sumo.apply_speed(v, action)

                individual    = speed / 8 - wait / 15
                global_reward = avg_speed / 8 - avg_wait / 15
                reward        = 0.7 * individual + 0.3 * global_reward
                states_memory.append((v, s, action_idx, reward))

            rewards.append(reward)

        # ── DQN update (modes 2, 3, 4 only) ─────────────────────────────────
        if mode != 1:
            sumo.step()
            next_vehicles = sumo.get_vehicles()

            for v, s, a, r in states_memory:
                if v in next_vehicles:
                    _, speed, wait, _ = sumo.get_state(v)
                    ns = discretize(speed, wait)
                    agent.update(v, s, a, r, ns, alpha, gamma)

        epsilon = max(0.05, epsilon * 0.995)

        reward_history.append(np.mean(rewards) if rewards else 0)
        wait_history.append(avg_wait)
        speed_history.append(avg_speed)

        label = f"Seed {seed} | " if mode != 1 else ""
        avg_reward = np.mean(rewards) if rewards else 0

        print(f"{label}Mode {mode} | Step {step:3d} | "
              f"Reward {avg_reward:.3f} | "
              f"Speed {avg_speed:.2f} | Wait {avg_wait:.2f}")

        time.sleep(0.02)

    sumo.close()
    return reward_history, wait_history, speed_history


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    SEEDS = [42, 123, 256, 789, 1024,
         2048, 4096, 8192, 16384, 32768]
    all_results = {1: [], 2: [], 3: [], 4: []}

 
    for mode in [1,2, 3, 4]:
        for seed in SEEDS:
            print(f"\n{'='*50}")
            print(f"  MODE {mode}  |  SEED {seed}")
            print(f"{'='*50}\n")

            r, w, s = run_experiment(mode, seed)
            all_results[mode].append((r, w, s))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n================ RESULTS (mean ± std) ================\n")

    labels = {
        1: "Mode 1 — Fixed Rule                    ",
        2: "Mode 2 — Pure DQN                      ",
        3: "Mode 3 — DQN + Reward Shaping (ablation)",
        4: "Mode 4 — DQN + Trust Proposed (ours)   ",
    }

    summary = {}
    for mode in [1, 2, 3, 4]:
        all_r = [np.mean(r) for r, w, s in all_results[mode]]
        all_w = [np.mean(w) for r, w, s in all_results[mode]]
        all_s = [np.mean(s) for r, w, s in all_results[mode]]

        summary[mode] = {
            "reward": (np.mean(all_r), np.std(all_r)),
            "wait":   (np.mean(all_w), np.std(all_w)),
            "speed":  (np.mean(all_s), np.std(all_s)),
        }

        r_m, r_s = summary[mode]["reward"]
        w_m, w_s = summary[mode]["wait"]
        s_m, s_s = summary[mode]["speed"]

        print(f"{labels[mode]}")
        print(f"  Avg Reward : {r_m:.3f} ± {r_s:.3f}")
        print(f"  Avg Wait   : {w_m:.3f} ± {w_s:.3f} s")
        print(f"  Avg Speed  : {s_m:.3f} ± {s_s:.3f} m/s")
        print()

    # ── Wilcoxon significance tests (Mode 4 vs others) ───────────────────────
    from scipy import stats

    print("\n================ SIGNIFICANCE TESTS ================\n")
    print("Wilcoxon signed-rank test: Mode 4 (Proposed) vs others\n")

    mode4_r = [np.mean(r) for r, w, s in all_results[4]]
    mode4_w = [np.mean(w) for r, w, s in all_results[4]]

    for compare_mode in [2, 3]:
        compare_r = [np.mean(r) for r, w, s in all_results[compare_mode]]
        compare_w = [np.mean(w) for r, w, s in all_results[compare_mode]]

        stat_r, p_r = stats.wilcoxon(mode4_r, compare_r)
        stat_w, p_w = stats.wilcoxon(mode4_w, compare_w)

        print(f"Mode 4 vs Mode {compare_mode}:")
        print(f"  Reward  — W={stat_r:.3f}, p={p_r:.4f}")
        print(f"  Wait    — W={stat_w:.3f}, p={p_w:.4f}")
        print()

    # ── Save results ──────────────────────────────────────────────────────────
    np.save("all_results_4mode.npy", all_results)
    print("Saved: all_results_4mode.npy")

    # ── Plots ─────────────────────────────────────────────────────────────────# ── Plots ─────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

mode_labels = {
    1: "Mode 1 (Rule)",
    2: "Mode 2 (DQN)",
    3: "Mode 3 (Reward Shaping)",
    4: "Mode 4 (DQN+Trust)"
}

colors = {
    1: "#888888",
    2: "#5B8DB8",
    3: "#F5A623",
    4: "#E05C4B"
}

metrics = [
    ("reward", "Avg Reward",      "reward_comparison.png"),
    ("wait",   "Avg Wait (s)",    "wait_comparison.png"),
    ("speed",  "Avg Speed (m/s)", "speed_comparison.png"),
]

for metric_key, ylabel, filename in metrics:

    plt.figure(figsize=(10, 6))

    for mode in [1, 2, 3, 4]:

        if metric_key == "reward":
            histories = [r for r, w, s in all_results[mode]]

        elif metric_key == "wait":
            histories = [w for r, w, s in all_results[mode]]

        else:
            histories = [s for r, w, s in all_results[mode]]

        mean_curve = np.mean(histories, axis=0)
        std_curve  = np.std(histories, axis=0)
        steps      = np.arange(len(mean_curve))

        plt.plot(
            steps,
            mean_curve,
            label=mode_labels[mode],
            color=colors[mode],
            linewidth=2.2
        )

        plt.fill_between(
            steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
            color=colors[mode]
        )

    plt.title(
        f"{ylabel} — Mode Comparison (mean ± std, 10 seeds)",
        fontsize=15
    )

    plt.xlabel(
        "Simulation Steps",
        fontsize=14
    )

    plt.ylabel(
        ylabel,
        fontsize=14
    )

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(
        fontsize=12
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"Saved {filename}")