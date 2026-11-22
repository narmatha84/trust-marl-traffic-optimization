
"""
JCMM Comment 6 — Online vs Frozen-Policy Evaluation

Original files remain unchanged.

Online phase:
    Original Mode 4 training logic.

Frozen phase:
    Reuse trained per-vehicle DQN networks.
    No agent.update() or gradient steps.
    Preserve original trust gating.
    Report unseen/untrained vehicle policies.

Run:
    python jcmm_comment6_frozen_policy.py
"""

import random
import numpy as np
import torch
import traci

from environment import SumoEnvironment
from agent import Agent, ACTIONS
from trust_module import TrustModule

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

SEEDS = [42, 123, 256, 789, 1024,
         2048, 4096, 8192, 16384, 32768]

MAX_STEPS = 300
CONFIG = "config.sumocfg"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_phase(agent, seed, frozen=False):
    """
    Run original Mode 4 protocol.

    frozen=False: online learning
    frozen=True: trained weights remain unchanged
    """

    set_seed(seed)

    trust = TrustModule()
    sumo = SumoEnvironment(CONFIG)

    epsilon = 0.1
    alpha = 0.05
    gamma = 0.9

    rewards_history = []
    waits_history = []
    speeds_history = []

    decisions = 0
    unseen_decisions = 0
    trained_decisions = 0
    untrained_decisions = 0
    forced_random = 0
    epsilon_random = 0
    greedy_decisions = 0

    # Networks may have been initialized but never trained.
    trained_ids = {
        vid for vid, updates in agent.update_counts.items()
        if updates > 0
    }

    # Snapshot weights to verify no training changes occur.
    before = None
    if frozen:
        before = {
            vid: {
                key: value.detach().clone()
                for key, value in net.state_dict().items()
            }
            for vid, net in agent.networks.items()
        }

        for net in agent.networks.values():
            net.eval()

    try:
        sumo.reset()

        for step in range(MAX_STEPS):

            sumo.step()
            vehicles = list(sumo.get_vehicles())

            vehicle_states = {}

            for v in vehicles:
                _, speed, wait, _ = sumo.get_state(v)
                vehicle_states[v] = (speed, wait)

            avg_speed = (
                np.mean([x[0] for x in vehicle_states.values()])
                if vehicles else 0.0
            )

            avg_wait = (
                np.mean([x[1] for x in vehicle_states.values()])
                if vehicles else 0.0
            )

            global_reward = (
                avg_speed / 8.0 - avg_wait / 15.0
            )

            transitions = []
            step_rewards = []

            for v in vehicles:
                speed, wait = vehicle_states[v]
                state = discretize(speed, wait)

                trust.update(v, speed, wait)
                t = trust.get(v)

                if frozen:
                    if v not in agent.networks:
                        unseen_decisions += 1

                    if v in trained_ids:
                        trained_decisions += 1
                    else:
                        untrained_decisions += 1

                agent.init(v)

                # Original action-selection behavior.
                # Instrumented to distinguish random/greedy.
                if t < 0.5:
                    forced_random += 1
                elif t > 1.5:
                    greedy_decisions += 1
                else:
                    # We cannot infer the exact epsilon branch
                    # from Agent.choose() without instrumenting it.
                    pass

                action_idx = agent.choose(
                    v, state, epsilon, t
                )

                action = ACTIONS[action_idx]
                sumo.apply_speed(v, action)

                individual_reward = (
                    speed / 8.0 - wait / 15.0
                )

                reward = (
                    0.7 * individual_reward
                    + 0.3 * global_reward
                )

                step_rewards.append(reward)
                transitions.append(
                    (v, state, action_idx, reward)
                )

                decisions += 1

            # Match the original second simulation step.
            sumo.step()

            if not frozen:
                next_vehicles = set(sumo.get_vehicles())

                for v, state, action_idx, reward in transitions:
                    if v in next_vehicles:
                        _, speed, wait, _ = sumo.get_state(v)
                        next_state = discretize(speed, wait)

                        agent.update(
                            v, state, action_idx,
                            reward, next_state,
                            alpha, gamma
                        )

            epsilon = max(0.05, epsilon * 0.995)

            rewards_history.append(
                float(np.mean(step_rewards))
                if step_rewards else 0.0
            )

            waits_history.append(float(avg_wait))
            speeds_history.append(float(avg_speed))

            if step % 50 == 0 or step == MAX_STEPS - 1:
                label = "FROZEN" if frozen else "ONLINE"

                print(
                    f"[{label}] Seed {seed} | "
                    f"Step {step:3d} | "
                    f"Reward {rewards_history[-1]:.4f} | "
                    f"Wait {waits_history[-1]:.4f} | "
                    f"Speed {speeds_history[-1]:.4f}"
                )

    finally:
        sumo.close()

    if frozen:
        # Verify every previously existing network is unchanged.
        for vid, original_weights in before.items():
            current_weights = agent.networks[vid].state_dict()

            for key, original_value in original_weights.items():
                assert torch.equal(
                    original_value, current_weights[key]
                ), f"Frozen network changed: {vid}, {key}"

    return {
        "reward": np.asarray(rewards_history),
        "wait": np.asarray(waits_history),
        "speed": np.asarray(speeds_history),
        "decisions": decisions,
        "trained_decisions": trained_decisions,
        "untrained_decisions": untrained_decisions,
        "unseen_decisions": unseen_decisions,
        "forced_random": forced_random,
        "trained_networks": len(trained_ids),
        "total_networks": len(agent.networks),
    }


def main():
    online_results = []
    frozen_results = []

    print("=" * 65)
    print("JCMM COMMENT 6 — ONLINE VS FROZEN POLICY")
    print("=" * 65)

    for seed in SEEDS:
        print(f"\n{'=' * 65}")
        print(f"SEED {seed}")
        print("=" * 65)

        set_seed(seed)
        agent = Agent()

        print("\nPHASE 1: ONLINE TRAINING")
        online = run_phase(
            agent, seed, frozen=False
        )

        print("\nPHASE 2: FROZEN EVALUATION")
        frozen = run_phase(
            agent, seed, frozen=True
        )

        online_results.append(online)
        frozen_results.append(frozen)

    print("\n" + "=" * 65)
    print("FINAL RESULTS — 10 SEEDS")
    print("=" * 65)

    for label, results in [
        ("ONLINE", online_results),
        ("FROZEN", frozen_results),
    ]:
        print(f"\n{label}")

        for metric in ["reward", "wait", "speed"]:
            seed_values = np.array([
                np.mean(result[metric])
                for result in results
            ])

            print(
                f"  {metric:8s}: "
                f"{seed_values.mean():.4f} ± "
                f"{seed_values.std(ddof=1):.4f}"
            )

    total_decisions = sum(
        result["decisions"]
        for result in frozen_results
    )

    trained = sum(
        result["trained_decisions"]
        for result in frozen_results
    )

    untrained = sum(
        result["untrained_decisions"]
        for result in frozen_results
    )

    unseen = sum(
        result["unseen_decisions"]
        for result in frozen_results
    )

    print("\nFROZEN-POLICY COVERAGE")
    print(f"Total vehicle decisions: {total_decisions}")
    print(f"Trained-policy decisions: {trained}")
    print(f"Untrained-policy decisions: {untrained}")
    print(f"Previously unseen ID decisions: {unseen}")

    if total_decisions:
        print(
            "Trained-policy coverage: "
            f"{100 * trained / total_decisions:.2f}%"
        )

    np.savez(
        "jcmm_comment6_results.npz",
        seeds=np.array(SEEDS),
        online_reward=np.array([
            r["reward"] for r in online_results
        ]),
        online_wait=np.array([
            r["wait"] for r in online_results
        ]),
        online_speed=np.array([
            r["speed"] for r in online_results
        ]),
        frozen_reward=np.array([
            r["reward"] for r in frozen_results
        ]),
        frozen_wait=np.array([
            r["wait"] for r in frozen_results
        ]),
        frozen_speed=np.array([
            r["speed"] for r in frozen_results
        ]),
    )

    print("\nSaved: jcmm_comment6_results.npz")
    print("Original project files were not modified.")


if __name__ == "__main__":
    main()
