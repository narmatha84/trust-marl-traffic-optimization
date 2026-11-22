
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import traci

from environment import SumoEnvironment
from agent import Agent, ACTIONS
from trust_module import TrustModule

SEEDS = [42, 123, 256, 789, 1024,
         2048, 4096, 8192, 16384, 32768]

STEPS = 300
GAMMA = 0.90
GAE_LAMBDA = 0.95
ROLLOUT = 32
PPO_EPOCHS = 4
BATCH_SIZE = 64


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discretize(speed, wait):
    s = min(int(speed // 2), 4)
    w = (0 if wait < 5 else
         1 if wait < 10 else
         2 if wait < 20 else 3)
    return 4 * s + w


def onehot(states):
    x = torch.zeros((len(states), 20))
    x[torch.arange(len(states)),
      torch.tensor(states)] = 1.0
    return x


def observe(env):
    result = {}
    for v in env.get_vehicles():
        _, speed, wait, _ = env.get_state(v)
        result[v] = (float(speed), float(wait))
    return result


def global_features(obs):
    if not obs:
        return np.zeros(3, dtype=np.float32), 0.0, 0.0

    speed = float(np.mean([v[0] for v in obs.values()]))
    wait = float(np.mean([v[1] for v in obs.values()]))

    g = np.array([
        min(speed / 8, 1),
        min(wait / 50, 1),
        min(len(obs) / 50, 1)
    ], dtype=np.float32)

    return g, speed, wait


def reward(speed, wait, avg_speed, avg_wait):
    individual = speed / 8 - wait / 15
    global_reward = avg_speed / 8 - avg_wait / 15
    return 0.7 * individual + 0.3 * global_reward


class SeededEnvironment(SumoEnvironment):
    def __init__(self, seed, binary):
        super().__init__("config.sumocfg")
        self.seed = int(seed)
        self.binary = binary

    def reset(self):
        traci.start([
            self.binary,
            "-c", self.config_file,
            "--seed", str(self.seed),
            "--no-step-log", "true"
        ])
        self.driver_profiles = {}


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)


class CentralizedCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(23, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, global_state):
        x = torch.cat([state, global_state], dim=-1)
        return self.net(x).squeeze(-1)


class CorrectedMAPPO:
    def __init__(self):
        self.actor = Actor()
        self.critic = CentralizedCritic()

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=3e-4
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=1e-3
        )

        self.updates = 0
        self.transitions = 0

    def act(self, state, global_state):
        with torch.no_grad():
            s = onehot([state])
            g = torch.from_numpy(
                global_state
            ).unsqueeze(0)

            distribution = Categorical(
                logits=self.actor(s)
            )
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            value = self.critic(s, g)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item())
        )

    def value(self, state, global_state):
        with torch.no_grad():
            s = onehot([state])
            g = torch.from_numpy(
                global_state
            ).unsqueeze(0)
            return float(self.critic(s, g).item())

    def optimize(self, data):
        if not data:
            return

        # Keep each vehicle's temporal trajectory separate.
        vehicle_indices = defaultdict(list)

        for i, transition in enumerate(data):
            vehicle_indices[
                transition["vid"]
            ].append(i)

        advantages = np.zeros(
            len(data), dtype=np.float32
        )

        # Correct, vehicle-wise GAE.
        for indices in vehicle_indices.values():
            gae = 0.0

            for i in reversed(indices):
                t = data[i]
                mask = 0.0 if t["done"] else 1.0

                delta = (
                    t["r"]
                    + GAMMA * mask * t["next_v"]
                    - t["v"]
                )

                gae = (
                    delta
                    + GAMMA * GAE_LAMBDA * mask * gae
                )

                advantages[i] = gae

        values = np.array(
            [t["v"] for t in data],
            dtype=np.float32
        )

        returns = torch.tensor(
            advantages + values
        )

        adv = torch.tensor(advantages)
        adv = (
            (adv - adv.mean())
            / (adv.std(unbiased=False) + 1e-8)
        )

        states = onehot(
            [t["state"] for t in data]
        )

        globals_ = torch.tensor(
            np.stack([t["g"] for t in data]),
            dtype=torch.float32
        )

        actions = torch.tensor(
            [t["action"] for t in data],
            dtype=torch.long
        )

        old_log_probs = torch.tensor(
            [t["logp"] for t in data],
            dtype=torch.float32
        )

        for _ in range(PPO_EPOCHS):
            indices = np.random.permutation(len(data))

            for start in range(0, len(indices), BATCH_SIZE):
                ix = torch.tensor(
                    indices[start:start+BATCH_SIZE]
                )

                distribution = Categorical(
                    logits=self.actor(states[ix])
                )

                new_log_probs = distribution.log_prob(
                    actions[ix]
                )

                ratio = torch.exp(
                    new_log_probs - old_log_probs[ix]
                )

                clipped = torch.clamp(
                    ratio, 0.8, 1.2
                )

                policy_loss = -torch.min(
                    ratio * adv[ix],
                    clipped * adv[ix]
                ).mean()

                value_loss = (
                    self.critic(states[ix], globals_[ix])
                    - returns[ix]
                ).square().mean()

                entropy = distribution.entropy().mean()

                loss = (
                    policy_loss
                    + 0.5 * value_loss
                    - 0.01 * entropy
                )

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 0.5
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), 0.5
                )

                self.actor_opt.step()
                self.critic_opt.step()

        self.updates += 1
        self.transitions += len(data)


def finish_pending(
    agent, pending, obs, global_state,
    avg_speed, avg_wait, buffer
):
    terminal_proxies = 0

    for vid, t in pending.items():
        done = vid not in obs

        if done:
            # SUMO no longer exposes completed vehicle state.
            # Use its last recorded observation as a proxy.
            speed, wait = t["pre"]
            terminal_proxies += 1
        else:
            speed, wait = obs[vid]

        t["r"] = reward(
            speed, wait, avg_speed, avg_wait
        )

        t["done"] = done

        t["next_v"] = (
            0.0 if done else agent.value(
                discretize(speed, wait),
                global_state
            )
        )

        buffer.append(t)

    return terminal_proxies


def run_mappo(agent, seed, binary, train):
    seed_all(seed)

    env = SeededEnvironment(seed, binary)
    env.reset()

    pending = {}
    buffer = []
    history = []

    totals = {
        "arrivals": 0,
        "departures": 0,
        "terminal_proxies": 0
    }

    def step():
        env.step()
        totals["arrivals"] += len(
            traci.simulation.getArrivedIDList()
        )
        totals["departures"] += len(
            traci.simulation.getDepartedIDList()
        )

    try:
        for k in range(STEPS):
            step()

            obs = observe(env)
            g, avg_speed, avg_wait = global_features(obs)

            # Complete previous decisions using the next
            # actual decision-time observation.
            if train and pending:
                totals["terminal_proxies"] += finish_pending(
                    agent, pending, obs, g,
                    avg_speed, avg_wait, buffer
                )

                pending = {}

                if k % ROLLOUT == 0:
                    agent.optimize(buffer)
                    buffer.clear()

            rewards = []

            for vid, (speed, wait) in obs.items():
                state = discretize(speed, wait)

                action, logp, value = agent.act(
                    state, g
                )

                env.apply_speed(
                    vid, ACTIONS[action]
                )

                if train:
                    pending[vid] = {
                        "vid": vid,
                        "state": state,
                        "g": g.copy(),
                        "action": action,
                        "logp": logp,
                        "v": value,
                        "pre": (speed, wait)
                    }

                # Common reporting metric matches M4's
                # pre-action metric calculation.
                rewards.append(
                    reward(
                        speed, wait,
                        avg_speed, avg_wait
                    )
                )

            history.append((
                float(np.mean(rewards))
                if rewards else 0.0,
                avg_wait,
                avg_speed
            ))

            # Matches M4's second simulation step.
            step()

        if train and pending:
            final_obs = observe(env)
            fg, fs, fw = global_features(final_obs)

            totals["terminal_proxies"] += finish_pending(
                agent, pending, final_obs,
                fg, fs, fw, buffer
            )

            agent.optimize(buffer)
            buffer.clear()

        totals["duration"] = float(
            traci.simulation.getTime()
        )

    finally:
        env.close()

    result = dict(
        zip(
            ["reward", "wait", "speed"],
            np.mean(history, axis=0)
        )
    )

    result.update(totals)

    result["throughput"] = (
        totals["arrivals"] * 3600
        / totals["duration"]
    )

    return result


def run_m4(agent, seed, binary, train):
    seed_all(seed)

    env = SeededEnvironment(seed, binary)
    trust = TrustModule()

    env.reset()

    epsilon = 0.1
    history = []

    totals = {
        "arrivals": 0,
        "departures": 0,
        "trained_decisions": 0,
        "total_decisions": 0
    }

    def step():
        env.step()
        totals["arrivals"] += len(
            traci.simulation.getArrivedIDList()
        )
        totals["departures"] += len(
            traci.simulation.getDepartedIDList()
        )

    try:
        for k in range(STEPS):
            step()

            obs = observe(env)
            _, avg_speed, avg_wait = global_features(obs)

            transitions = []
            rewards = []

            for vid, (speed, wait) in obs.items():
                trust.update(vid, speed, wait)
                t = float(trust.get(vid))

                state = discretize(speed, wait)

                agent.init(vid)

                totals["total_decisions"] += 1
                totals["trained_decisions"] += int(
                    agent.update_counts[vid] > 0
                )

                action = agent.choose(
                    vid, state, epsilon, t
                )

                env.apply_speed(
                    vid, ACTIONS[action]
                )

                r = reward(
                    speed, wait,
                    avg_speed, avg_wait
                )

                transitions.append((
                    vid, state, action, r
                ))

                rewards.append(r)

            step()

            if train:
                next_obs = observe(env)

                for vid, state, action, r in transitions:
                    if vid in next_obs:
                        speed, wait = next_obs[vid]

                        next_state = discretize(
                            speed, wait
                        )

                        # Original Agent.update(), unchanged.
                        agent.update(
                            vid, state, action,
                            r, next_state, 0.05, 0.9
                        )

            epsilon = max(
                0.05, epsilon * 0.995
            )

            history.append((
                float(np.mean(rewards))
                if rewards else 0.0,
                avg_wait,
                avg_speed
            ))

        totals["duration"] = float(
            traci.simulation.getTime()
        )

    finally:
        env.close()

    result = dict(
        zip(
            ["reward", "wait", "speed"],
            np.mean(history, axis=0)
        )
    )

    result.update(totals)

    result["throughput"] = (
        totals["arrivals"] * 3600
        / totals["duration"]
    )

    return result


def append_csv(filename, rows):
    if not rows:
        return

    with open(
        filename, "a", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )

        if f.tell() == 0:
            writer.writeheader()

        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=SEEDS
    )

    parser.add_argument(
        "--train-episodes", type=int,
        default=6
    )

    parser.add_argument(
        "--sumo", default="sumo"
    )

    args = parser.parse_args()

    if args.train_episodes < 2:
        parser.error(
            "At least two training episodes required."
        )

    torch.set_num_threads(1)

    results_file = (
        f"jcmm_comment15_results_e"
        f"{args.train_episodes}.csv"
    )

    curves_file = (
        f"jcmm_comment15_learning_e"
        f"{args.train_episodes}.csv"
    )

    completed_seeds = set()

    if Path(results_file).exists():
        with open(results_file, newline="") as f:
            for row in csv.DictReader(f):
                if (
                    row["algorithm"] ==
                    "MAPPO_extended_frozen"
                ):
                    completed_seeds.add(
                        int(row["seed"])
                    )

    for seed in args.seeds:
        if seed in completed_seeds:
            print(
                f"Skipping completed seed {seed}",
                flush=True
            )
            continue

        print(
            f"\nSEED {seed}: M4 training and evaluation",
            flush=True
        )

        seed_all(seed)
        m4 = Agent()

        m4_online = run_m4(
            m4, seed, args.sumo, train=True
        )

        eval_seed = seed + 100000

        m4_frozen = run_m4(
            m4, eval_seed, args.sumo, train=False
        )

        print(
            f"M4 frozen: reward={m4_frozen['reward']:.4f}, "
            f"wait={m4_frozen['wait']:.4f}, "
            f"speed={m4_frozen['speed']:.4f}",
            flush=True
        )

        print(
            f"SEED {seed}: corrected MAPPO training",
            flush=True
        )

        seed_all(seed)
        mappo = CorrectedMAPPO()

        curves = []
        matched_frozen = None

        for episode in range(args.train_episodes):
            episode_seed = seed + 1000 * episode

            training = run_mappo(
                mappo,
                episode_seed,
                args.sumo,
                train=True
            )

            curves.append({
                "seed": seed,
                "episode": episode + 1,
                "simulation_seconds_trained":
                    600 * (episode + 1),
                "reward": training["reward"],
                "wait": training["wait"],
                "speed": training["speed"],
                "gradient_updates": mappo.updates,
                "transitions": mappo.transitions
            })

            print(
                f"MAPPO episode {episode+1}: "
                f"reward={training['reward']:.4f}, "
                f"wait={training['wait']:.4f}, "
                f"speed={training['speed']:.4f}",
                flush=True
            )

            if episode == 0:
                matched_frozen = run_mappo(
                    mappo, eval_seed,
                    args.sumo, train=False
                )

                print(
                    "MAPPO equal-budget frozen: "
                    f"reward={matched_frozen['reward']:.4f}",
                    flush=True
                )

        extended_frozen = run_mappo(
            mappo, eval_seed,
            args.sumo, train=False
        )

        print(
            "MAPPO extended frozen: "
            f"reward={extended_frozen['reward']:.4f}, "
            f"wait={extended_frozen['wait']:.4f}, "
            f"speed={extended_frozen['speed']:.4f}",
            flush=True
        )

        rows = []

        for label, episodes, metrics in [
            ("M4_original_online", 1, m4_online),
            ("M4_frozen", 1, m4_frozen),
            ("MAPPO_matched_frozen", 1, matched_frozen),
            (
                "MAPPO_extended_frozen",
                args.train_episodes,
                extended_frozen
            )
        ]:
            rows.append({
                "seed": seed,
                "algorithm": label,
                "train_episodes": episodes,
                **{
                    key: metrics[key]
                    for key in [
                        "reward", "wait", "speed",
                        "throughput", "arrivals",
                        "departures", "duration"
                    ]
                }
            })

        append_csv(results_file, rows)
        append_csv(curves_file, curves)

        print(
            f"Seed {seed} saved successfully.",
            flush=True
        )

    print("\n" + "=" * 65)
    print("COMMENT 15: FROZEN POLICY EVALUATION")
    print("=" * 65)

    with open(results_file, newline="") as f:
        all_rows = list(csv.DictReader(f))

    for label in [
        "M4_frozen",
        "MAPPO_matched_frozen",
        "MAPPO_extended_frozen"
    ]:
        rows = [
            r for r in all_rows
            if r["algorithm"] == label
            and int(r["seed"]) in args.seeds
        ]

        if not rows:
            continue

        print(f"\n{label} | n={len(rows)}")

        for metric in [
            "reward", "wait", "speed", "throughput"
        ]:
            values = np.array(
                [float(r[metric]) for r in rows]
            )

            std = (
                values.std(ddof=1)
                if len(values) > 1 else 0.0
            )

            print(
                f"  {metric}: "
                f"{values.mean():.4f} +/- {std:.4f}"
            )

    print("\nSaved:")
    print(results_file)
    print(curves_file)

    print(
        "\nEqual-budget: 600s training + 600s frozen "
        "evaluation for both methods."
    )

    print(
        "Extended MAPPO uses additional training; "
        "this comparison is not budget-matched."
    )


if __name__ == "__main__":
    main()
