"""JCMM final-round reviewer Comment 1: exploration-frequency control.

ADD-ON ONLY. Use beside the UNMODIFIED files from the first-revision code.zip:
    agent.py, environment.py, trust_module.py, config.sumocfg, SUMO network/routes.

Mode 4: the exact original paper algorithm and experimental loop, with branch
        counters (the action-selection RNG sequence is preserved).
Mode 5: the same per-vehicle DQN and reward shaping, but replace trust gating
        with a trust-BLIND epsilon-greedy probability matched to the REALIZED
        random-action frequency of Mode 4 in the same nominal seed.

All results are written to comment1_original_results/. No original file edited.
CAUTION: It intentionally retains the first revision's simulation/reward timing.
It also inherits the original SUMO launch settings from environment.py.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from agent import ACTIONS, ACTION_DIM, Agent, state_to_tensor
from environment import SumoEnvironment
from trust_module import TrustModule

SEEDS = [42, 123, 256, 789, 1024, 2048, 4096, 8192, 16384, 32768]
OUT = Path(__file__).resolve().parent / 'comment1_original_results'


def discretize(speed, wait):
    # Bit-for-bit matching the state mapping in the first-revision final_code.py.
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


def seed_all(seed):
    # Same seeding as original final_code.py. SUMO behavior is controlled by
    # the original environment.py, which is intentionally not modified here.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class AuditedAgent(Agent):
    """Same original `choose` behavior, with branch counters; not a new network."""

    def __init__(self):
        super().__init__()
        self.decisions = 0
        self.forced_random = 0
        self.epsilon_random = 0
        self.matched_random = 0
        self.greedy = 0

    def _greedy(self, vid, state):
        with torch.no_grad():
            qvals = self.networks[vid](state_to_tensor(state))
        return int(torch.argmax(qvals).item())

    def choose_audited_trust(self, vid, state, epsilon, trust):
        # Identical ordering and calls to Agent.choose(): init, then branch.
        self.init(vid)
        self.decisions += 1
        if trust < 0.5:
            self.forced_random += 1
            return int(np.random.randint(ACTION_DIM))
        if trust > 1.5:
            self.greedy += 1
            return self._greedy(vid, state)
        if np.random.rand() < epsilon:
            self.epsilon_random += 1
            return int(np.random.randint(ACTION_DIM))
        self.greedy += 1
        return self._greedy(vid, state)

    def choose_matched_no_trust(self, vid, state, probability):
        self.init(vid)
        self.decisions += 1
        if np.random.rand() < probability:
            self.matched_random += 1
            return int(np.random.randint(ACTION_DIM))
        self.greedy += 1
        return self._greedy(vid, state)


def run_original_protocol(mode, seed, matched_probability=None, quiet=True):
    """Faithfully reproduces first-revision final_code.run_experiment for mode 4.

    For mode 5 the ONLY methodological substitution is action selection:
    replace `agent.choose(... epsilon, trust_score)` by
    `agent.choose_matched_no_trust(... matched_probability)`.
    """
    if mode not in (4, 5):
        raise ValueError('This add-on only runs original Mode 4 and control Mode 5.')
    if mode == 5 and (matched_probability is None or not 0 <= matched_probability <= 1):
        raise ValueError('Mode 5 requires a match probability in [0, 1].')

    seed_all(seed)
    agent = AuditedAgent()
    trust = TrustModule() if mode == 4 else None
    sumo = SumoEnvironment('config.sumocfg')
    epsilon, alpha, gamma = 0.1, 0.05, 0.9
    reward_history, wait_history, speed_history = [], [], []
    sumo.reset()
    try:
        # Keep the original 300 logical iterations and two-step DQN protocol.
        for step in range(300):
            sumo.step()
            vehicles = sumo.get_vehicles()
            vehicle_states = {}
            for v in vehicles:
                _, speed, wait, _ = sumo.get_state(v)
                vehicle_states[v] = (speed, wait)

            avg_speed = np.mean([s for s, w in vehicle_states.values()]) if vehicles else 0
            avg_wait = np.mean([w for s, w in vehicle_states.values()]) if vehicles else 0
            states_memory = []
            rewards = []

            for v in vehicles:
                speed, wait = vehicle_states[v]
                s = discretize(speed, wait)
                if mode == 4:
                    trust.update(v, speed, wait)
                    t = trust.get(v)
                    agent.init(v)
                    ai = agent.choose_audited_trust(v, s, epsilon, t)
                else:
                    agent.init(v)
                    ai = agent.choose_matched_no_trust(v, s, matched_probability)

                sumo.apply_speed(v, ACTIONS[ai])
                individual = speed / 8 - wait / 15
                global_reward = avg_speed / 8 - avg_wait / 15
                reward = 0.7 * individual + 0.3 * global_reward
                states_memory.append((v, s, ai, reward))
                rewards.append(reward)

            # Same second simulation step and skip of exited vehicles as original.
            sumo.step()
            next_vehicles = sumo.get_vehicles()
            for vid, state, ai, reward in states_memory:
                if vid in next_vehicles:
                    _, n_speed, n_wait, _ = sumo.get_state(vid)
                    ns = discretize(n_speed, n_wait)
                    agent.update(vid, state, ai, reward, ns, alpha, gamma)

            epsilon = max(0.05, epsilon * 0.995)
            reward_history.append(float(np.mean(rewards)) if rewards else 0.0)
            wait_history.append(float(avg_wait))
            speed_history.append(float(avg_speed))
            if not quiet:
                print(f'mode={mode} seed={seed} step={step} reward={reward_history[-1]:.4f} ' 
                      f'wait={wait_history[-1]:.4f} speed={speed_history[-1]:.4f}')
            time.sleep(0.02)  # same as original first-revision run
    finally:
        sumo.close()

    total_random = agent.forced_random + agent.epsilon_random + agent.matched_random
    return {
        'mode': mode,
        'seed': seed,
        'method': 'original_first_revision_online_training',
        'mean_reward': float(np.mean(reward_history)),
        'mean_wait': float(np.mean(wait_history)),
        'mean_speed': float(np.mean(speed_history)),
        'decisions': agent.decisions,
        'forced_random': agent.forced_random,
        'epsilon_random': agent.epsilon_random,
        'matched_random': agent.matched_random,
        'greedy': agent.greedy,
        'random_fraction': total_random / agent.decisions if agent.decisions else 0.0,
        'forced_fraction': agent.forced_random / agent.decisions if agent.decisions else 0.0,
        'target_probability': matched_probability,
        'steps': 300,
        'reward_history': reward_history,
        'wait_history': wait_history,
        'speed_history': speed_history,
    }


def main():
    parser = argparse.ArgumentParser(description='First revision protocol: Comment 1 matched exploration')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--verbose', action='store_true', help='Print every step (original behavior)')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    rows = []
    for seed in args.seeds:
        print(f'\nSEED {seed}: Measuring original Mode 4 exploration ...', flush=True)
        m4 = run_original_protocol(4, seed, quiet=not args.verbose)
        p = m4['random_fraction']
        print(f'   M4: reward={m4["mean_reward"]:.4f} wait={m4["mean_wait"]:.4f} '
              f'speed={m4["mean_speed"]:.4f} random={p:.2%} '
              f'forced={m4["forced_fraction"]:.2%}', flush=True)
        print(f'SEED {seed}: Running trust-blind Mode 5 with probability p={p:.5f} ...', flush=True)
        m5 = run_original_protocol(5, seed, matched_probability=p, quiet=not args.verbose)
        print(f'   M5: reward={m5["mean_reward"]:.4f} wait={m5["mean_wait"]:.4f} '
              f'speed={m5["mean_speed"]:.4f} random={m5["random_fraction"]:.2%} '
              f'gap={(m5["random_fraction"] - p)*100:+.2f} pp', flush=True)
        for data in (m4, m5):
            filepath = OUT / f'comment1_mode{data["mode"]}_seed{seed}.json'
            filepath.write_text(json.dumps(data, indent=2), encoding='utf-8')
        rows.append({
            'seed': seed,
            'm4_reward': m4['mean_reward'], 'm4_wait': m4['mean_wait'], 'm4_speed': m4['mean_speed'],
            'm5_reward': m5['mean_reward'], 'm5_wait': m5['mean_wait'], 'm5_speed': m5['mean_speed'],
            'm4_random': m4['random_fraction'], 'm4_forced': m4['forced_fraction'],
            'm5_random': m5['random_fraction'],
            'gap_percentage_points': (m5['random_fraction'] - m4['random_fraction']) * 100,
            'm4_decisions': m4['decisions'], 'm5_decisions': m5['decisions'],
        })
        csvpath = OUT / 'comment1_pairwise_summary.csv'
        with csvpath.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if rows:
        print('\n=== COMMENT 1 SUMMARY (original symmetric protocol; online training) ===')
        print('Seeds:', len(rows))
        for key in ('m4_reward', 'm5_reward', 'm4_wait', 'm5_wait', 'm4_speed', 'm5_speed',
                    'm4_random', 'm4_forced', 'm5_random'):
            value = np.mean([r[key] for r in rows])
            print(f'{key}: {value:.4f}')
        gaps = np.array([r['gap_percentage_points'] for r in rows])
        print(f'Rate-match mean abs gap: {np.mean(np.abs(gaps)):.2f} pp; maximum: {np.max(np.abs(gaps)):.2f} pp')
        print('Saved:', csvpath)
        print('NOT VALIDATED AS EXACT MATCH; please report realized rates and seed gaps.')


if __name__ == '__main__':
    main()
