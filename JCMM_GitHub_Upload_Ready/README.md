# JCMM Manuscript 1084 — SUMO / Trust-MARL reproducibility archive

This repository contains source files, SUMO configurations, recorded results, and analysis scripts from the original **per-vehicle DQN** traffic-speed-control study associated with JCMM manuscript 1084. The materials were selected from the authors' actual `sumo_demo` working folder. They have **not been replaced with a parameter-shared or signal-controlled redesign**.

## Important scientific/protocol distinctions

- `agent.py` creates a **separate DQN, target network, optimizer, and replay buffer for each vehicle ID**. This is independent per-vehicle learning, not a shared neural network.
- `trust_module.py` initializes trust at 1.0, uses smoothing coefficient 0.9, clamps to [0.1, 2.0], and sends trust <0.5 to random exploration. In the stated speed range, the >1.5 exploitation branch cannot be reached. Do not claim that branch operated in the reported experiments.
- Older `mappo.py` and `Mappo_asymmetric.py` and their `mappo_results*.npy` are **historical first-revision outputs**; do not interpret old comparisons as equal-duration. Use `jcmm_comment15_mappo_corrected.py` and `jcmm_comment15_results_e6.csv` for the corrected matched-duration and extended-training MAPPO analysis.
- Original `final_code.py` advances SUMO **once per iteration for Mode 1**, but **twice per iteration for DQN modes**. Its 300 iterations are consequently not equal simulated durations. Interpret legacy fixed-rule comparisons descriptively, not as clean matched-budget tests.
- The corrected MAPPO evaluation uses a common **600 s frozen evaluation** following 600 s or 3,600 s training. Matched duration does not imply matched optimizer updates. The six-episode MAPPO learning reward did not demonstrate a plateau.
- `jcmm_comment18_high_demand.csv` reports **online-training** stress tests, not necessarily independent frozen-policy evaluations. Under 600 veh/h/direction some vehicles could not depart within the 600 s horizon; unserved demand must be acknowledged.
- `jcmm_comment11_statistics.csv` has a `matched_duration` column. Comparisons marked `False` are descriptive only, not evidence for a fair matched-duration superiority claim.
- The original `config.sumocfg` contains a `time/end` tag, but the scripts themselves control simulation time using explicit TraCI calls. See the `steps` values in each experiment script.
- The source code and recorded CSVs were statically audited here; a full fresh SUMO rerun has **not been performed in this packaging environment**. Reproduction depends on SUMO, Python versions, and simulation settings.

## Source and simulation files

- **Core:** `agent.py`, `environment.py`, `trust_module.py`.
- **Original comparisons:** `final_code.py`, `final_code_timing.py`, `experiment_asymmetric.py`.
- **Original parameter studies:** `experiment_sensitivity.py`, `experiment_smoothening.py`, `experiment_clipping.py`.
- **Additional reviewer experiments:** `jcmm_comment1_original_addon.py` (exploration-matched control); `jcmm_comment3_random_baseline.py`; `jcmm_comment5_adaptive_baseline.py` / `max_pressue.py` (pressure-based *speed* heuristic, not classical signal-phase Max-Pressure); `jcmm_comment6_frozen_policy.py`; `jcmm_comment8_throughput.py`; `jcmm_comment9_driver_profiles.py`; `jcmm_comments13_14_sensitivity.py`; `jcmm_comment15_mappo_corrected.py`; `jcmm_comment18_high_demand.py`.
- **Statistics and figures:** `jcmm_comment11_stats.py`, `jcmm_comment16_curves.py`, `figures/`.
- **Network and routes:** `map.net.xml`, `nodes.nod.xml`, `edges.edg.xml`, `routes.rou.xml`, `routes_asymmetric.rou.xml`, all `*.sumocfg`, and three higher-demand `jcmm_comment18_routes_*.rou.xml` files.

## Setup (Windows / PowerShell)

1. Install **Eclipse SUMO**, make sure its `sumo.exe` and `sumo-gui.exe` are available; the manuscript reports SUMO 1.26.0. Ensure your system has a compatible Python/PyTorch installation.
2. In this repository directory, create a virtual environment and install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
3. The original `environment.py` launches **`sumo-gui`** (GUI required) and assumes `sumo-gui` is available via PATH. Scripts accepting `--sumo` can instead use a full path to `sumo.exe` or `sumo-gui.exe`.

## Examples

Original independent-DQN modes (launches SUMO GUI, emits original protocol results):

```powershell
python final_code.py
```

Corrected MAPPO comparison (600-second equal-duration train/eval plus six episodes of extended MAPPO training):

```powershell
python jcmm_comment15_mappo_corrected.py --train-episodes 6 --sumo 'C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe'
```

Replot recorded MAPPO episode training histories:

```powershell
python jcmm_comment16_curves.py
```

Higher-demand experiment (10 seeds, 200/400/600 veh/h/direction):

```powershell
python jcmm_comment18_high_demand.py --sumo 'C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe'
```

Recompute summary statistics directly from the archived CSV data (does **not** run SUMO):

```powershell
python verify_results.py
```

**Rerun warning:** Some scripts detect existing CSV results and skip completed seeds. To rerun experiments from scratch, use a clean copy of the source/config files **without the existing result CSVs**; preserve the archived results separately.

## Archived records

- `jcmm_comments13_14_results.csv`: 5 configurations × 10 seeds, reward / wait / speed.
- `jcmm_comment15_results_e6.csv`: 4 algorithm/training configurations × 10 seeds; includes frozen and online evaluations.
- `jcmm_comment15_learning_e6.csv`: MAPPO episode 1–6 training trajectories × 10 seeds; see `figures/` for graphs.
- `jcmm_comment18_high_demand.csv`: 3 traffic rates × 3 modes × 10 seeds.
- `comment1_original_results/`: exploration-gated versus exploration-matched results and 20 per-seed JSON traces.
- `comment3_random_results/`: uniformly random speed-selection baseline.
- `comment5_adaptive_results/`: archived queue-aware speed heuristic output; its exact generating implementation is **not separately identified in this folder**, so no exact re-run command is asserted for this CSV.
- Earlier `.npy`/`.npz` files reflect legacy or supplementary experiments. The legacy MAPPO values should not be used in comparisons with corrected MAPPO runs.

## Reproducibility limitations

This is a best-effort archive of the actual submitted-study working files, not a claim of full bit-for-bit reproduction across machines. Authors should reconcile all values in the final manuscript against saved per-seed records, verify the exact environment versions, and document the study's limitations. No separate software license has been granted by default; reuse permissions require the authors' decision.
