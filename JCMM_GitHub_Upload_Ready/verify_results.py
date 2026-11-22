"""Offline summary audit of recorded per-seed CSVs (does not run SUMO)."""
import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev
ROOT=Path(__file__).resolve().parent
SPECS=[
('jcmm_comments13_14_results.csv',['config'],['reward','wait_s','speed_m_s'],5,10),
('jcmm_comment15_results_e6.csv',['algorithm'],['reward','wait','speed','throughput'],4,10),
('jcmm_comment15_learning_e6.csv',['episode'],['reward','wait','speed'],6,10),
('jcmm_comment18_high_demand.csv',['rate_per_direction','mode'],['reward','wait_s','speed_m_s','throughput_veh_h'],9,10),
]
for fn, group_fields, metrics, expected_groups, expected_each in SPECS:
 with (ROOT/fn).open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 groups=defaultdict(list)
 for row in rows:groups[tuple(row[x] for x in group_fields)].append(row)
 print(f'\n{fn}: {len(rows)} records across {len(groups)} groups')
 assert len(groups)==expected_groups, f'Expected {expected_groups} groups'
 for group,part in sorted(groups.items()):
  assert len(part)==expected_each, f'{group}: expected {expected_each} records'
  if 'seed' in part[0]:
   assert len({r['seed'] for r in part})==len(part), f'{group}: duplicate seed'
  stats=' | '.join(f'{m}: {mean(float(r[m]) for r in part):.4f} +/- {stdev(float(r[m]) for r in part):.4f}' for m in metrics)
  print(' ', '/'.join(group), stats)
print('\nPASS: all four CSV record sets have complete ten-seed groups.')
