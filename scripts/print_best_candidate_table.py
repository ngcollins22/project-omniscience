import json
import os
import sys
import math
import numpy as np

# ensure repo root is on sys.path so local modules can be imported when running from scripts/
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from mars_constellation import ConstellationConfig, build_constellation, propagate
from geometry import precompute_site_geometry
from deployment_search import compute_site_timeseries_metrics

# load candidate info
cand_path = os.path.join('results', 'candidate_1', 'info.json')
with open(cand_path, 'r') as f:
    info = json.load(f)

subsets_sids = info['subsets_sids']  # list of lists of sat IDs

# constellation config (must match find_best_four.py)
cfg = ConstellationConfig(
    name="mars_gnss",
    inclination_deg=55.0,
    altitude_km=9000.0,
    total_sats=24,
    planes=6,
    phasing=1,
)

const = build_constellation(cfg)

# mapping sat_id -> index
sat_ids = [s.sat_id for s in const.satellites]
id_to_index = {sid: i for i, sid in enumerate(sat_ids)}

# sites
sites_deg = [
    (5.0, 137.6, 0.0), # Gale Crater
    (18.4, 77.5, 0.0), # Jezero
    (18.2, 335.4, 0.0), # Oxia Planum
]

# propagation params (fine)
duration_sec = 88642.663 * 5
step_sec = 2 * 60.0

print('Propagating constellation (fine) — this may take some time...')
times, inertial_pvs, fixed_pvs = propagate(const, duration_sec, step_sec)

visible, uvec = precompute_site_geometry(const, times, inertial_pvs, sat_ids, sites_deg, min_elev_deg=10.0)

# Compute per-stage cumulative metrics
cumulative = []
used = set()
for stage_idx, stage in enumerate(subsets_sids, start=1):
    used.update(stage)
    union_indices = [id_to_index[sid] for sid in sorted(used)]
    metrics = compute_site_timeseries_metrics(union_indices, visible, uvec, times)
    mean_frac_ge3 = float(np.mean(metrics['frac_ge3']))
    mean_frac_ge4 = float(np.mean(metrics['frac_ge4']))
    mean_pdop_p95 = float(np.mean(metrics['pdop_p95']))
    worst_outage = float(np.max(1.0 - metrics['frac_ge4']))

    # revisit aggregate across sites: show median of medians and p95 of p95s (simple aggregation)
    revisit_medians = [m['median'] for m in metrics['revisit_stats']]
    revisit_p95s = [m['p95'] for m in metrics['revisit_stats']]
    revisit_worsts = [m['worst'] for m in metrics['revisit_stats']]

    print('\nStage %d — added %d sats: %s' % (stage_idx, len(stage), stage))
    print('  Union sat count: %d' % len(union_indices))
    print('  Mean fraction with >=3 sats (averaged across sites): %.4f' % mean_frac_ge3)
    print('  Mean fraction with >=4 sats (averaged across sites): %.4f' % mean_frac_ge4)
    print('  Mean PDOP P95 (avg across sites): %.3f' % mean_pdop_p95)
    print('  Worst outage fraction (1 - frac_ge4) across sites: %.4f' % worst_outage)
    print('  Revisit (median across sites): median=%.1f s, p95=%.1f s, worst=%.1f s' % (
        np.median(revisit_medians), np.median(revisit_p95s), np.median(revisit_worsts)
    ))

    # also print per-site breakdown briefly
    print('  Per-site (site_idx: frac_ge3, frac_ge4, pdop_p95):')
    for s in range(len(sites_deg)):
        print('    %d: %.3f, %.3f, %.3f' % (s, metrics['frac_ge3'][s], metrics['frac_ge4'][s], metrics['pdop_p95'][s]))

    cumulative.append({
        'stage': stage_idx,
        'added_sids': stage,
        'union_indices': union_indices,
        'metrics': metrics,
    })

print('\nDone.')
