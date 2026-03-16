from mars_constellation import ConstellationConfig, build_constellation, propagate
from geometry import precompute_site_geometry, find_best_four_sats
from deployment_search import beam_search_deployment, compute_site_timeseries_metrics, dump_summary_csv, plot_site_timeseries
import os
import json
import numpy as np




cfg = ConstellationConfig(
    name="mars_gnss",
    inclination_deg=55.0,
    altitude_km=9000.0,
    total_sats=24,
    planes=6,
    phasing=1,
)

const = build_constellation(cfg)

# time grid: start with 1 sol at 120s step (fast), then refine later
duration_sec = 88642.663 * 5
step_sec = 10*60.0

times, inertial_pvs, fixed_pvs = propagate(const, duration_sec, step_sec)
sat_ids = [s.sat_id for s in const.satellites]

sites_deg = [
    (5.0, 137.6, 0.0), # Gale Crater
    (18.4, 77.5, 0.0), # Jezero
    (18.2, 335.4, 0.0), # Oxia Planum
]

visible, uvec = precompute_site_geometry(
    const, times, inertial_pvs, sat_ids, sites_deg, min_elev_deg=10.0
)

# find best individual 4-satellite subsets (quick diagnostic)
best = find_best_four_sats(sat_ids, visible, uvec, top_k=20)

for rank, (sc, subset_idx, site_results) in enumerate(best[:10], start=1):
    subset_sids = [sat_ids[i] for i in subset_idx]
    worst_outage = max(r["outage_frac"] for r in site_results)
    worst_hdop_p95 = max(r["hdop_p95"] for r in site_results)
    print(rank, sc, subset_sids, "worst_outage", worst_outage, "worst_hdop_p95", worst_hdop_p95)


# Now run a beam-search to find good deployment sequences (4,8,8,4)
N = len(sat_ids)
sat_indices = list(range(N))
# build mapping index -> plane index from constellation satellites
plane_of_index = {i: const.satellites[i].plane_index for i in sat_indices}

print("Running beam search over deployment sequences (coarse step)")
results = beam_search_deployment(sat_indices, plane_of_index, visible, uvec, times, stages=[4,8,8,4], beam_size=200)

# take top 10 candidates and re-evaluate with a finer propagation (2 minute step)
top_k = 10
fine_step_sec = 2 * 60.0
print(f"Refining top {top_k} candidates with finer propagation (step={fine_step_sec}s)")

for rank, (prim_score, subsets, metrics_coarse) in enumerate(results[:top_k], start=1):
    # map subsets (which are tuples of indices) to sat_ids for display
    subsets_sids = [[sat_ids[i] for i in s] for s in subsets]
    union_indices = sorted(set().union(*[set(s) for s in subsets]))

    # perform fine propagation once per candidate and recompute visibility
    print(f"Candidate {rank}: prim_score={prim_score:.4f}, subsets={subsets_sids}")
    times_f, inertial_pvs_f, fixed_pvs_f = propagate(const, duration_sec, fine_step_sec)
    visible_f, uvec_f = precompute_site_geometry(const, times_f, inertial_pvs_f, sat_ids, sites_deg, min_elev_deg=10.0)

    # For each stage, compute cumulative union metrics and write per-stage outputs
    out_dir = os.path.join("results", f"candidate_{rank}")
    os.makedirs(out_dir, exist_ok=True)

    per_stage_metrics = []
    cumulative_used = set()
    for s_idx, stage_subset in enumerate(subsets, start=1):
        cumulative_used.update(stage_subset)
        union_idx = sorted(cumulative_used)

        metrics_fine_stage = compute_site_timeseries_metrics(union_idx, visible_f, uvec_f, times_f)
        mean_frac_ge3 = float(metrics_fine_stage['frac_ge3'].mean())
        mean_frac_ge4 = float(metrics_fine_stage['frac_ge4'].mean())
        mean_pdop_p95 = float(np.mean(metrics_fine_stage['pdop_p95']))

        print(f"  Stage {s_idx} (added {len(stage_subset)} sats): mean_frac_ge3={mean_frac_ge3:.4f}, mean_frac_ge4={mean_frac_ge4:.4f}, mean_pdop_p95={mean_pdop_p95:.3f}")

        # store stage metrics
        per_stage_metrics.append({
            'stage': s_idx,
            'added_count': len(stage_subset),
            'union_indices': union_idx,
            'metrics': metrics_fine_stage,
            'mean_frac_ge3': mean_frac_ge3,
            'mean_frac_ge4': mean_frac_ge4,
            'mean_pdop_p95': mean_pdop_p95,
        })

        # write stage-specific outputs
        stage_dir = os.path.join(out_dir, f"stage_{s_idx}")
        os.makedirs(stage_dir, exist_ok=True)

        summary_csv = os.path.join(stage_dir, "summary.csv")
        dump_summary_csv(summary_csv, subsets, metrics_fine_stage, sat_ids)

        info = {
            'candidate_rank': rank,
            'stage': s_idx,
            'added_subset_sids': [sat_ids[i] for i in stage_subset],
            'union_sids': [sat_ids[i] for i in union_idx],
            'mean_frac_ge3': mean_frac_ge3,
            'mean_frac_ge4': mean_frac_ge4,
            'mean_pdop_p95': mean_pdop_p95,
        }
        with open(os.path.join(stage_dir, 'info.json'), 'w') as jf:
            json.dump(info, jf, indent=2)

        # plots per site for this stage
        for site_i in range(len(sites_deg)):
            prefix = os.path.join(stage_dir, f"timeseries")
            plot_site_timeseries(prefix, times_f, metrics_fine_stage, site_i)

    # attach per-stage metrics back to results
    results[rank-1] = (prim_score, subsets, per_stage_metrics)


# After refining candidates, print a succinct per-stage table for the best candidate
if len(results) > 0:
    best = results[0]
    prim_score, subsets_best, per_stage = best

    print('\nFinal summary table for best candidate (by primary score)')
    hdr = ('Stage', 'Added_sids', 'TWR_uptime(>=3)%', 'OWR_uptime(>=4)%', 'Comms_uptime(>1)%',
           'TWR_DOP_P95', 'TWR_avail(%)', 'OWR_DOP_P95', 'OWR_avail(%)', 'Mean_comms_outage(s)')
    print(' | '.join(hdr))
    print('-' * 120)

    for entry in per_stage:
        s_idx = entry['stage']
        added = [sat_ids[i] for i in entry['union_indices'] if i in entry['union_indices']]  # list of union sids
        metrics = entry['metrics']

        twr_uptime = float(np.mean(metrics['frac_ge3'])) * 100.0
        owr_uptime = float(np.mean(metrics['frac_ge4'])) * 100.0
        comms_uptime = float(np.mean((metrics['vis_count_series'] > 0).mean(axis=1))) * 100.0

        # DOPs and availability
        twr_dop_p95 = float(np.mean(metrics.get('twr_pdop_p95', np.array([float('inf')]))))
        twr_avail = float(np.mean(metrics.get('twr_pdop_avail_frac', np.array([0.0])))) * 100.0

        owr_dop_p95 = float(np.mean(metrics.get('pdop_p95', np.array([float('inf')]))))
        owr_avail = float(np.mean(metrics.get('pdop_avail_frac', np.array([0.0])))) * 100.0

        # mean comms outage: average across sites of revisit mean
        revisit_means = [rs.get('mean', float('inf')) for rs in metrics['revisit_stats']]
        mean_comms_outage = float(np.mean([v for v in revisit_means if not np.isinf(v)]) ) if any(not np.isinf(v) for v in revisit_means) else float('inf')

        print(f"{s_idx:^5} | {str(added):^20} | {twr_uptime:>14.2f} | {owr_uptime:>13.2f} | {comms_uptime:>12.2f} | {twr_dop_p95:>11.3f} | {twr_avail:>10.1f} | {owr_dop_p95:>11.3f} | {owr_avail:>10.1f} | {mean_comms_outage:>18.1f}")
