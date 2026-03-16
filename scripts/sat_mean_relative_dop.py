"""Compute mean TWR DOP for each satellite relative to other visible satellites.

Outputs a CSV with columns:
 sat_id, mean_twr_pdop, twr_avail_frac, mean_num_neighbors

Usage:
  python .\scripts\sat_mean_relative_dop.py --duration-sol-mult 1 --step 120
"""
import os
import sys
import csv
import argparse
import numpy as np

# ensure local repo modules importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from mars_constellation import ConstellationConfig, build_constellation, propagate
from geometry import get_los_between


def main():
    parser = argparse.ArgumentParser(description='Per-satellite mean relative TWR DOP')
    parser.add_argument('--duration-sol-mult', type=float, default=1.0, help='number of Mars sidereal days')
    parser.add_argument('--step', type=float, default=120.0, help='propagation step in seconds')
    parser.add_argument('--out', type=str, default='results/sat_relative_dop.csv')
    args = parser.parse_args()

    cfg = ConstellationConfig(
        name="mars_gnss",
        inclination_deg=55.0,
        altitude_km=9000.0,
        total_sats=24,
        planes=6,
        phasing=1,
    )

    const = build_constellation(cfg)

    duration_sec = 88642.663 * args.duration_sol_mult

    print(f"Propagating constellation for {duration_sec} s at step {args.step} s...")
    times, inertial_pvs, fixed_pvs = propagate(const, duration_sec, args.step)

    sat_ids = [s.sat_id for s in const.satellites]
    N = len(sat_ids)
    T = len(times)

    # prepare storage
    twr_series = {sid: [] for sid in sat_ids}
    neigh_counts = {sid: [] for sid in sat_ids}

    # Mars radius for occultation checks (meters)
    try:
        body_radius = const.mars_shape.getEquatorialRadius()
    except Exception:
        # fallback approximate
        body_radius = 3394200.0

    print('Computing per-epoch neighbor geometry...')
    for t_idx, t in enumerate(times):
        # positions in inertial (meters)
        pos = {}
        for sid in sat_ids:
            pv = inertial_pvs[sid][t_idx]
            p = pv.getPosition()
            pos[sid] = np.array([p.getX(), p.getY(), p.getZ()], dtype=float)

        for i in sat_ids:
            unit_list = []
            for j in sat_ids:
                if j == i:
                    continue
                # check LOS (use body-centered occultation test)
                pv_i = inertial_pvs[i][t_idx]
                pv_j = inertial_pvs[j][t_idx]
                los = get_los_between(pv_i, pv_j, body_radius)
                if los is None:
                    continue
                # use vector from i->j
                v = pos[j] - pos[i]
                d = np.linalg.norm(v)
                if d <= 0:
                    continue
                unit_list.append(v / d)

            m = len(unit_list)
            neigh_counts[i].append(m)
            if m >= 3:
                G = np.vstack(unit_list)  # m x 3
                GTG = G.T @ G
                try:
                    if np.linalg.cond(GTG) < 1e12:
                        Q = np.linalg.inv(GTG)
                        twr_pdop = float(np.sqrt(np.trace(Q)))
                        twr_series[i].append(twr_pdop)
                    else:
                        twr_series[i].append(np.nan)
                except np.linalg.LinAlgError:
                    twr_series[i].append(np.nan)
            else:
                twr_series[i].append(np.nan)

    # compute per-satellite summaries
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sat_id', 'mean_twr_pdop', 'twr_avail_frac', 'mean_num_neighbors'])
        for sid in sat_ids:
            series = np.array(twr_series[sid], dtype=float)
            valid = ~np.isnan(series)
            mean_twr = float(np.nanmean(series)) if np.any(valid) else float('inf')
            avail = float(np.mean(valid))
            mean_neigh = float(np.mean(neigh_counts[sid]))
            w.writerow([sid, mean_twr, avail, mean_neigh])

    print(f'Wrote per-satellite relative TWR DOP to {args.out}')


if __name__ == '__main__':
    main()
