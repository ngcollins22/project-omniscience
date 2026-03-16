"""Export Orekit propagation results to CSV ephemeris files usable by STK.

Usage (from repo root):
  python .\scripts\export_to_stk.py --candidate 1 --stage 2 --step 120

This will:
 - build the same constellation used in the analysis
 - propagate at the given step (seconds)
 - write per-satellite CSV files under results/candidate_{n}/stage_{s}/ephem_sat{sat_id}.csv

CSV format columns:
 Time_UTC, X_km, Y_km, Z_km, VX_km_s, VY_km_s, VZ_km_s

Notes for STK import:
 - Create a new scenario and set the central body to Mars (Scenario Properties -> Central Body).
 - Insert -> Satellite -> Import Ephemeris and select the CSV. Choose time format = UTC (Gregorian),
   units: length = km, velocity = km/s, coordinate system = J2000 (inertial). Map columns accordingly.
 - Repeat for each satellite file, or use an STK script to batch import.

"""
import os
import sys
import argparse
import csv

# ensure local repo modules importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from mars_constellation import ConstellationConfig, build_constellation, propagate


def write_sat_ephem_csv(out_dir, sat_id, times, inertial_pvs):
    # inertial_pvs: dict[sat_id] -> list[PVCoordinates]
    fname = os.path.join(out_dir, f"ephem_sat{sat_id}.csv")
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        # header for readability
        w.writerow(["Time_UTC", "X_km", "Y_km", "Z_km", "VX_km_s", "VY_km_s", "VZ_km_s"])
        for t, pv in zip(times, inertial_pvs[sat_id]):
            # Orekit AbsoluteDate -> ISO UTC
            try:
                time_str = t.toString()
            except Exception:
                time_str = str(t)

            p = pv.getPosition()
            v = pv.getVelocity()
            # Orekit returns meters, convert to km
            row = [time_str,
                   p.getX() / 1000.0, p.getY() / 1000.0, p.getZ() / 1000.0,
                   v.getX() / 1000.0, v.getY() / 1000.0, v.getZ() / 1000.0]
            w.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Export constellation ephemerides for STK')
    parser.add_argument('--candidate', type=int, default=1)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--step', type=float, default=120.0, help='propagation step in seconds')
    parser.add_argument('--duration-sol-mult', type=float, default=5.0, help='number of Mars sidereal days to propagate')
    args = parser.parse_args()

    # Build constellation (must match your analysis config)
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

    print(f"Propagating constellation for {duration_sec} s with step {args.step} s...")
    times, inertial_pvs, fixed_pvs = propagate(const, duration_sec, args.step)

    # load candidate info file to get which sats are in the stage union (if available)
    cand_info = os.path.join('results', f'candidate_{args.candidate}', 'info.json')
    if not os.path.exists(cand_info):
        print(f"Candidate info {cand_info} not found. Exporting all satellites.")
        sat_ids = [s.sat_id for s in const.satellites]
    else:
        import json
        with open(cand_info, 'r') as f:
            info = json.load(f)
        # if stage folders were created earlier, try to read union_sids from stage info
        stage_info = os.path.join('results', f'candidate_{args.candidate}', f'stage_{args.stage}', 'info.json')
        if os.path.exists(stage_info):
            with open(stage_info, 'r') as sf:
                sinfo = json.load(sf)
            sat_ids = sinfo.get('union_sids', [s.sat_id for s in const.satellites])
        else:
            # fallback: read subsets_sids from candidate info and take cumulative union up to stage
            subsets = info.get('subsets_sids', [])
            union = set()
            for i, ss in enumerate(subsets, start=1):
                union.update(ss)
                if i == args.stage:
                    break
            sat_ids = sorted(list(union)) if union else [s.sat_id for s in const.satellites]

    out_dir = os.path.join('results', f'candidate_{args.candidate}', f'stage_{args.stage}', 'stk_ephemeris')
    os.makedirs(out_dir, exist_ok=True)

    # write per-satellite ephemeris CSVs
    for sid in sat_ids:
        print(f"Writing ephemeris for sat {sid}...")
        write_sat_ephem_csv(out_dir, sid, times, inertial_pvs)

    print(f"Wrote {len(sat_ids)} ephemeris files to {out_dir}")


if __name__ == '__main__':
    main()
