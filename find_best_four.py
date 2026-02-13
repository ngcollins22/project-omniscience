from mars_constellation import ConstellationConfig, build_constellation, propagate
from geometry import precompute_site_geometry, find_best_four_sats




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
duration_sec = 88642.663 * 365 * 4
step_sec = 120.0

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

best = find_best_four_sats(sat_ids, visible, uvec, top_k=20)

for rank, (sc, subset_idx, site_results) in enumerate(best[:10], start=1):
    subset_sids = [sat_ids[i] for i in subset_idx]
    worst_outage = max(r["outage_frac"] for r in site_results)
    worst_hdop_p95 = max(r["hdop_p95"] for r in site_results)
    print(rank, sc, subset_sids, "worst_outage", worst_outage, "worst_hdop_p95", worst_hdop_p95)
