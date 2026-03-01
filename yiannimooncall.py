from yiannimoon import ConstellationConfig, build_moonconstellation, propagate, precompute_site_geometry
from geometry import find_best_four_sats,score_subset,evaluate_subset,dop_from_unit_los
from moonconstellationplot import plot_constellation

cfg = ConstellationConfig(
    name="moon_test",
    inclination_deg=55.0,
    altitude_km= 8000.0,
    total_sats=24,
    planes=6,
    phasing=1,
)

const = build_moonconstellation(cfg)

# Visualize constellation
#plot_constellation(const) 
#print(f"Total satellites created: {len(const.satellites)}")

# time grid: start with 1 sol at 120s step (fast), then refine later
duration_sec = 86400 * 365 * 4
step_sec = 3600*24*5

times, inertial_pvs, fixed_pvs = propagate(const, duration_sec, step_sec)
sat_ids = [s.sat_id for s in const.satellites]

sites_deg = [
    (0.67, 23.47, 0.0), #Sea of Tranquility, Apollo 11 (NE)
    (-3.01, -32.42, 0.0), #Ocean of Storms, Apollo 12 (SW)
    (-3.65, -17.47, 0.0)  #Fra Mauro, Apollo 14 (SW)
]

visible, uvec = precompute_site_geometry(
    const, times, inertial_pvs, sat_ids, sites_deg, min_elev_deg=10.0
)

best = find_best_four_sats(sat_ids, visible, uvec, top_k=20)

for rank, (sc, subset_idx, site_results) in enumerate(best[:10], start=1):
    subset_sids = [sat_ids[i] for i in subset_idx]
    worst_outage = max(r["outage_frac"] for r in site_results) #days
    worst_hdop_p95 = max(r["hdop_p95"] for r in site_results)
    print(rank, sc, subset_sids, "worst_outage", worst_outage, "worst_hdop_p95", worst_hdop_p95)