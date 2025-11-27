import orekit as ok
from orekit.pyhelpers import setup_orekit_curdir
from mars_constellation import Constellation, ConstellationConfig, build_constellation, propagate, compute_ground_tracks, compute_pdop_p95_map, constellation_meets_5sat_requirement, pdop_p95_requirement_met, divisors, findMinSatsForGlobal
from typing import List
from read_constellations import read_constellations, clean_path
from visualization import show_mars_basemap_from_file, plot_ground_tracks_on_basemap, plot_pdop_p95_map_on_basemap, plot_across_mars_path
from geometry import compute_los_latency_tensor, compute_self_dop_from_latencies, estimate_warm_start_time_metric, compute_network_metrics, approximateOverHeadPassTime, calculateCost, compute_across_mars_latency
import csv
import time
import numpy as np

import argparse


def generate_sat_configs_from_df(df) -> List[ConstellationConfig]:
    #TODO: validate
    configs = []
    for _, row in df.iterrows():
        cfg = ConstellationConfig(
            name = str(row['Constellation']),
            inclination_deg=float(row['i0']),
            total_sats=int(row['t']),
            planes=int(row['p']),
            phasing=int(row['f']),
            altitude_km=float(row['h']),
            pattern=str(row.get('Pattern', 'DELTA')).upper()
        )
        print(cfg)
        configs.append(cfg)
    return configs

def run_analysis(xlsx, sheet, cell_range): # saves plots and prints results, doesn't show plots
    df = read_constellations(xlsx, sheet, cell_range)
    print(df)

    configs = generate_sat_configs_from_df(df)

    # Need to write the data into a .csv

    # --- propagate ---
    duration_sec = 24 * 3600      # 24 hours
    step_sec = 300               # 5 min

    
    with open('constellation_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['name', 'inclination_deg', 'total_sats', 'planes', 'phasing', 'altitude_km', 'pattern', 'meets_pdop_6_requirement', 'p95_pdop', 'p95_warm_start_time_metric', 'number_of_nodes', 
                      'number_of_links', 'redundancy', 'degree_per_node', 'density_per_node', 'average_clustering_coefficient', 'across_mars_latency', 'across_mars_num_sats_in_path'
                      ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()            

        # --- pick the first constellation ---
        for i, cfg in enumerate(configs):
            constellation = build_constellation(cfg) # build the constellation
            times, inertial_pvs, fixed_pvs = propagate(
                constellation,
                duration_sec=duration_sec,
                step_sec=step_sec
            )

            # --- compute ground tracks ---
            tracks = compute_ground_tracks(constellation, times, fixed_pvs)
            
            basemap_path = "mars_viking_full.jpg"
            # plot_ground_tracks_on_basemap(
            #     tracks,
            #     basemap_path=basemap_path,
            #     constellation=constellation,  # title auto-built from config
            #     silent=True # writes to files instead of showing
            # )

            # --- compute PDOP/P95 map ---
            lat_vals, lon_vals, p95_pdop = compute_pdop_p95_map(
            constellation,
            times,
            lat_min_deg=-45.0,
            lat_max_deg=45.0,
            lat_step_deg=5.0,
            lon_min_deg=-180.0,
            lon_max_deg=180.0,
            lon_step_deg=10.0,
            min_elev_deg=5.0
            )

            #Check if there a PDOP value somewhere > 6
            meets_pdop_6_requirement = not (p95_pdop > 6).any()
            worst_pdop = p95_pdop.max()
            print(f"Worst-case PDOP P95 over lat +-45°: {worst_pdop:.2f} for constellation '{cfg.name}' ({i}). Meets PDOP ≤ 6 requirement: {meets_pdop_6_requirement}")

            # plot_pdop_p95_map_on_basemap(
            #     lat_vals,
            #     lon_vals,
            #     p95_pdop,
            #     basemap_path=basemap_path,
            #     title=f"PDOP P95 for Constellation {i+1} (lat +-45°)",
            #     silent=True # writes to files instead of showing
            # )

            # --- Connection Matrix and latencies ---

            latencies, sat_ids = compute_los_latency_tensor(inertial_pvs, body_radius_m=3389.5e3)  # Mars radius in meters

            # Calculate the network metrics
            number_of_links, redundancy, degree_per_node, density_per_node, average_clustering_coefficient = compute_network_metrics(latencies)
            # Print the network metrics over time
            # print(f"Network Metrics for constellation '{cfg.name}' ({i}):")
            # print("Time Step\tNumber of Links\tRedundancy\tDegree per Node\tDensity per Node\tAverage Clustering Coefficient")
            # for t_idx in range(latencies.shape[0]):
            #     print(f"{t_idx}\t\t{number_of_links[t_idx]}\t\t{redundancy[t_idx]}\t\t{degree_per_node[t_idx]:.2f}\t\t{density_per_node[t_idx]:.4f}\t\t{average_clustering_coefficient[t_idx]:.4f}")
            # Average metrics over time
            av_number_of_links = np.mean(number_of_links)
            av_redundancy = np.mean(redundancy)
            av_degree_per_node = np.mean(degree_per_node)
            av_density_per_node = np.mean(density_per_node)
            av_average_clustering_coefficient = np.mean(average_clustering_coefficient)
            
            
            dop_self = compute_self_dop_from_latencies(times, latencies, inertial_pvs, sat_ids) # compute self-DOP for each satellite

            warm_start_time_metrics, p95_warm_start_time_metric = estimate_warm_start_time_metric(times, dop_self) # estimate warm-start time metric, see function for details
            # Note the first return is more detailed. See function for exactly what it contains.

            print(f"p95 Warm-Start Time Metric (not real time, proportional to real time) P95 over lat +-45°: {p95_warm_start_time_metric:.2f} seconds for constellation '{cfg.name}' ({i}).")

            # --- Compute worst-case latency from one side of mars to the other --- 
            across_mars_latency, path, path_latencies = compute_across_mars_latency(constellation, times, latencies, inertial_pvs, sat_ids)
            print(f"Worst-case across-Mars latency: {across_mars_latency:.2f} seconds using {len(path)} satellites for constellation '{cfg.name}' ({i}).")

            # Visualize the across-Mars path at the first timestep
            # plot_across_mars_path( # Uncomment to enable plotting
            #     constellation,
            #     times,
            #     inertial_pvs,
            #     sat_ids,
            #     path,
            #     path_latencies,
            #     mars_texture_path=None,
            #     t_idx=0, # lazy just make sure this is the same as in compute_across_mars_latency
            #     lat1=0.0,
            #     lon1=0.0,
            #     lat2=0.0,
            #     lon2=180.0
            # )

            # Need to compute age of clock and age of epheremis next
            # And approximate time-to-first-fix (TTFF) as well

            # for now, just print out the latency matrices at each time step
            # for t_idx, latency_matrix in enumerate(latency_matrices):
            #     print(f"Time step {t_idx}:")
            #     print(latency_matrix)

            # Write the data out
            writer.writerow({
                'name': cfg.name,
                'inclination_deg': cfg.inclination_deg,
                'total_sats': cfg.total_sats,
                'planes': cfg.planes,
                'phasing': cfg.phasing,
                'altitude_km': cfg.altitude_km,
                'pattern': cfg.pattern, 
                'meets_pdop_6_requirement': meets_pdop_6_requirement,
                'p95_pdop': worst_pdop,
                'p95_warm_start_time_metric': p95_warm_start_time_metric,
                'number_of_nodes': len(sat_ids),
                'number_of_links': av_number_of_links,
                'redundancy': av_redundancy,
                'degree_per_node': av_degree_per_node,
                'density_per_node': av_density_per_node,
                'average_clustering_coefficient': av_average_clustering_coefficient,
                'across_mars_latency': across_mars_latency, 
                'across_mars_num_sats_in_path': len(path)
            })


def runSweepAnalysis(altRange = [8000, 21000], maxSatsInput = 25):
    starttime = time.time()
    maxAlt = altRange[1]
    currAlt = altRange[0]

    duration_sec = 24 * 3600      # 24 hours
    step_sec = 300*6               # 30 min

    # precompute possible planes so it isn't done every iteration
    planes_by_sat_count = {}
    for numSats in range(12, maxSatsInput):
        d = divisors(numSats)
        d = [p for p in d if p != 1 and p <= 6]
        if d:
            planes_by_sat_count[numSats] = d
    

    with open('constellation_data2.csv', 'w', newline='') as csvfile:
        fieldnames = ['name', 'inclination_deg', 'total_sats', 'planes', 'phasing', 'altitude_km', 'pattern', 'meets_pdop_6_requirement', 'p95_pdop', 'p95_warm_start_time_metric', 'number_of_nodes', 
                      'number_of_links', 'redundancy', 'degree_per_node', 'density_per_node', 'average_clustering_coefficient', 'overhead_pass_time', 'cost', 'across_mars_latency', 'across_mars_num_sats_in_path', 
                       'min_sats_for_global', 'additional_constellation', 'upgrade_cost'
                      ] 

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()            

        attemptCount = 0
        validCount = 0
        # sweep through alitude band
        for currAlt in range(altRange[0], altRange[1], 500):
            #sweep through inclination band
            if currAlt < 14000:
                minSats = 16
                maxSats = maxSatsInput
            else:
                minsats = 14
                maxSats = 22
            for i in range(20, 60, 5):
                for numSats in range (minSats,maxSats+1): # need to determine number of sats required to meet min5 req
                    #get planes from precomputed list
                    planes = planes_by_sat_count.get(numSats)
                    if not planes:
                        continue
                    
                    for numPlanes in planes:
                        for f in range(0, (numPlanes)):
                            attemptCount = attemptCount+1
                            if attemptCount % 100 == 0:
                                print(f"attempt: {attemptCount}, valid: {validCount}, alt:{currAlt} i:{i} sats:{numSats} planes:{numPlanes} f:{f}")
                            cfg = ConstellationConfig(
                            name = str("c"+(str(attemptCount))),
                            inclination_deg=float(i),
                            total_sats=int(numSats),
                            planes=int(numPlanes),
                            phasing=int(f),
                            altitude_km=float(currAlt),
                            pattern=str('DELTA'))

                            constellation = build_constellation(cfg)

                            #check requirements (only checking one hemisphere for efficiency)
                            if not constellation_meets_5sat_requirement(constellation, lat_min_deg=0, lat_max_deg=45):
                                continue

                            times, inertial_pvs, fixed_pvs = propagate(
                            constellation,
                            duration_sec=duration_sec,
                            step_sec=step_sec)
                            
                            #meets_pdop_6_requirement = not (p95_pdop > 6).any() # OLD

                            # calls a more efficent PDOP check that exits as soon as it fails. check only northern hemisphere for efficiency,
                            # when worst case PDOP is calculated, we check everywhere
                            meets_pdop_6_requirement = pdop_p95_requirement_met(constellation, times)
                            if not meets_pdop_6_requirement:
                                continue
                            validCount = validCount+1
                            #if it reached this point, calculate everything

                            _, _, p95_pdop = compute_pdop_p95_map(constellation,
                            times)
                            
                            worst_pdop = p95_pdop.max()
                            #double check meets pdop requirement
                            if worst_pdop > 6:
                                continue
                            latencies, sat_ids = compute_los_latency_tensor(inertial_pvs, body_radius_m=3389.5e3)  # Mars radius in meters

                            # Calculate the network metrics
                            number_of_links, redundancy, degree_per_node, density_per_node, average_clustering_coefficient = compute_network_metrics(latencies)
                            # Average metrics over time
                            av_number_of_links = np.mean(number_of_links)
                            av_redundancy = np.mean(redundancy)
                            av_degree_per_node = np.mean(degree_per_node)
                            av_density_per_node = np.mean(density_per_node)
                            av_average_clustering_coefficient = np.mean(average_clustering_coefficient)

                            # Calculate surface-surface latency
                            # Gonna use A* for this - will need latencies (the graph) and satellite positions for heuristic
                            
                            # this just does for the first timestep for now
                            # Should do for all timesteps and average but this is good enough for now
                            across_mars_latency, path, path_latencies = compute_across_mars_latency(constellation, times, latencies, inertial_pvs, sat_ids)
                
                            dop_self = compute_self_dop_from_latencies(times, latencies, inertial_pvs, sat_ids) # compute self-DOP for each satellite
                            warm_start_time_metrics, p95_warm_start_time_metric = estimate_warm_start_time_metric(times, dop_self) # estimate warm-start time metric, see function for details
                            overheadPassTime = approximateOverHeadPassTime(cfg.altitude_km)
                            cost = calculateCost(cfg.planes, cfg.total_sats)
                            #_, _, highLatP95PDOP = compute_pdop_p95_map(constellation, times, 45.0, 0.0)
                            #meanHighLatPDOP = np.nanmean(highLatP95PDOP)

                            #Purpose: Find minimum additional satelites for global coverage (PDOP<6)
                            #call a function, pass it the original constellation, have it build new additional constellations until it finds the minimum
                            #sweep satelites first 0-21 in increasing order, all planes 1-4 planes( this excludes ), i = 75, all f, exit at earlist solution
                            #return number of additional sats, and the constellation that worked

                            minSatsForGlobal, c2 = findMinSatsForGlobal(constellation, times, planes_by_sat_count)
                        

                            if c2:
                                additionalConstString = str(c2.total_sats)+"/"+str(c2.planes)+"/"+str(c2.phasing)+" alt="+str(c2.altitude_km)+ " i="+str(c2.inclination_deg)
                                upgradeCost = calculateCost(c2.planes, c2.total_sats)

                            else:
                                additionalConstString = "None"
                                upgradeCost=0
                            
                            writer.writerow({
                                'name': cfg.name,
                                'inclination_deg': cfg.inclination_deg,
                                'total_sats': cfg.total_sats,
                                'planes': cfg.planes,
                                'phasing': cfg.phasing,
                                'altitude_km': cfg.altitude_km,
                                'pattern': cfg.pattern, 
                                'meets_pdop_6_requirement': meets_pdop_6_requirement,
                                'p95_pdop': worst_pdop,
                                'p95_warm_start_time_metric': p95_warm_start_time_metric,
                                'number_of_nodes': len(sat_ids),
                                'number_of_links': av_number_of_links,
                                'redundancy': av_redundancy,
                                'degree_per_node': av_degree_per_node,
                                'density_per_node': av_density_per_node,
                                'average_clustering_coefficient': av_average_clustering_coefficient, 
                                'overhead_pass_time': overheadPassTime, 
                                'cost': cost,
                                'across_mars_latency': across_mars_latency,
                                'across_mars_num_sats_in_path': len(path),
                                'min_sats_for_global': minSatsForGlobal, 
                                'additional_constellation': additionalConstString, 
                                'upgrade_cost':upgradeCost})
    endtime = time.time()
    elapsedTime = (endtime-starttime)/60
    print("Execution time: "+str(elapsedTime)+" minutes")

if __name__ == "__main__":
    ok.initVM()
    setup_orekit_curdir(from_pip_library=True)

    xlsx = clean_path(r"C:\Users\natha\OneDrive - Virginia Tech\Tabor, Andrew's files - RASCAL_MarsPNT_1\AHP and VSD Spreadsheets for Project\NEW AHP and VSD.xlsx")
    sheet = "Constellation Options"
    cell_range = "A3:G14"

    # Argparse --sweep, --excel
    parser = argparse.ArgumentParser(description="Run PNT VSD analysis.")
    parser.add_argument("--sweep", action="store_true", help="Run sweep analysis")
    parser.add_argument("--excel", action="store_true", help="Run Excel analysis")
    args = parser.parse_args()

    if args.excel:
        run_analysis(xlsx, sheet, cell_range)
    elif args.sweep:
        runSweepAnalysis()
    else:
        print("Please specify either --sweep or --excel to run the desired analysis.")

    
    

