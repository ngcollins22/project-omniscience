import orekit as ok
from orekit.pyhelpers import setup_orekit_curdir
from mars_constellation import Constellation, ConstellationConfig, build_constellation, propagate, compute_ground_tracks, compute_pdop_p95_map, constellation_meets_5sat_requirement
from typing import List
from read_constellations import read_constellations, clean_path
from visualization import show_mars_basemap_from_file, plot_ground_tracks_on_basemap, plot_pdop_p95_map_on_basemap
from geometry import compute_los_latency_tensor, compute_self_dop_from_latencies, estimate_warm_start_time_metric, compute_network_metrics, calculateCost, approximateOverHeadPassTime
import csv

import numpy as np


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
                      'number_of_links', 'redundancy', 'degree_per_node', 'density_per_node', 'average_clustering_coefficient'
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
            
            basemap_path = "mars_1k_color.jpg"
            plot_ground_tracks_on_basemap(
                tracks,
                basemap_path=basemap_path,
                constellation=constellation,  # title auto-built from config
                silent=True # writes to files instead of showing
            )

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

            plot_pdop_p95_map_on_basemap(
                lat_vals,
                lon_vals,
                p95_pdop,
                basemap_path=basemap_path,
                title=f"PDOP P95 for Constellation {i+1} (lat +-45°)",
                silent=True # writes to files instead of showing
            )

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

            # Can definetinly do more with the latency matrices here but ignoring that for now

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
                'average_clustering_coefficient': av_average_clustering_coefficient
            })

def divisors(n: int) -> list[int]:
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def runSweepAnalysis(altRange = [6000, 7000], maxSats = 30):
    maxAlt = altRange[1]
    currAlt = altRange[0]

    duration_sec = 24 * 3600      # 24 hours
    step_sec = 300*6               # 30 min

    with open('constellation_data3.csv', 'w', newline='') as csvfile:
        fieldnames = ['name', 'inclination_deg', 'total_sats', 'planes', 'phasing', 'altitude_km', 'pattern', 'meets_pdop_6_requirement', 'p95_pdop', 'p95_warm_start_time_metric', 'number_of_nodes', 
                      'number_of_links', 'redundancy', 'degree_per_node', 'density_per_node', 'average_clustering_coefficient', 'overhead_pass_time', 'cost', 'mean_high_lat_P95_PDOP'
                      ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()            


        # sweep through alitude band
        attemptCount = 0
        validCount = 0
        for currAlt in range(altRange[0], altRange[1], 500):
            #sweep through inclination band
            for i in range(20, 65, 5):
                for numSats in range (12,maxSats): # need to determine number of sats required to meet min5 req
                    planes = divisors(numSats)
                    planes.pop(0)
                    
                    #exclude more than 6 planes since GPS uses that many
                    planes = [p for p in planes if p <= 6]


                    if len(planes) == 0:
                        continue
                    
                    for numPlanes in planes:
                        for f in range(0, (numPlanes-1)):
                            attemptCount = attemptCount+1
                            print("attempt count: "+str(attemptCount)+", valid count: "+str(validCount)+" alt:" + str(currAlt) + " i:"+str(i)+" sats:" + str(numSats) + " planes:" + str(numPlanes) + " f:" + str(f))
                            cfg = ConstellationConfig(
                            name = str("c"+(str(attemptCount))),
                            inclination_deg=float(i),
                            total_sats=int(numSats),
                            planes=int(numPlanes),
                            phasing=int(f),
                            altitude_km=float(currAlt),
                            pattern=str('DELTA'))

                            constellation = build_constellation(cfg)

                            #check requirements
                            if not constellation_meets_5sat_requirement(constellation):
                                continue

                            times, inertial_pvs, fixed_pvs = propagate(
                            constellation,
                            duration_sec=duration_sec,
                            step_sec=step_sec)

                            lat_vals, lon_vals, p95_pdop = compute_pdop_p95_map(constellation,
                            times)
                            meets_pdop_6_requirement = not (p95_pdop > 6).any()
                            if not meets_pdop_6_requirement:
                                continue
                            validCount = validCount+1
                            #if it reached this point, calculate everything
                            worst_pdop = p95_pdop.max()
                            latencies, sat_ids = compute_los_latency_tensor(inertial_pvs, body_radius_m=3389.5e3)  # Mars radius in meters

                            # Calculate the network metrics
                            number_of_links, redundancy, degree_per_node, density_per_node, average_clustering_coefficient = compute_network_metrics(latencies)
                            # Average metrics over time
                            av_number_of_links = np.mean(number_of_links)
                            av_redundancy = np.mean(redundancy)
                            av_degree_per_node = np.mean(degree_per_node)
                            av_density_per_node = np.mean(density_per_node)
                            av_average_clustering_coefficient = np.mean(average_clustering_coefficient)
                
                            dop_self = compute_self_dop_from_latencies(times, latencies, inertial_pvs, sat_ids) # compute self-DOP for each satellite
                            warm_start_time_metrics, p95_warm_start_time_metric = estimate_warm_start_time_metric(times, dop_self) # estimate warm-start time metric, see function for details
                            overheadPassTime = approximateOverHeadPassTime(cfg.altitude_km)
                            cost = calculateCost(cfg.planes, cfg.total_sats)
                            _, _, highLatP95PDOP = compute_pdop_p95_map(constellation, times, 45.0, 90.0)
                            meanHighLatPDOP = np.nanmean(highLatP95PDOP)
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
                                'mean_high_lat_P95_PDOP': meanHighLatPDOP 
                            })

                        





if __name__ == "__main__":
    ok.initVM()
    setup_orekit_curdir(from_pip_library=True)

    xlsx = clean_path(r"C:\Users\awt\Downloads\NewAHPandVSDlocal.xlsx")
    sheet = "Constellation Options"
    cell_range = "A3:G14"

    #run_analysis(xlsx, sheet, cell_range)
    runSweepAnalysis()
    

