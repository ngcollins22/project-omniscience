# geometry.py

import math
from dataclasses import dataclass

import orekit
orekit.initVM()

from java.io import File

from org.orekit.bodies import CelestialBodyFactory, GeodeticPoint
from org.orekit.frames import Frame
from org.orekit.orbits import KeplerianOrbit, PositionAngleType, WalkerConstellation
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants

import numpy as np
from typing import List, Dict, Tuple


@dataclass
class MarsConstellationConfig:
    inclination_deg: float     # orbital inclination
    p: int                     # number of planes
    t: int                     # total satellites
    f: int                     # Walker phasing parameter
    altitude_km: float         # altitude above Mars mean radius
    pattern: str = "DELTA"     # "DELTA" or "STAR" (only DELTA implemented here)


@dataclass
class Satellite:
    sat_id: int
    plane: int
    slot_index: int
    propagator: KeplerianPropagator


@dataclass
class Constellation:
    cfg: MarsConstellationConfig
    epoch: AbsoluteDate
    mars: object
    inertial_frame: Frame
    body_frame: Frame
    satellites: list   # list[Satellite]


def build_mars_walker_constellation(cfg: MarsConstellationConfig,
                                    orekit_data_dir: str) -> Constellation:
    """
    Build a Mars Walker constellation using Orekit's WalkerConstellation.

    Returns a Constellation object with one KeplerianPropagator per satellite.
    """

    # Time / epoch
    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(2025, 1, 1, 0, 0, 0.0, utc)  # arbitrary reference epoch

    # Mars body, GM and frames
    mars = CelestialBodyFactory.getMars()
    mu = mars.getGM()                                # m^3/s^2
    inertial = mars.getInertiallyOrientedFrame()     # Mars-centered inertial frame 
    body = mars.getBodyOrientedFrame()               # Mars-fixed, rotates with the planet 

    # Mars mean radius (NASA number, meters) 
    R_MARS = 3389_500.0

    # Reference orbit (plane 0, sat 0)
    a = R_MARS + cfg.altitude_km * 1000.0
    e = 0.0 # circular
    i = math.radians(cfg.inclination_deg)
    omega = 0.0 # some simplifications
    raan0 = 0.0
    M0 = 0.0

    ref_orbit = KeplerianOrbit(
        a, e, i, omega, raan0, M0,
        PositionAngleType.MEAN,
        inertial,
        epoch,
        mu
    )

    # Walker builder
    if cfg.pattern.upper() == "DELTA":
        walker = WalkerConstellation(cfg.t, cfg.p, cfg.f)
    else:
        raise NotImplementedError("STAR pattern wiring not shown yet")

    # List<List<WalkerConstellationSlot>> 
    slots_by_plane = walker.buildRegularSlots(ref_orbit)

    satellites = []
    sat_id = 0
    for plane_idx in range(slots_by_plane.size()):
        plane_slots = slots_by_plane.get(plane_idx)
        for slot_idx in range(plane_slots.size()):
            slot = plane_slots.get(slot_idx)
            orbit = slot.getOrbit()
            prop = KeplerianPropagator(orbit)
            satellites.append(
                Satellite(
                    sat_id=sat_id,
                    plane=plane_idx,
                    slot_index=slot_idx,
                    propagator=prop
                )
            )
            sat_id += 1

    return Constellation(
        cfg=cfg,
        epoch=epoch,
        mars=mars,
        inertial_frame=inertial,
        body_frame=body,
        satellites=satellites
    )

def propagate_constellation(constellation: Constellation,
                            duration_sec: float,
                            step_sec: float):
    """
    Propagate every satellite with its KeplerianPropagator over [epoch, epoch+duration].

    Returns:
        times: list[AbsoluteDate]
        inertial_pvs: dict[sat_id] -> list[PVCoordinates]
    """

    n_steps = int(duration_sec // step_sec) + 1
    epoch = constellation.epoch  # typo guard

    times = [epoch.shiftedBy(k * step_sec) for k in range(n_steps)]

    inertial_pvs = {sat.sat_id: [] for sat in constellation.satellites}

    for t in times:
        for sat in constellation.satellites:
            pv = sat.propagator.getPVCoordinates(t)
            inertial_pvs[sat.sat_id].append(pv)

    return times, inertial_pvs

def pv_to_mars_fixed(constellation: Constellation,
                     times,
                     inertial_pvs):
    """
    Convert inertial PVCoordinates to Mars body-fixed frame for each sat/time.

    Returns:
        body_pvs: dict[sat_id] -> list[PVCoordinates in body frame]
    """
    inertial = constellation.inertial_frame
    body = constellation.body_frame

    body_pvs = {sat_id: [] for sat_id in inertial_pvs.keys()}

    for sat_id, pv_list in inertial_pvs.items():
        for t, pv_inertial in zip(times, pv_list):
            # Transform from inertial -> body-fixed 
            transform = inertial.getTransformTo(body, t)
            pv_body = transform.transformPVCoordinates(pv_inertial)
            body_pvs[sat_id].append(pv_body)

    return body_pvs

def get_los_between(pv1, pv2, radius_body) -> tuple | None:
    """
    Compute the line-of-sight (LOS) vector between two PVCoordinates,
    accounting for occultation by a spherical body of given radius. 
    PVCoordinates must be in a reference frame centered on the body.

    Returns:
        los_dist: norm of LOS vector if not occulted, else None
    """

    r1 = pv1.getPosition()
    r2 = pv2.getPosition()

    vec1 = np.array([r1.getX(), r1.getY(), r1.getZ()])
    vec2 = np.array([r2.getX(), r2.getY(), r2.getZ()])

    los = np.subtract(vec2, vec1)
    d = np.linalg.norm(los)

    # Check for occultation
    # Using geometric method: check closest approach to body center
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    los = vec2 - vec1
    d = np.linalg.norm(los)
    
    r0 = np.array([0.0, 0.0, 0.0])  # body center
    t = -np.dot(los, vec1 - r0) / (d * d)
    t = max(0.0, min(1.0, t))  # clamp to segment
    
    closest_point = vec1 + t * los
    dist_to_center = np.linalg.norm(closest_point)

    if dist_to_center < radius_body:
        return None  # occulted

    return los


def compute_los_latency_tensor(
    inertial_pvs: Dict[int, List],  # Dict[sat_id, List[PVCoordinates]]
    body_radius_m: float,
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute per-timestep satellite-to-satellite latency tensor with Mars
    occultation handled inside get_los_between.

    Returns:
        latencies : np.ndarray (T, N, N)
            One-way latency in seconds where a link exists, np.nan where no LoS.
        sat_ids : list[int]
            Ordering of satellites corresponding to matrix indices.
    """
    sat_ids = sorted(inertial_pvs.keys())
    id_to_idx = {sid: idx for idx, sid in enumerate(sat_ids)}

    n = len(sat_ids)
    num_times = len(next(iter(inertial_pvs.values())))
    c = Constants.SPEED_OF_LIGHT  # m/s

    latencies = np.full((num_times, n, n), np.nan, dtype=float)

    for t_idx in range(num_times):
        for sid_i, pv_list_i in inertial_pvs.items():
            i = id_to_idx[sid_i]
            pv_i = pv_list_i[t_idx]

            for sid_j, pv_list_j in inertial_pvs.items():
                j = id_to_idx[sid_j]
                if i >= j:
                    continue  # no self, enforce symmetry once

                pv_j = pv_list_j[t_idx]

                # You already have this; must return None if blocked,
                # and some vector (e.g. r_j - r_i) if LoS
                los_vec = get_los_between(pv_i, pv_j, body_radius_m)
                if los_vec is None:
                    continue  # leave as NaN = no link

                # If los_vec is the relative position vector:
                d_ij = np.linalg.norm(los_vec)       # meters

                tau = d_ij / c  # seconds

                latencies[t_idx, i, j] = tau
                latencies[t_idx, j, i] = tau

    return latencies, sat_ids


def compute_self_dop_from_latencies(times,
                                    latencies: np.ndarray,  # (T, N, N)
                                    inertial_pvs,
                                    sat_ids,
                                    min_neighbors: int = 4,
                                    max_cond: float = 1e8) -> np.ndarray:
    T, N, _ = latencies.shape
    dop_self = np.full((T, N), np.nan, dtype=float)

    for t in range(T):
        # positions at epoch t
        pos = []
        for sid in sat_ids:
            r = inertial_pvs[sid][t].getPosition()
            pos.append(np.array([r.getX(), r.getY(), r.getZ()], dtype=float))
        pos = np.stack(pos, axis=0)  # (N, 3)

        for i in range(N):
            # neighbors = finite latency entries
            neighbors = np.where(~np.isnan(latencies[t, i]))[0]
            neighbors = neighbors[neighbors != i]
            if neighbors.size < min_neighbors:
                continue

            r_i = pos[i]
            G_rows = []
            for j in neighbors:
                r_j = pos[j]
                d = r_j - r_i
                norm = np.linalg.norm(d)
                if norm <= 0:
                    continue
                G_rows.append(d / norm)

            if len(G_rows) < 3:
                continue

            G = np.vstack(G_rows)          # (M, 3)
            GTG = G.T @ G
            if np.linalg.cond(GTG) > max_cond:
                continue

            try:
                GTG_inv = np.linalg.inv(GTG)
            except np.linalg.LinAlgError:
                continue

            dop_self[t, i] = float(np.sqrt(np.trace(GTG_inv)))

    return dop_self

def estimate_warm_start_time_metric(times,
                                  dop_self: np.ndarray) -> float:
    # Per-epoch information proxy (average across satellites)
    information_rate = np.nanmean(1.0 / (dop_self ** 2), axis=0)  
    # Above computes 1/DOP^2 per satellite, then averages over time
    # Which provides us with a mean information rate per satellite

    # Then we can calculate "time to covergence"-like metric by inverting the mean information rate for each satellite
    warm_start_time_metrics = 1.0 / information_rate  # seconds

    # Then we take the P95 across satellites as the constellation-level metric
    p95_warm_start_time_metric = np.nanpercentile(warm_start_time_metrics, 95.0)
    return warm_start_time_metrics, p95_warm_start_time_metric


def compute_network_metrics(latency_tensor: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
        Compute:
        - number_of_links
        - redundancy
        - degree_per_node
        - density_per_node
        - average_clustering_coefficient
    """

    # Convert the latency tensor to a binary adjacency matrix
    #print(latency_tensor.shape)

    # Convert nan -> 0, finite -> 1
    adjacency_matrix = np.where(np.isnan(latency_tensor), 0, 1)

    n_nodes = adjacency_matrix.shape[1]

    nt = adjacency_matrix.shape[0]

    min_links = n_nodes - 1

    clustering_coeffs = np.zeros(nt, dtype=float) # one for each timestep
    #possible_links = np.zeros(nt, dtype=float) # one for each timestep
    average_degrees = np.zeros(nt, dtype=float) # one for each timestep
    densities = np.zeros(nt, dtype=float) # one for each timestep
    redundancies = np.zeros(nt, dtype=float) # one for each timestep
    number_of_links = np.zeros(nt, dtype=float) # one for each timestep

    # compute the metrics at each timestep and then average afterward
    for i in range(adjacency_matrix.shape[0]):
        adjacency_matrix_instant = adjacency_matrix[i] # grab the NxN matrix at time i

        number_of_links[i] = np.nansum(adjacency_matrix_instant) / 2  # undirected graph
        redundancies[i] = number_of_links[i] - min_links

        average_degrees[i] = np.nanmean(np.nansum(adjacency_matrix_instant, axis=0))
        densities[i] = average_degrees[i] / (n_nodes - 1)

        # Clustering coefficient
        clustering_coeffs_instant = np.zeros(n_nodes, dtype=float)
        for j in range(n_nodes):
            neighbors = np.where(adjacency_matrix_instant[j] > 0)[0]
            k = len(neighbors)
            if k < 2:
                clustering_coeffs_instant[j] = 0.0
                continue

            # Count links between neighbors
            links_between_neighbors = 0
            for m in range(k):
                for n in range(m + 1, k):
                    if adjacency_matrix_instant[neighbors[m], neighbors[n]] > 0:
                        links_between_neighbors += 1

            possible_links = k * (k - 1) / 2
            clustering_coeffs_instant[j] = links_between_neighbors / possible_links
        clustering_coeffs[i] = np.nanmean(clustering_coeffs_instant)
    
    return (number_of_links, redundancies, average_degrees, densities, clustering_coeffs)

def calculateConstellationCost(planes:int, sats:int)->float:
    """This function calculates the cost of ONLY the Mars constellation
        It assumes one falcon heavy launch per orbital plane
        and that each satellite costs the same as an average Earth GNSS satellite"""
    avgSatCost = 136400000
    FH_LaunchCost = 150e6

    return sats*avgSatCost + FH_LaunchCost*planes

def calculateOverHeadPassTime(altitude_km:float)->float:
    """This function calculates the overhead pass time in seconds
      for a satelite orbiting Mars at the specified altitude"""
    R = 3396.18255 # Mars radius in km
    ep_mask = 10*(np.pi/180)
    mu = 4.282837*1e4 #Mars gravitational parameter (km^3/s^2)

    A = R + altitude_km #semi major axis
    B=R

    cos_psi_max = (B*(np.cos(ep_mask)**2) + np.sin(ep_mask)*np.sqrt(A**2+B**2 * (np.cos(ep_mask)**2)))/A
    psi_max = np.arccos(cos_psi_max)

    w_omega = np.sqrt(mu/(A**3)) #mean angular rate for circular orbit around Mars
    omega_sat = w_omega*(B/A) # angular rate of sats subpoint on Mars
    omega_mars = (2*np.pi)/88642; #angular rotation of Mars

    O_omega = abs(omega_sat-omega_mars) 

    T_s = ((2*psi_max)/O_omega)
    return T_s

def compute_across_mars_latency(
                                constellation: Constellation, 
                                times: List[AbsoluteDate], 
                                latencies: np.ndarray, 
                                inertial_pvs: Dict[int, List], 
                                sat_ids: List[int],
                                t_idx: int = 0,
                                lat1: float = 0.0, 
                                lon1: float = 0.0, 
                                lat2: float = 0.0, 
                                lon2: float = 180.0) -> Tuple[float, List[int]]:
    """
        Compute worst-case surface-surface latency across Mars using the satellite constellation.
        Will use A* to find lowest-latency path between sats on opposite sides of Mars.
        Heuristic: straight-line distance / c
        Path Cost: sum of latencies along edges
    """

    mars_shape = constellation.mars_shape

    gp1 = GeodeticPoint(math.radians(lat1), math.radians(lon1), 0.0)
    gp2 = GeodeticPoint(math.radians(lat2), math.radians(lon2), 0.0)

    body = constellation.mars_fixed
    inertial = constellation.mars_inertial
    date = times[t_idx]

    # Body-fixed -> inertial transform at this date
    body_to_inertial = body.getTransformTo(inertial, date).toStaticTransform()

    # Geodetic -> Cartesian in body-fixed frame
    pos_gp1_body = mars_shape.transform(gp1)  # Vector3D in `body`
    pos_gp2_body = mars_shape.transform(gp2)

    # Body-fixed -> inertial
    pos_gp1_inertial = body_to_inertial.transformPosition(pos_gp1_body)
    pos_gp2_inertial = body_to_inertial.transformPosition(pos_gp2_body)

    # Numpy vectors
    r_gp1 = np.array([pos_gp1_inertial.getX(),
                    pos_gp1_inertial.getY(),
                    pos_gp1_inertial.getZ()])
    r_gp2 = np.array([pos_gp2_inertial.getX(),
                    pos_gp2_inertial.getY(),
                    pos_gp2_inertial.getZ()])
    # Find nearest satellite to each ground point
    sat_idx_gp1 = None
    sat_idx_gp2 = None
    min_dist1 = float("inf")
    min_dist2 = float("inf")
    for sid in sat_ids:
        pv_sat = inertial_pvs[sid][t_idx]
        r_sat = np.array([pv_sat.getPosition().getX(),
                          pv_sat.getPosition().getY(),
                          pv_sat.getPosition().getZ()])
        dist1 = np.linalg.norm(r_sat - r_gp1)
        dist2 = np.linalg.norm(r_sat - r_gp2)
        if dist1 < min_dist1:
            min_dist1 = dist1
            sat_idx_gp1 = sid
        if dist2 < min_dist2:
            min_dist2 = dist2
            sat_idx_gp2 = sid

    # Quickly calculate latency between ground point and nearest satellite
    c = Constants.SPEED_OF_LIGHT
    tau_gp1_to_sat = min_dist1 / c
    tau_gp2_to_sat = min_dist2 / c

    # We'll add those later on
    
    # Now we have two sats to connect via A*
    """
        Recall that the latency_tensor is (T, N, N) with NaN for no link.
        We'll build a graph where each node is a satellite, and edges exist
        where latency_tensor[t_idx, i, j] is finite, and the cost is that latency.
    """

    from queue import PriorityQueue

    # forward def of heuristic for A*
    # considered precomputing all heuristics here for simplicity, since we already know the goal sat, but that's actually less efficient in time and space
    def heuristic(sat_a_idx, sat_b_idx):
        # Straight-line distance between satellites / c
        pv_a = inertial_pvs[sat_a_idx][t_idx]
        pv_b = inertial_pvs[sat_b_idx][t_idx]
        r_a = np.array([pv_a.getPosition().getX(),
                        pv_a.getPosition().getY(),
                        pv_a.getPosition().getZ()])
        r_b = np.array([pv_b.getPosition().getX(),
                        pv_b.getPosition().getY(),
                        pv_b.getPosition().getZ()])
        dist = np.linalg.norm(r_b - r_a)
        return dist / c # seconds
    
    def reconstruct_path(came_from, current_sid):
        total_path = [current_sid]
        while current_sid in came_from:
            current_sid = came_from[current_sid]
            total_path.append(current_sid)
        total_path.reverse()
        return total_path
    
    # forward def of A*
    def a_star(start_sid, goal_sid, k_t_idx=t_idx) -> Tuple[float, List[int]]:
        open_set = PriorityQueue()
        open_set.put((heuristic(start_sid, goal_sid), start_sid)) # (f_score, sat_id)

        came_from = {}
        g_score = {sid: float("inf") for sid in sat_ids}
        g_score[start_sid] = 0.0

        f_score = {sid: float("inf") for sid in sat_ids}
        f_score[start_sid] = heuristic(start_sid, goal_sid)

        while not open_set.empty():
            current_f, current_sid = open_set.get()

            if current_sid == goal_sid:
                return g_score[goal_sid], reconstruct_path(came_from, current_sid)

            current_idx = sat_ids.index(current_sid)

            for neighbor_idx, tau in enumerate(latencies[t_idx, current_idx]):
                if np.isnan(tau):
                    continue  # no link

                neighbor_sid = sat_ids[neighbor_idx]
                tentative_g_score = g_score[current_sid] + tau

                if tentative_g_score < g_score[neighbor_sid]:
                    came_from[neighbor_sid] = current_sid
                    g_score[neighbor_sid] = tentative_g_score
                    f_score[neighbor_sid] = tentative_g_score + heuristic(neighbor_sid, goal_sid)
                    open_set.put((f_score[neighbor_sid], neighbor_sid))


        return float("inf"), []  # no path found
    
    network_latency, path = a_star(sat_idx_gp1, sat_idx_gp2)

    total_latency = network_latency + tau_gp1_to_sat + tau_gp2_to_sat

    # Edit: want to return all latencies along the path as well.
    path_latencies = []
    for i in range(len(path) - 1):
        idx_a = sat_ids.index(path[i])
        idx_b = sat_ids.index(path[i+1])
        path_latencies.append(latencies[t_idx, idx_a, idx_b])

    # pre and append ground-sat latencies
    path_latencies.insert(0, tau_gp1_to_sat)
    path_latencies.append(tau_gp2_to_sat)

    return total_latency, path, path_latencies

    