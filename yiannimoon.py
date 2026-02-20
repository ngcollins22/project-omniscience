#Yianni tries python orekit yay (balls!)
#check propogate and build constellation
import math
import orekit
import numpy as np
orekit.initVM()

from typing import Optional
from dataclasses import dataclass
from typing import List, Dict, Tuple

from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir(from_pip_library=True)

from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid, GeodeticPoint
from org.orekit.frames import Frame, TopocentricFrame
from org.orekit.orbits import KeplerianOrbit, PositionAngleType, WalkerConstellation
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import PVCoordinates
from org.orekit.orbits import WalkerConstellationSlot
from org.orekit.gnss import DOPComputer


from java.util import List as JavaList
from java.util import ArrayList

#added imports
import itertools
import heapq


@dataclass
class ConstellationConfig:
    name: str
    inclination_deg: float
    altitude_km: float
    total_sats: int
    planes: int
    phasing: int
    pattern: str = "DELTA"  # Only "DELTA" supported for now

    def __print__(self):
        return (f"ConstellationConfig(name={self.name}, inclination_deg={self.inclination_deg}, "
                f"altitude_km={self.altitude_km}, total_sats={self.total_sats}, "
                f"planes={self.planes}, phasing={self.phasing}, pattern={self.pattern})")
    
    def __str__(self):
        return (f"{self.name}: Walker {self.pattern} {self.total_sats}/{self.planes}/{self.phasing} @ i={self.inclination_deg}°, h={self.altitude_km} km")

@dataclass
class Satellite:
    sat_id: int
    plane_index: int
    slot_index: int
    propagator: KeplerianPropagator


@dataclass
class Constellation:
    config: ConstellationConfig
    epoch: AbsoluteDate
    satellites: List[Satellite]
    moon_inertial: Frame
    moon_fixed: Frame
    moon_shape: OneAxisEllipsoid

def build_moonconstellation(config: ConstellationConfig) -> Constellation:
    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(2025, 1, 1, 0, 0, 0.0, utc)

    moon = CelestialBodyFactory.getMoon()
    mu = moon.getGM()                         # gravitational parameter (m^3/s^2)
    inertial = moon.getInertiallyOrientedFrame()
    fixed = moon.getBodyOrientedFrame()

    # Mars ellipsoid (IAU value)
    R_moon = 1737400.0  # meters
    f_moon = 0.0012475
    moon_shape = OneAxisEllipsoid(R_moon, f_moon, fixed)

    # Reference orbit
    a = R_moon + config.altitude_km * 1e3
    e = 0.0
    i = math.radians(config.inclination_deg)
    omega = 0.0
    raan0 = 0.0
    M0 = 0.0

    ref_orbit = KeplerianOrbit(a, e, i, omega, raan0, M0,
                                PositionAngleType.MEAN,
                                inertial, epoch, mu)

    # Walker constellation
    walker = WalkerConstellation(config.total_sats, config.planes, config.phasing)
    slot_matrix = walker.buildRegularSlots(ref_orbit)

    satellites = []
    sat_id = 0
    for p in range(slot_matrix.size()):
        plane_slots = JavaList.cast_(slot_matrix.get(p))
        for s in range(plane_slots.size()):
            slot = WalkerConstellationSlot.cast_(plane_slots.get(s))
            orbit = slot.getOrbit()
            prop = KeplerianPropagator(orbit)
            satellites.append(Satellite(sat_id, p, s, prop))
            sat_id += 1

    return Constellation(
        config=config, 
        epoch=epoch,
        satellites=satellites,
        moon_inertial=inertial,
        moon_fixed=fixed,
        moon_shape=moon_shape
    )

def propagate(constellation: Constellation,
              duration_sec: float,
              step_sec: float
              ) -> Tuple[List[AbsoluteDate], Dict[int, List[PVCoordinates]], Dict[int, List[PVCoordinates]]]:
    times = [constellation.epoch.shiftedBy(float(t)) for t in range(0, int(duration_sec)+1, int(step_sec))]

    inertial_pvs = {sat.sat_id: [] for sat in constellation.satellites}
    fixed_pvs = {sat.sat_id: [] for sat in constellation.satellites}

    for t in times:
        for sat in constellation.satellites:
            state = sat.propagator.propagate(t)
            pv = state.getPVCoordinates()
            
            inertial_pvs[sat.sat_id].append(pv)

            transform = constellation.moon_inertial.getTransformTo(constellation.moon_fixed, t)
            pv_fixed = transform.transformPVCoordinates(pv)
            fixed_pvs[sat.sat_id].append(pv_fixed)

    return times, inertial_pvs, fixed_pvs

def precompute_site_geometry(constellation,
                             times,
                             inertial_pvs,   # dict[sat_id] -> list[PVCoordinates] in inertial frame
                             sat_ids,        # list of sat_ids in the order you want
                             sites_deg,      # list[(lat_deg, lon_deg, alt_m)]
                             min_elev_deg=10.0):
    """
    Returns:
      visible: bool array (S, T, N)
      uvec:    float array (S, T, N, 3) unit vectors user->sat (in inertial frame)
    """
    moon_shape = constellation.moon_shape
    body = constellation.moon_fixed
    inertial = constellation.moon_inertial

    S = len(sites_deg)
    T = len(times)
    N = len(sat_ids)

    min_elev_rad = math.radians(min_elev_deg)

    # Build topo frames and store geodetic points
    topo_frames = []
    gpoints = []
    for (lat_deg, lon_deg, alt_m) in sites_deg:
        gp = GeodeticPoint(math.radians(lat_deg), math.radians(lon_deg), float(alt_m))
        topo = TopocentricFrame(moon_shape, gp, "site")
        topo_frames.append(topo)
        gpoints.append(gp)

    visible = np.zeros((S, T, N), dtype=bool)
    uvec = np.zeros((S, T, N, 3), dtype=float)

    for t_idx, date in enumerate(times):
        # body-fixed -> inertial (static transform for positions at this date)
        body_to_inertial = body.getTransformTo(inertial, date).toStaticTransform()

        # User positions in inertial for each site at this time
        r_user = []
        for gp in gpoints:
            # position of site in body-fixed:
            r_bf = moon_shape.transform(gp)     # Vector3D in body frame
            # convert to inertial:
            r_i = body_to_inertial.transformPosition(r_bf)
            r_user.append(np.array([r_i.getX(), r_i.getY(), r_i.getZ()], dtype=float))
        r_user = np.stack(r_user, axis=0)  # (S,3)

        # Satellite positions in inertial at this time
        r_sat = np.zeros((N, 3), dtype=float)
        for j, sid in enumerate(sat_ids):
            pv = inertial_pvs[sid][t_idx]
            p = pv.getPosition()
            r_sat[j, :] = [p.getX(), p.getY(), p.getZ()]

        # Elevation test using topo frame (needs sat position + frame + date)
        # And unit LOS (user->sat)
        for s in range(S):
            topo = topo_frames[s]
            ru = r_user[s]
            for j, sid in enumerate(sat_ids):
                pv = inertial_pvs[sid][t_idx]
                pos = pv.getPosition()
                elev = topo.getElevation(pos, inertial, date)  # rad

                if elev >= min_elev_rad:
                    visible[s, t_idx, j] = True
                    d = r_sat[j] - ru
                    n = np.linalg.norm(d)
                    if n > 0:
                        uvec[s, t_idx, j, :] = d / n

    return visible, uvec













#All functions below this line were not modified
def evaluate_subset(subset_indices,  # list of 4 indices into sat_ids
                    visible, uvec,
                    max_cond=1e10):
    """
    visible: (S,T,N), uvec: (S,T,N,3)
    Returns per site:
      outage_frac, hdop_series, pdop_series, hdop_p95, pdop_p95
    """
    S, T, N = visible.shape
    subset_indices = np.array(subset_indices, dtype=int)

    results = []
    for s in range(S):
        hdop = np.full(T, np.nan, dtype=float)
        pdop = np.full(T, np.nan, dtype=float)

        for t in range(T):
            vis_mask = visible[s, t, subset_indices]
            if np.count_nonzero(vis_mask) < 4:
                continue
            unit_los = uvec[s, t, subset_indices, :]

            # all 4 must be visible (mask already ensures that)
            p, h = dop_from_unit_los(unit_los, max_cond=max_cond)
            pdop[t] = p
            hdop[t] = h

        valid = ~np.isnan(hdop)
        outage = 1.0 - np.mean(valid)
        if np.any(valid):
            hdop_p95 = float(np.percentile(hdop[valid], 95))
            pdop_p95 = float(np.percentile(pdop[valid], 95))
        else:
            hdop_p95 = float("inf")
            pdop_p95 = float("inf")

        results.append({
            "outage_frac": outage,
            "hdop": hdop,
            "pdop": pdop,
            "hdop_p95": hdop_p95,
            "pdop_p95": pdop_p95,
        })
    return results

def score_subset(site_results, outage_weight=1e3):
    worst_outage = max(r["outage_frac"] for r in site_results)
    worst_hdop_p95 = max(r["hdop_p95"] for r in site_results)
    return outage_weight * worst_outage + worst_hdop_p95

def dop_from_unit_los(unit_los: np.ndarray, max_cond: float = 1e10):
    """
    unit_los: (m,3) array of unit vectors from user -> sat for the sats used at this epoch.
              Must have m >= 4 for full 3D+clock solution.
    Returns: (pdop, hdop) or (np.nan, np.nan) if singular/bad geometry.
    """
    m = unit_los.shape[0]
    if m < 4:
        return np.nan, np.nan

    H = np.ones((m, 4), dtype=float)
    H[:, 0:3] = unit_los

    HT_H = H.T @ H
    if np.linalg.cond(HT_H) > max_cond:
        return np.nan, np.nan

    try:
        Q = np.linalg.inv(HT_H)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    pdop = float(np.sqrt(Q[0,0] + Q[1,1] + Q[2,2]))
    hdop = float(np.sqrt(Q[0,0] + Q[1,1]))
    return pdop, hdop

def find_best_four_sats(sat_ids, visible, uvec, top_k=20):
    """
    Returns: list of (score, subset_indices, site_results_summary)
    subset_indices are indices into sat_ids
    """
    N = len(sat_ids)
    heap = []  # max-heap via negative score

    for subset in itertools.combinations(range(N), 4):
        site_results = evaluate_subset(subset, visible, uvec)
        sc = score_subset(site_results)

        # keep top_k smallest scores
        if len(heap) < top_k:
            heapq.heappush(heap, (-sc, subset, site_results))
        else:
            if sc < -heap[0][0]:
                heapq.heapreplace(heap, (-sc, subset, site_results))

    # return sorted best
    best = [(-h[0], h[1], h[2]) for h in heap]
    best.sort(key=lambda x: x[0])
    return best
