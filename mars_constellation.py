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
    mars_inertial: Frame
    mars_fixed: Frame
    mars_shape: OneAxisEllipsoid

def build_constellation(config: ConstellationConfig) -> Constellation:
    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(2025, 1, 1, 0, 0, 0.0, utc)

    mars = CelestialBodyFactory.getMars()
    mu = mars.getGM()
    inertial = mars.getInertiallyOrientedFrame()
    fixed = mars.getBodyOrientedFrame()

    # Mars ellipsoid (IAU value)
    R_mars = 3394200.0  # meters
    f_mars = 0.00589
    mars_shape = OneAxisEllipsoid(R_mars, f_mars, fixed)

    # Reference orbit
    a = R_mars + config.altitude_km * 1e3
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
        mars_inertial=inertial,
        mars_fixed=fixed,
        mars_shape=mars_shape
    )



def addConstellation(secondConfig: ConstellationConfig, originalConst: Constellation) -> Constellation:
    """This definition takes a constellation object and adds a new Walker-Delta constellation to the 
    satellites that are already there. It is used to create the "double" constellations in the analysis 
    of how many additional satellites are required for global coverage"""
    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(2025, 1, 1, 0, 0, 0.0, utc)

    mars = CelestialBodyFactory.getMars()
    mu = mars.getGM()
    inertial = mars.getInertiallyOrientedFrame()
    fixed = mars.getBodyOrientedFrame()

    # Mars ellipsoid 
    R_mars = 3394200.0  # meters
    f_mars = 0.00589
    mars_shape = OneAxisEllipsoid(R_mars, f_mars, fixed)

    # Reference orbit
    a = R_mars + secondConfig.altitude_km * 1e3
    e = 0.0
    i = math.radians(secondConfig.inclination_deg)
    omega = 0.0
    raan0 = 0.0
    M0 = 0.0

    ref_orbit = KeplerianOrbit(a, e, i, omega, raan0, M0,
                                PositionAngleType.MEAN,
                                inertial, epoch, mu)

    # New Walker constellation
    walker = WalkerConstellation(secondConfig.total_sats, secondConfig.planes, secondConfig.phasing)
    slot_matrix = walker.buildRegularSlots(ref_orbit)

    satellites = list(originalConst.satellites)  # copy list
    #set the new satelite id to start at the max of the previous list +1
    if satellites:
        sat_id = max(s.sat_id for s in satellites) + 1
    else:
        sat_id = 0

    #iterate through and add new satellites
    for p in range(slot_matrix.size()):
        plane_slots = JavaList.cast_(slot_matrix.get(p))
        for s in range(plane_slots.size()):
            slot = WalkerConstellationSlot.cast_(plane_slots.get(s))
            orbit = slot.getOrbit()
            prop = KeplerianPropagator(orbit)
            satellites.append(Satellite(sat_id, p, s, prop))
            sat_id += 1

    return Constellation(
        config=secondConfig, #This currently works but is misleading since it only represents the second configuration
        epoch=epoch,         
        satellites=satellites, #but the satellites includes the satellites from both constellations
        mars_inertial=inertial,
        mars_fixed=fixed,
        mars_shape=mars_shape
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

            transform = constellation.mars_inertial.getTransformTo(constellation.mars_fixed, t)
            pv_fixed = transform.transformPVCoordinates(pv)
            fixed_pvs[sat.sat_id].append(pv_fixed)

    return times, inertial_pvs, fixed_pvs


def divisors(n: int) -> list[int]:
    """This is a helper function that is useful for determing the possible 
    number of orbital planes for a specific number of satellites in a Walker Delta constellation"""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)

def findMinSatsForGlobal(originalConstellation: Constellation, times: List[AbsoluteDate],planes_by_sat_count, maxSats:int = 21, i:float = 75)-> Tuple[int, ConstellationConfig]:
    """Returns both the minimum number of additional satellites for global coverage and the added constellation config in a Tuple. 
    Assumes the same altitude as the original constellation and an inclination of 75 as a default unless only 1 plane in which case i = 90
        The list of planes per satellites number and the times are passed in so they aern't recomputed every time for effiency"""
    #duration_sec = 24 * 3600      # 24 hours
    #step_sec = 300*6               # 30 min
    #alt = originalConstellation.config.altitude_km
    #times, inertial_pvs, fixed_pvs = propagate(
    #                            originalConstellation,
    #                            duration_sec=duration_sec,
    #                            step_sec=step_sec)
    #check no additional sats

    #I think its faster to just check high and low latitudes rather than globally at once.
    if (constellation_meets_5sat_requirement(originalConstellation, lat_min_deg=45.0, lat_max_deg=90.0) 
        and constellation_meets_5sat_requirement(originalConstellation, lat_min_deg=-90.0, lat_max_deg=-45.0) 
        and pdop_p95_requirement_met(originalConstellation, times, 45, 90) 
        and pdop_p95_requirement_met(originalConstellation, times, -90, -45)):
        return 0, None
    
    # precompute possible planes so it isn't done every iteration
  
    attemptCount = 0
    #generate range of sat numbers
    Trange = range(3, maxSats+1)
    for T in Trange:
        #get correct list of planes from precomputed planes
        planes = planes_by_sat_count.get(T)
        if not planes:
            continue
        for numPlanes in planes:
            #if only 1 plane, i = 90. This seems to never win but ill leave it in
            if numPlanes ==1:
                curr_i = 90
            else:
                curr_i = i
            for f in range(0, (numPlanes)):
                attemptCount = attemptCount+1
                #make the new additional configuration that will be evaluated against the requirements
                #assumes same altitude as original
                additionalcfg = ConstellationConfig(
                                name = str("c2nd"+(str(attemptCount))),
                                inclination_deg=float(curr_i),
                                total_sats=int(T),
                                planes=int(numPlanes),
                                phasing=int(f),
                                altitude_km=float(originalConstellation.config.altitude_km),
                                pattern=str('DELTA'))
                #creates the couble constellation
                doubleConstellation = addConstellation(additionalcfg, originalConstellation)
                

                if (constellation_meets_5sat_requirement(doubleConstellation, lat_min_deg=45.0, lat_max_deg=90.0) 
                    and constellation_meets_5sat_requirement(doubleConstellation, lat_min_deg=-90.0, lat_max_deg=-45.0) 
                    and pdop_p95_requirement_met(doubleConstellation, times, 45, 90) 
                    and pdop_p95_requirement_met(doubleConstellation, times, -90, -45)):
                    return T, additionalcfg
    #if it fails to find something
    return -1, None
        
def compute_ground_tracks(constellation,
                          times,  # List[AbsoluteDate]
                          fixed_pvs: Dict[int, List[PVCoordinates]]):
    """
    Convert Mars-fixed PVCoordinates into lat/lon ground tracks (degrees).

    times: list[AbsoluteDate] used when generating fixed_pvs (same length as each pv_list)
    fixed_pvs: dict[sat_id] -> list[PVCoordinates in Mars-fixed frame]

    Returns:
        tracks: dict[sat_id] -> {"lat": [...], "lon": [...]}
    """
    tracks = {}
    mars_shape = constellation.mars_shape
    body_frame = constellation.mars_fixed

    for sat_id, pv_list in fixed_pvs.items():
        if len(pv_list) != len(times):
            raise ValueError(f"Length mismatch for sat {sat_id}: "
                             f"{len(pv_list)} PVs vs {len(times)} times")

        lats = []
        lons = []
        for t, pv in zip(times, pv_list):
            # pv is in Mars-fixed frame already; we still must pass the date
            geodetic = mars_shape.transform(pv.getPosition(), body_frame, t)

            lat_deg = math.degrees(geodetic.getLatitude())
            lon_deg = math.degrees(geodetic.getLongitude())

            # Normalize longitude to [-180, 180] to match basemap
            if lon_deg > 180.0:
                lon_deg -= 360.0
            elif lon_deg < -180.0:
                lon_deg += 360.0

            lats.append(lat_deg)
            lons.append(lon_deg)

        tracks[sat_id] = {"lat": lats, "lon": lons}

    return tracks

def compute_pdop_p95_map(constellation,
                         times: List,          # List[AbsoluteDate]
                         lat_min_deg: float = -45.0,
                         lat_max_deg: float = 45.0,
                         lat_step_deg: float = 5.0,
                         lon_min_deg: float = -180.0,
                         lon_max_deg: float = 180.0,
                         lon_step_deg: float = 10.0,
                         min_elev_deg: float = 5.0
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PDOP time series and P95 PDOP for a lat/lon grid.

    Returns:
        lat_vals : 1D array of latitudes (deg)
        lon_vals : 1D array of longitudes (deg)
        p95_pdop : 2D array [n_lat, n_lon] with PDOP 95th percentile at each grid point
                   (NaN where PDOP cannot be computed / insufficient coverage)
    """
    mars_shape = constellation.mars_shape

    # Build Java list of propagators once
    gnss_list = ArrayList()
    #print(type(gnss_list))
    for sat in constellation.satellites:
        gnss_list.add(sat.propagator)

    lat_vals = np.arange(lat_min_deg, lat_max_deg + 1e-6, lat_step_deg)
    lon_vals = np.arange(lon_min_deg, lon_max_deg + 1e-6, lon_step_deg)

    n_lat = lat_vals.size
    n_lon = lon_vals.size
    n_t   = len(times)

    # Store PDOP time series per grid point
    # shape: (n_lat, n_lon, n_t)
    pdop_series = np.full((n_lat, n_lon, n_t), np.nan, dtype=float)

    # Pre-create DOPComputers for each grid point
    dop_computers: Dict[Tuple[int, int], DOPComputer] = {}


    min_elev_rad = math.radians(min_elev_deg)

    for i_lat, lat_deg in enumerate(lat_vals):
        lat_rad = math.radians(lat_deg)
        for j_lon, lon_deg in enumerate(lon_vals):
            lon_rad = math.radians(lon_deg)
            gp = GeodeticPoint(lat_rad, lon_rad, 0.0)  # altitude 0 for now
            comp = DOPComputer.create(mars_shape, gp).withMinElevation(min_elev_rad)
            dop_computers[(i_lat, j_lon)] = comp

    # Time loop
    for k, date in enumerate(times):
        # For each grid point, compute PDOP at this date
        for (i_lat, j_lon), comp in dop_computers.items():
            try:
                dop = comp.compute(date, gnss_list)
                pdop = dop.getPdop()
            # If fewer than 4 visible sats, PDOP will be NaN; we keep NaN
            except orekit.JavaError:
                #singular geometry matrix
                pdop = float('nan')
            pdop_series[i_lat, j_lon, k] = pdop

    # Now compute P95 along the time axis, ignoring NaNs
    p95_pdop = np.full((n_lat, n_lon), np.nan, dtype=float)
    for i_lat in range(n_lat):
        for j_lon in range(n_lon):
            series = pdop_series[i_lat, j_lon, :]
            # Drop NaNs (times with < 4 visible sats)
            valid = series[~np.isnan(series)]
            if valid.size == 0:
                p95 = np.nan
            else:
                p95 = np.percentile(valid, 95)
            p95_pdop[i_lat, j_lon] = p95

    return lat_vals, lon_vals, p95_pdop

def pdop_p95_requirement_met(constellation: Constellation,
                             times: List,
                             lat_min_deg: float=-45.0,
                             lat_max_deg: float=45.0,
                             lat_step_deg: float=5.0,
                             lon_min_deg: float=-180.0,
                             lon_max_deg: float=180.0,
                             lon_step_deg: float=10.0,
                             min_elev_deg: float=5.0,
                             pdop_limit: float=6.0)->bool:
    """
    Fast early-exit PDOP P95 check.
    Returns True if EVERY grid point has P95 PDOP <= pdop_limit.
    Returns False immediately when a violation is detected.
    """

    mars_shape = constellation.mars_shape

    # Build Java list of propagators once
    gnss_list = ArrayList()
    for sat in constellation.satellites:
        gnss_list.add(sat.propagator)

    lat_vals = np.arange(lat_min_deg, lat_max_deg + 1e-6, lat_step_deg)
    lon_vals = np.arange(lon_min_deg, lon_max_deg + 1e-6, lon_step_deg)

    min_elev_rad = math.radians(min_elev_deg)

    #pre-create DOPComputers for each grid point
    dop_computers = {}
    pdop_records = {}   # store list of PDOP values for each point

    for i_lat, lat_deg in enumerate(lat_vals):
        lat_rad = math.radians(lat_deg)
        for j_lon, lon_deg in enumerate(lon_vals):
            lon_rad = math.radians(lon_deg)
            gp = GeodeticPoint(lat_rad, lon_rad, 0.0)
            comp = DOPComputer.create(mars_shape, gp).withMinElevation(min_elev_rad)
            dop_computers[(i_lat, j_lon)] = comp
            pdop_records[(i_lat, j_lon)] = []

    # Time loop with early exit
    for date in times:
        for key, comp in dop_computers.items():

            # catch singular-matrix cases that was causing issues (is there an underlying 
            # cause that needs to be addressed rather than just catching the error?)
            try:
                dop = comp.compute(date, gnss_list)
                pdop_val = dop.getPdop()
            except orekit.JavaError:
                pdop_val = float('nan')

            if not math.isnan(pdop_val):
                rec = pdop_records[key]
                rec.append(pdop_val)

                # Early check on P95 only when we have enough samples
                n = len(rec)
                if n >= 20:  # avoid unstable early percentiles
                    sorted_vals = sorted(rec)
                    #there may be a better way to do this part... not sure what about the other PDOP function makes it P95
                    idx95 = int(0.95 * (n - 1))
                    if sorted_vals[idx95] > pdop_limit:
                        return False  # EARLY EXIT

    # Final pass over all grid points
    for key, values in pdop_records.items():
        if len(values) == 0:
            return False  # no coverage requirement fails

        p95 = np.percentile(values, 95)
        if p95 > pdop_limit:
            return False

    return True  # requirement met



def constellation_meets_5sat_requirement( 
    constellation,
    start_date: Optional[AbsoluteDate] = None,
    lat_min_deg: float = -45.0,
    lat_max_deg: float = 45.0,
    lat_step_deg: float = 5.0,
    lon_min_deg: float = -180.0,
    lon_max_deg: float = 180.0,
    lon_step_deg: float = 10.0,
    min_elev_deg: float = 10.0,
    min_sats_in_view: int = 5,
    time_step_sec: float = 300,
) -> bool:

    if start_date is None:
        utc = TimeScalesFactory.getUTC()
        start_date = AbsoluteDate(2025, 1, 1, 0, 0, 0.0, utc)

    mars_shape = constellation.mars_shape

    propagators = [sat.propagator for sat in constellation.satellites]

    lat_vals = np.arange(lat_min_deg, lat_max_deg + 1e-6, lat_step_deg)
    lon_vals = np.arange(lon_min_deg, lon_max_deg + 1e-6, lon_step_deg)

    min_elev_rad = math.radians(min_elev_deg)

    topo_frames = {}
    #get frames for lat long grid
    for i_lat, lat_deg in enumerate(lat_vals):
        lat_rad = math.radians(lat_deg)
        for j_lon, lon_deg in enumerate(lon_vals):
            lon_rad = math.radians(lon_deg)
            gp = GeodeticPoint(lat_rad, lon_rad, 0.0)
            topo_frames[(i_lat, j_lon)] = TopocentricFrame(mars_shape, gp, f"pt_{i_lat}_{j_lon}")

    mars_sidereal_day_sec = 88642.663
    n_steps = int(math.ceil(mars_sidereal_day_sec / time_step_sec))

    #time loop
    for k in range(n_steps):
        date = start_date.shiftedBy(float(k * time_step_sec))

        states = [prop.propagate(date) for prop in propagators]

        for (i_lat, j_lon), topo in topo_frames.items():
            visible_count = 0
            for state in states:
                pos = state.getPVCoordinates().getPosition()
                frame = state.getFrame()
                elev = topo.getElevation(pos, frame, date)

                if elev >= min_elev_rad:
                    visible_count += 1
                    if visible_count >= min_sats_in_view:
                        break

            if visible_count < min_sats_in_view:
                return False

    return True #requirement met

