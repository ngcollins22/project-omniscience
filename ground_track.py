"""
Ground-track utilities for the OMNIScience gantry experiment.

Workflow
--------
1. Physically calibrate the map::

       cal = MapCalibration(corner_br, corner_tl, region)

2. Build or load a ground track::

       track = generate_synthetic_pass(region, duration_s=120, n_points=50)
       track = load_csv("results/ground_tracks/pass.csv")

3. Execute on the gantry::

       runner = TrackRunner(cal, track, gantry_worker,
                            speed_mm_min=3000, dry_run=True,
                            log_cb=print)
       runner.start()
       ...
       runner.stop()
"""

from __future__ import annotations

import csv
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Waypoint:
    """A single ground-track waypoint."""
    t_s:     float   # elapsed simulation time (seconds)
    lat_deg: float   # geodetic latitude  (degrees)
    lon_deg: float   # geographic longitude (degrees)


@dataclass
class GroundTrack:
    """Ordered list of ground-track waypoints."""
    waypoints: List[Waypoint] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.waypoints)

    def path_length_deg(self) -> float:
        """Cumulative arc length in degrees (flat-Earth approximation)."""
        total = 0.0
        for i in range(1, len(self.waypoints)):
            a, b = self.waypoints[i - 1], self.waypoints[i]
            dlat = b.lat_deg - a.lat_deg
            dlon = b.lon_deg - a.lon_deg
            total += math.sqrt(dlat ** 2 + dlon ** 2)
        return total


# ── Map calibration ───────────────────────────────────────────────────────────

@dataclass
class DemRegionBounds:
    """Lat/lon bounding box — thin wrapper so we don't depend on config types."""
    lat_min_deg: float
    lat_max_deg: float
    lon_min_deg: float
    lon_max_deg: float


@dataclass
class MapCalibration:
    """
    Stores two physically recorded gantry positions that bracket the print area
    and provides lat/lon → gantry-mm conversion.

    Axis convention (OMNIScience print orientation)
    -----------------------------------------------
    Gantry X  → latitude  (N-S, smaller X = north in this layout)
    Gantry Y  → longitude (E-W)

    Corners (recorded with ArduCam centre aligned to map edge)
    ----------------------------------------------------------
    corner_br : gantry (x, y) mm  when camera centre is at map bottom-right
                = (lat_min, lon_max) in dem.region
    corner_tl : gantry (x, y) mm  when camera centre is at map top-left
                = (lat_max, lon_min) in dem.region
    """
    corner_br: Tuple[float, float]   # (x_mm, y_mm)
    corner_tl: Tuple[float, float]   # (x_mm, y_mm)
    region:    DemRegionBounds

    def to_mm(self, lat_deg: float, lon_deg: float) -> Tuple[float, float]:
        """
        Convert a (lat, lon) coordinate to gantry (x_mm, y_mm).

        Aspect-ratio-preserving fit
        ---------------------------
        The ground track's lat/lon bounding box is scaled uniformly so that
        the wider dimension (in degrees-per-mm) fills its map axis completely.
        The narrower dimension is centered within the remaining space.

        Axis mapping (OMNIScience print orientation):
            Gantry X: corner_tl → corner_br  maps  lat_max → lat_min  (N to S)
            Gantry Y: corner_tl → corner_br  maps  lon_min → lon_max  (W to E)
        """
        r = self.region

        lat_span = r.lat_max_deg - r.lat_min_deg   # total degrees, positive
        lon_span = r.lon_max_deg - r.lon_min_deg   # total degrees, positive

        map_x_span = self.corner_br[0] - self.corner_tl[0]   # signed mm (e.g. -378)
        map_y_span = self.corner_br[1] - self.corner_tl[1]   # signed mm (e.g. -789)

        # Uniform scale: pick whichever axis is the binding constraint
        # so the shape is preserved (no stretching)
        if lat_span > 0 and lon_span > 0:
            scale_x = abs(map_x_span) / lat_span   # mm/deg if lat fills X
            scale_y = abs(map_y_span) / lon_span   # mm/deg if lon fills Y
            scale   = min(scale_x, scale_y)
        else:
            scale = 1.0

        # Fractional position within the bounding box
        # lat: 0.0 = lat_max (north, TL side), 1.0 = lat_min (south, BR side)
        lat_frac = (r.lat_max_deg - lat_deg) / lat_span if lat_span else 0.5
        # lon: 0.0 = lon_min (west, TL side), 1.0 = lon_max (east, BR side)
        lon_frac = (lon_deg - r.lon_min_deg) / lon_span if lon_span else 0.5

        # Offset from centre of the map box in mm
        # (frac - 0.5) goes -0.5..+0.5; multiply by scale*span to get mm
        lat_offset_mm = (lat_frac - 0.5) * lat_span * scale
        lon_offset_mm = (lon_frac - 0.5) * lon_span * scale

        # Centre of the gantry map in mm
        cx = (self.corner_tl[0] + self.corner_br[0]) / 2.0
        cy = (self.corner_tl[1] + self.corner_br[1]) / 2.0

        # X increases as lat_frac increases (TL→BR = north→south)
        # Y increases as lon_frac increases (TL→BR = west→east)
        # map_x_span is negative (-378), so we go negative as lat_frac grows
        x_mm = cx + lat_offset_mm * (1 if map_x_span >= 0 else -1)
        y_mm = cy + lon_offset_mm * (1 if map_y_span >= 0 else -1)

        # Clamp to map boundaries
        x_lo, x_hi = sorted([self.corner_tl[0], self.corner_br[0]])
        y_lo, y_hi = sorted([self.corner_tl[1], self.corner_br[1]])
        x_mm = max(x_lo, min(x_hi, x_mm))
        y_mm = max(y_lo, min(y_hi, y_mm))

        return x_mm, y_mm

    def to_lat_lon(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """
        Convert a gantry (x_mm, y_mm) position to (lat_deg, lon_deg).

        This is the inverse of to_mm().  The result is NOT clamped — positions
        outside the calibrated map corners will return lat/lon values outside
        the region bounds.

        Returns (nan, nan) if the region spans zero degrees in either axis.
        """
        r = self.region

        lat_span = r.lat_max_deg - r.lat_min_deg
        lon_span = r.lon_max_deg - r.lon_min_deg
        if lat_span == 0.0 or lon_span == 0.0:
            import math
            return math.nan, math.nan

        map_x_span = self.corner_br[0] - self.corner_tl[0]
        map_y_span = self.corner_br[1] - self.corner_tl[1]

        scale_x = abs(map_x_span) / lat_span
        scale_y = abs(map_y_span) / lon_span
        scale   = min(scale_x, scale_y)

        cx = (self.corner_tl[0] + self.corner_br[0]) / 2.0
        cy = (self.corner_tl[1] + self.corner_br[1]) / 2.0

        sign_x = 1.0 if map_x_span >= 0 else -1.0
        sign_y = 1.0 if map_y_span >= 0 else -1.0

        lat_offset_mm = (x_mm - cx) * sign_x
        lon_offset_mm = (y_mm - cy) * sign_y

        lat_frac = lat_offset_mm / (lat_span * scale) + 0.5
        lon_frac = lon_offset_mm / (lon_span * scale) + 0.5

        lat_deg = r.lat_max_deg - lat_frac * lat_span
        lon_deg = r.lon_min_deg + lon_frac * lon_span

        return lat_deg, lon_deg

    def map_size_mm(self) -> Tuple[float, float]:
        """Return (width_mm, height_mm) of the calibrated map area."""
        dx = abs(self.corner_br[0] - self.corner_tl[0])
        dy = abs(self.corner_br[1] - self.corner_tl[1])
        return dx, dy


# ── Ground-track generators ───────────────────────────────────────────────────

def generate_synthetic_pass(
    region:     DemRegionBounds,
    duration_s: float = 120.0,
    n_points:   int   = 50,
    n_cycles:   float = 1.5,
) -> GroundTrack:
    """
    Generate a sinusoidal orbital ground track over the DEM region.

    Longitude sweeps linearly from lon_min → lon_max (primary direction).
    Latitude oscillates as a sine wave around the region centre, mimicking
    the appearance of a real orbital ground track on a Mercator projection.

        lat(t) = lat_centre + lat_amplitude * sin(2π · n_cycles · t/T)
        lon(t) = lon_min + (t/T) · (lon_max − lon_min)

    Parameters
    ----------
    region     : lat/lon bounding box (from cfg.dem.region)
    duration_s : total simulated pass duration in seconds
    n_points   : number of evenly spaced waypoints
    n_cycles   : number of full sine oscillations across the pass (default 1.5)
    """
    lat_centre = (region.lat_max_deg + region.lat_min_deg) / 2.0
    lat_amp    = (region.lat_max_deg - region.lat_min_deg) / 2.0

    waypoints = []
    for i in range(n_points):
        frac = i / max(n_points - 1, 1)
        lat  = lat_centre + lat_amp * math.sin(2.0 * math.pi * n_cycles * frac)
        lon  = region.lon_min_deg + frac * (region.lon_max_deg - region.lon_min_deg)
        t_s  = frac * duration_s
        waypoints.append(Waypoint(t_s=t_s, lat_deg=lat, lon_deg=lon))
    return GroundTrack(waypoints=waypoints)


def generate_synthetic_pass_normalized(
    n_points:   int   = 50,
    duration_s: float = 120.0,
    n_cycles:   float = 1.5,
) -> GroundTrack:
    """
    Fallback for when dem.region is not configured.

    Generates a sinusoidal pass in normalised [0, 1] lat/lon space.
    Useful for dry-run motion testing when the physical region is unknown.
    """
    dummy = DemRegionBounds(
        lat_min_deg=0.0, lat_max_deg=1.0,
        lon_min_deg=0.0, lon_max_deg=1.0,
    )
    return generate_synthetic_pass(dummy, duration_s=duration_s,
                                   n_points=n_points, n_cycles=n_cycles)


def load_csv(path: str) -> GroundTrack:
    """
    Load a ground track from a CSV file.

    Required columns (header row):
        time_s    – elapsed simulation time in seconds
        lat_deg   – geodetic latitude in degrees
        lon_deg   – geographic longitude in degrees

    The delimiter is detected automatically — tab-separated and
    comma-separated files both work. Column order does not matter.
    Any additional columns are silently ignored.
    """
    waypoints = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        # ── Auto-detect delimiter (handles tab and comma) ─────────────────
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel   # fall back to standard comma CSV

        reader = csv.DictReader(fh, dialect=dialect)

        # ── Strip whitespace and drop empty Excel ghost columns ───────────
        # utf-8-sig handles the BOM; strip() catches spaces; the filter
        # drops the empty trailing columns Excel adds when saving .csv
        reader.fieldnames = (
            [name.strip() for name in reader.fieldnames if name.strip()]
            if reader.fieldnames else []
        )

        missing = {"time_s", "lat_deg", "lon_deg"} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}. "
                f"Columns found: {reader.fieldnames}"
            )

        for row in reader:
            waypoints.append(Waypoint(
                t_s=float(row["time_s"]),
                lat_deg=float(row["lat_deg"]),
                lon_deg=float(row["lon_deg"]),
            ))

    waypoints.sort(key=lambda w: w.t_s)
    return GroundTrack(waypoints=waypoints)


# ── Track runner ──────────────────────────────────────────────────────────────

class TrackRunner(threading.Thread):
    """
    Daemon thread that drives the gantry through a ground track.

    At each waypoint the runner:
        1. Converts (lat, lon) → (x_mm, y_mm) via MapCalibration.to_mm()
        2. Issues a goto() command to the gantry (or logs a dry-run line)
        3. Polls GRBL state via worker.get_position() until the state is Idle

    Between waypoints the thread sleeps for the real-time equivalent of the
    inter-waypoint simulation interval divided by ``time_scale_factor``.
    """

    IDLE_POLL_S = 0.05   # how often to poll GRBL state while waiting for Idle

    def __init__(
        self,
        calibration:       MapCalibration,
        track:             GroundTrack,
        gantry_worker,                        # GantryWorker instance (or None in dry-run)
        speed_mm_min:      float = 3000.0,
        time_scale_factor: float = 1.0,
        dry_run:           bool  = True,
        log_cb:            Optional[Callable[[str], None]] = None,
    ):
        super().__init__(daemon=True, name="TrackRunner")
        self._cal    = calibration
        self._track  = track
        self._worker = gantry_worker
        self._speed  = speed_mm_min
        self._tscale = time_scale_factor
        self._dry    = dry_run
        self._log    = log_cb or (lambda msg: None)
        self._stop   = threading.Event()

    def stop(self) -> None:
        """Signal the runner to abort after the current waypoint."""
        self._stop.set()

    def run(self) -> None:
        wps = self._track.waypoints
        if not wps:
            self._log("[WARN ] TrackRunner: empty track — nothing to do.")
            return

        mode = "DRY RUN" if self._dry else "LIVE"
        self._log(
            f"[INFO ] TrackRunner: {mode} — {len(wps)} waypoints, "
            f"speed {self._speed:.0f} mm/min, time scale ×{self._tscale}"
        )

        for idx, wp in enumerate(wps):
            if self._stop.is_set():
                self._log("[INFO ] TrackRunner: stop requested.")
                break

            x_mm, y_mm = self._cal.to_mm(wp.lat_deg, wp.lon_deg)

            if self._dry:
                self._log(
                    f"[DRY  ] wp {idx + 1}/{len(wps)}  "
                    f"lat={wp.lat_deg:.4f}°  lon={wp.lon_deg:.4f}°  "
                    f"→ X={x_mm:.2f}  Y={y_mm:.2f} mm"
                )
            else:
                self._worker.goto(x=x_mm, y=y_mm, feed_mm_min=self._speed, rapid=False)
                self._log(
                    f"[INFO ] wp {idx + 1}/{len(wps)}  "
                    f"lat={wp.lat_deg:.4f}°  lon={wp.lon_deg:.4f}°  "
                    f"→ X={x_mm:.2f}  Y={y_mm:.2f} mm"
                )
                self._wait_idle()

            # Time-scaled inter-waypoint delay
            if idx + 1 < len(wps):
                sim_dt  = wps[idx + 1].t_s - wp.t_s
                real_dt = sim_dt / max(self._tscale, 1e-6)
                deadline = time.monotonic() + real_dt
                while time.monotonic() < deadline and not self._stop.is_set():
                    time.sleep(min(0.05, deadline - time.monotonic()))

        if not self._stop.is_set():
            self._log("[OK   ] TrackRunner: track complete.")
        self._log("[INFO ] TrackRunner: exiting.")

    def _wait_idle(self) -> None:
        """Block until GRBL reports Idle (or stop is requested)."""
        while not self._stop.is_set():
            _, state = self._worker.get_position()
            if state == "Idle":
                return
            time.sleep(self.IDLE_POLL_S)