"""
Experiment configuration loader.

Usage anywhere in the project:

    from config import load_config
    cfg = load_config()            # loads experiment_config.yml from repo root
    print(cfg.realsense.fov_x_deg)
    print(cfg.gantry.camera_altitude_mm)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import yaml

# Resolve the repo root relative to this file so imports work regardless of
# the calling script's working directory.
_REPO_ROOT    = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "experiment_config.yml")


# ── Sub-configs ───────────────────────────────────────────────────────────────

@dataclass
class PathsConfig:
    dem_geotiff:       Optional[str]
    results_dir:       str
    scans_dir:         str
    ground_tracks_dir: str


@dataclass
class DemRegion:
    lat_min_deg: Optional[float]
    lat_max_deg: Optional[float]
    lon_min_deg: Optional[float]
    lon_max_deg: Optional[float]


@dataclass
class DemUncertainties:
    horizontal_position_m: float
    radius_m:              float
    elevation_m:           float
    areoid_m:              float


@dataclass
class DemConfig:
    meters_per_pixel: float
    region:           DemRegion
    uncertainties:    DemUncertainties


@dataclass
class PrintConfig:
    width_mm:             float
    height_mm:            float
    base_thickness_mm:    Optional[float]
    vertical_exaggeration: Optional[float]


@dataclass
class GantryConfig:
    serial_port:         Optional[str]
    baud_rate:           Optional[int]
    travel_x_mm:         Optional[float]
    travel_y_mm:         Optional[float]
    travel_z_mm:         Optional[float]
    max_speed_mm_s:      Optional[float]
    camera_altitude_mm:  float
    heading_deg:         float
    rs_offset_x_mm:      float
    rs_offset_y_mm:      float
    rs_offset_z_mm:      float


@dataclass
class RealSenseSensorConfig:
    resolution_x:          int
    resolution_y:          int
    fps:                   int
    fov_x_deg:             float
    fov_y_deg:             float
    min_range_mm:          Optional[float]
    max_range_mm:          float
    enable_emitter:        bool
    emitter_power:         int
    exposure_us:           int
    range_noise_sigma_mm:  Optional[float]
    dropout_probability:   Optional[float]
    systematic_bias_mm:    float
    output_grid_res_mm:    float
    smoothing_sigma_mm:    float


@dataclass
class ArducamConfig:
    camera_index: int
    resolution_x: Optional[int]
    resolution_y: Optional[int]
    fov_x_deg:    Optional[float]
    fov_y_deg:    Optional[float]
    offset_x_mm:  Optional[float]
    offset_y_mm:  Optional[float]
    offset_z_mm:  Optional[float]
    noise_sigma:  Optional[float]


@dataclass
class SimulationConfig:
    random_seed:            Optional[int]
    measurement_spacing_mm: Optional[float]
    time_scale_factor:      Optional[float]
    n_particles:            Optional[int]
    process_noise_x_mm:     Optional[float]
    process_noise_y_mm:     Optional[float]
    initial_spread_x_mm:    Optional[float]
    initial_spread_y_mm:    Optional[float]


# ── Top-level config ──────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    paths:      PathsConfig
    dem:        DemConfig
    print_:     PrintConfig      # trailing underscore avoids shadowing built-in
    gantry:     GantryConfig
    realsense:  RealSenseSensorConfig
    arducam:    ArducamConfig
    simulation: SimulationConfig

    @classmethod
    def from_yaml(cls, path: str = _DEFAULT_CONFIG) -> ExperimentConfig:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        # Replace any bare '??' strings (unfilled placeholders) with None
        raw = _nullify(raw)

        p   = raw["paths"]
        d   = raw["dem"]
        pr  = raw["print"]
        g   = raw["gantry"]
        rs  = raw["realsense_sensor"]
        arc = raw["arducam"]
        sim = raw["simulation"]

        return cls(
            paths=PathsConfig(
                dem_geotiff=p["dem_geotiff"],
                results_dir=p["results_dir"],
                scans_dir=p["scans_dir"],
                ground_tracks_dir=p["ground_tracks_dir"],
            ),
            dem=DemConfig(
                meters_per_pixel=d["meters_per_pixel"],
                region=DemRegion(
                    lat_min_deg=d["region"]["lat_min_deg"],
                    lat_max_deg=d["region"]["lat_max_deg"],
                    lon_min_deg=d["region"]["lon_min_deg"],
                    lon_max_deg=d["region"]["lon_max_deg"],
                ),
                uncertainties=DemUncertainties(
                    horizontal_position_m=d["uncertainties"]["horizontal_position_m"],
                    radius_m=d["uncertainties"]["radius_m"],
                    elevation_m=d["uncertainties"]["elevation_m"],
                    areoid_m=d["uncertainties"]["areoid_m"],
                ),
            ),
            print_=PrintConfig(
                width_mm=pr["width_mm"],
                height_mm=pr["height_mm"],
                base_thickness_mm=pr["base_thickness_mm"],
                vertical_exaggeration=pr["vertical_exaggeration"],
            ),
            gantry=GantryConfig(
                serial_port=g["serial_port"],
                baud_rate=g["baud_rate"],
                travel_x_mm=g["travel_x_mm"],
                travel_y_mm=g["travel_y_mm"],
                travel_z_mm=g["travel_z_mm"],
                max_speed_mm_s=g["max_speed_mm_s"],
                camera_altitude_mm=g["camera_altitude_mm"],
                heading_deg=g["heading_deg"],
                rs_offset_x_mm=g["rs_offset_x_mm"],
                rs_offset_y_mm=g["rs_offset_y_mm"],
                rs_offset_z_mm=g["rs_offset_z_mm"],
            ),
            realsense=RealSenseSensorConfig(
                resolution_x=rs["resolution_x"],
                resolution_y=rs["resolution_y"],
                fps=rs["fps"],
                fov_x_deg=rs["fov_x_deg"],
                fov_y_deg=rs["fov_y_deg"],
                min_range_mm=rs["min_range_mm"],
                max_range_mm=rs["max_range_mm"],
                enable_emitter=rs["enable_emitter"],
                emitter_power=rs["emitter_power"],
                exposure_us=rs["exposure_us"],
                range_noise_sigma_mm=rs["range_noise_sigma_mm"],
                dropout_probability=rs["dropout_probability"],
                systematic_bias_mm=rs["systematic_bias_mm"],
                output_grid_res_mm=rs["output_grid_res_mm"],
                smoothing_sigma_mm=rs["smoothing_sigma_mm"],
            ),
            arducam=ArducamConfig(
                camera_index=arc["camera_index"],
                resolution_x=arc["resolution_x"],
                resolution_y=arc["resolution_y"],
                fov_x_deg=arc["fov_x_deg"],
                fov_y_deg=arc["fov_y_deg"],
                offset_x_mm=arc["offset_x_mm"],
                offset_y_mm=arc["offset_y_mm"],
                offset_z_mm=arc["offset_z_mm"],
                noise_sigma=arc["noise_sigma"],
            ),
            simulation=SimulationConfig(
                random_seed=sim["random_seed"],
                measurement_spacing_mm=sim["measurement_spacing_mm"],
                time_scale_factor=sim["time_scale_factor"],
                n_particles=sim["n_particles"],
                process_noise_x_mm=sim["process_noise_x_mm"],
                process_noise_y_mm=sim["process_noise_y_mm"],
                initial_spread_x_mm=sim["initial_spread_x_mm"],
                initial_spread_y_mm=sim["initial_spread_y_mm"],
            ),
        )


# ── Cached loader ─────────────────────────────────────────────────────────────

_cache: Optional[ExperimentConfig] = None


def load_config(path: str = _DEFAULT_CONFIG, reload: bool = False) -> ExperimentConfig:
    """
    Load and return the experiment config, caching after the first call.

    Parameters
    ----------
    path   : path to the yml file (defaults to experiment_config.yml in repo root)
    reload : force re-read from disk even if already cached
    """
    global _cache
    if _cache is None or reload:
        _cache = ExperimentConfig.from_yaml(path)
    return _cache


# ── Helpers ───────────────────────────────────────────────────────────────────

def _nullify(obj):
    """Recursively replace the string '??' with None throughout a parsed yaml tree."""
    if isinstance(obj, dict):
        return {k: _nullify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nullify(v) for v in obj]
    if obj == "??":
        return None
    return obj


if __name__ == "__main__":
    cfg = load_config()
    print(cfg)
