"""
Tau LiDAR camera projection model.

Converts Tau camera frames (or synthetic depth maps produced by the simulator)
into gridded elevation patches in the print/terrain coordinate frame.

Geometry matches TauLidarCommon FrameBuilder exactly, so that simulated patches
and real-camera patches go through the same math:

    gamma_h(x) = alpha_H + x * (theta_H / W)
    gamma_v(y) = alpha_V + y * (theta_V / H)

    where  alpha_H = (pi - theta_H) / 2
           alpha_V = 2*pi - theta_V / 2

    Z[y,x] = distance_mm * 0.001 * |sin(gamma_h[x])| * |cos(gamma_v[y])|  [m]
    X[y,x] = Z[y,x] / tan(gamma_h[x])                                      [m]
    Y[y,x] = -Z[y,x] * tan(gamma_v[y])                                     [m]

frame.data_depth is the pre-computed Z map (float32, metres, shape H×W).
Invalid/low-amplitude pixels have Z = NaN.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy import ndimage, stats

# Make the repo root importable regardless of where Python is invoked from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import TauSensorConfig


class TauProjectionModel:
    """
    Projects Tau LiDAR depth data to a gridded elevation patch.

    Parameters
    ----------
    cfg          : TauSensorConfig loaded from experiment_config.yml
    altitude_mm  : default camera altitude above the terrain reference plane (mm).
                   Can be overridden on each call to project_depth / project.
    """

    def __init__(self, cfg: TauSensorConfig, altitude_mm: float):
        self.cfg = cfg
        self.default_altitude_mm = altitude_mm

        W = cfg.resolution_x   # 160
        H = cfg.resolution_y   # 60
        self.W = W
        self.H = H

        theta_h = math.radians(cfg.fov_x_deg)          # 80° → 1.3963 rad
        theta_v = math.radians(cfg.fov_y_deg)          # 30° → 0.5236 rad
        alpha_h = (math.pi - theta_h) / 2.0            # 50° → 0.8727 rad
        alpha_v = 2.0 * math.pi - theta_v / 2.0        # 345° → 6.0214 rad

        x_idx = np.arange(W, dtype=np.float64)
        y_idx = np.arange(H, dtype=np.float64)

        gamma_h = alpha_h + x_idx * (theta_h / W)      # shape (W,) : 50° … 129.5°
        gamma_v = alpha_v + y_idx * (theta_v / H)      # shape (H,) : 345° … 374.5°

        # ── Precomputed factor arrays ────────────────────────────────────────
        # Z[y,x] = distance_mm * 0.001 * sin_h[x] * cos_v[y]
        sin_h = np.abs(np.sin(gamma_h))                # (W,) all > 0 in [50°,130°]
        cos_v = np.abs(np.cos(gamma_v))                # (H,) all ≈ 0.97 in [345°,375°]

        # X[y,x] = Z[y,x] / tan(gamma_h[x])
        # At x ≈ 79.5 gamma_h ≈ 90°, tan → ∞, X → 0 (nadir).  Use safe reciprocal.
        tan_h = np.tan(gamma_h)                        # (W,)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_factor = np.where(np.abs(tan_h) > 1e-9, 1.0 / tan_h, 0.0)  # (W,)

        # Y[y,x] = -Z[y,x] * tan(gamma_v[y])
        y_factor = -np.tan(gamma_v)                    # (H,)

        # Store as (H,W) broadcasted arrays for fast vectorised projection
        self._X_factor = x_factor[np.newaxis, :]       # (1, W) → broadcasts over H
        self._Y_factor = y_factor[:, np.newaxis]        # (H, 1) → broadcasts over W
        self._cos_v    = cos_v[:, np.newaxis]           # (H, 1) — needed for simulation

        # Grid bounds for a flat surface at the default altitude (for reference)
        self._alpha_h = alpha_h
        self._theta_h = theta_h
        self._alpha_v = alpha_v
        self._theta_v = theta_v

    # ── Public API ────────────────────────────────────────────────────────────

    def project(self, frame, altitude_mm: float | None = None):
        """
        Project a Tau camera Frame object to a gridded elevation patch.

        Extracts frame.data_depth (float32, metres, H×W) then calls project_depth.

        Parameters
        ----------
        frame       : TauLidarCommon Frame returned by camera.readFrame()
        altitude_mm : camera altitude above reference plane (mm).
                      Defaults to the value passed to __init__.

        Returns
        -------
        patch   : (ny, nx) float32 elevation array, mm.  NaN where no data.
        x_grid  : (nx,) float32 — X coordinate of each column centre, mm.
        y_grid  : (ny,) float32 — Y coordinate of each row centre, mm.
        """
        depth_m = np.frombuffer(frame.data_depth, dtype=np.float32).reshape(self.H, self.W)
        return self.project_depth(depth_m, altitude_mm)

    def project_depth(self,
                      data_depth_m: np.ndarray,
                      altitude_mm: float | None = None):
        """
        Project a raw Z map (metres) to a gridded elevation patch.

        Parameters
        ----------
        data_depth_m : (H, W) float32/float64 array of Z values in metres.
                       NaN marks invalid (low-amplitude / saturated) pixels.
        altitude_mm  : camera altitude above reference plane (mm).

        Returns
        -------
        patch   : (ny, nx) float32 elevation array, mm.  NaN where no data.
        x_grid  : (nx,) float32 — X coordinate of each column centre, mm.
        y_grid  : (ny,) float32 — Y coordinate of each row centre, mm.
        """
        if altitude_mm is None:
            altitude_mm = self.default_altitude_mm

        Z_m = np.asarray(data_depth_m, dtype=np.float64)   # (H, W)

        # Print-plane coordinates (mm) for every pixel
        X_mm        = Z_m * self._X_factor * 1000.0        # (H, W)
        Y_mm        = Z_m * self._Y_factor * 1000.0        # (H, W)
        elev_mm     = altitude_mm - Z_m * 1000.0           # (H, W)

        # Discard invalid pixels and physically implausible depths.
        # - Z < min_range: below the sensor's hardware minimum — these pass the
        #   amplitude filter but are garbage (15–50mm phantoms on specular surfaces).
        # - Z > 3*altitude: more than 2 altitudes below the reference plane — ghost
        #   or multi-bounce returns.
        min_z_m  = (self.cfg.min_range_mm or 0.0) / 1000.0
        max_z_m  = 3.0 * altitude_mm / 1000.0
        valid    = np.isfinite(Z_m) & (Z_m > min_z_m) & (Z_m < max_z_m)
        x_pts    = X_mm[valid]
        y_pts    = Y_mm[valid]
        elev_pts = elev_mm[valid]

        x_grid, y_grid = self._footprint_grid(altitude_mm)

        if x_pts.size == 0:
            patch = np.full((y_grid.size, x_grid.size), np.nan, dtype=np.float32)
            return patch, x_grid.astype(np.float32), y_grid.astype(np.float32)

        # Bin valid points onto the regular grid (median per cell — robust to outliers)
        x_edges = _cell_edges(x_grid)
        y_edges = _cell_edges(y_grid)

        result = stats.binned_statistic_2d(
            x_pts, y_pts, elev_pts,
            statistic="median",
            bins=[x_edges, y_edges],
        )
        # scipy returns shape (nx, ny); transpose to (ny, nx) for image convention
        patch = result.statistic.T.astype(np.float64)

        # Gaussian smoothing with NaN-safe normalised convolution
        sigma_cells = self.cfg.smoothing_sigma_mm / self.cfg.output_grid_res_mm
        if sigma_cells > 0:
            patch = _gaussian_filter_nan(patch, sigma_cells)

        return patch.astype(np.float32), x_grid.astype(np.float32), y_grid.astype(np.float32)

    def footprint_mm(self, altitude_mm: float | None = None) -> tuple[float, float]:
        """
        Return the (width, height) in mm of the scan footprint on a flat surface.

        Uses the theoretical extreme-pixel angles, not an approximation.
        """
        if altitude_mm is None:
            altitude_mm = self.default_altitude_mm
        x_grid, y_grid = self._footprint_grid(altitude_mm)
        return float(x_grid[-1] - x_grid[0]), float(y_grid[-1] - y_grid[0])

    # ── Private helpers ───────────────────────────────────────────────────────

    def _footprint_grid(self, altitude_mm: float):
        """Build regular (x_grid, y_grid) axes covering the scan footprint."""
        res = self.cfg.output_grid_res_mm

        # Grid bounds: X and Y of the extreme pixels on a flat surface at altitude h
        # X[edge] = altitude * cot(gamma_h[edge]) ;  Y[edge] = -altitude * tan(gamma_v[edge])
        x_max = altitude_mm / math.tan(self._alpha_h)
        x_min = altitude_mm / math.tan(self._alpha_h + self._theta_h)
        y_max = -altitude_mm * math.tan(self._alpha_v)
        y_min = -altitude_mm * math.tan(self._alpha_v + self._theta_v)

        # Ensure min < max (cot can be negative at the other edge)
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        x_grid = np.arange(x_min, x_max + res, res)
        y_grid = np.arange(y_min, y_max + res, res)
        return x_grid, y_grid


# ── Module-level helpers ──────────────────────────────────────────────────────

def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Convert an array of evenly-spaced cell centres to bin edges."""
    half = (centers[1] - centers[0]) / 2.0
    return np.concatenate([[centers[0] - half], centers + half])


def _gaussian_filter_nan(arr: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian filter that correctly handles NaN cells (normalised convolution).

    Cells whose entire neighbourhood is NaN stay NaN.
    """
    mask       = np.isnan(arr)
    filled     = np.where(mask, 0.0, arr)
    weights    = np.where(mask, 0.0, 1.0)

    smoothed        = ndimage.gaussian_filter(filled,   sigma)
    weight_smoothed = ndimage.gaussian_filter(weights,  sigma)

    with np.errstate(invalid="ignore"):
        result = smoothed / weight_smoothed

    result[weight_smoothed < 1e-6] = np.nan
    return result
