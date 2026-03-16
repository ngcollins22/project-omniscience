"""
Intel RealSense D435 projection model.

Converts D435 depth frames (or synthetic depth maps) into gridded elevation
patches in the print/terrain coordinate frame.

Uses a standard pinhole camera model:

    fx = (W/2) / tan(fov_x_rad/2)
    fy = (H/2) / tan(fov_y_rad/2)
    cx = W/2,  cy = H/2

    X[y,x] = Z[y,x] * (x - cx) / fx     [m]
    Y[y,x] = Z[y,x] * (y - cy) / fy     [m]
    elev[y,x] = altitude_mm - Z[y,x] * 1000

The D435 depth stream provides the Z component (perpendicular distance from
the image plane) as uint16 millimetres.  Invalid pixels are 0.  The worker
thread converts this to float32 metres and marks zeros as NaN before queuing,
so project_depth() receives the same format as TauProjectionModel.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy import ndimage, stats

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import RealSenseSensorConfig


class RealSenseProjectionModel:
    """
    Projects RealSense D435 depth data to a gridded elevation patch.

    Parameters
    ----------
    cfg          : RealSenseSensorConfig loaded from experiment_config.yml
    altitude_mm  : default camera altitude above the terrain reference plane (mm).
                   Can be overridden on each call to project_depth / project.
    """

    def __init__(self, cfg: RealSenseSensorConfig, altitude_mm: float):
        self.cfg = cfg
        self.default_altitude_mm = altitude_mm

        W = cfg.resolution_x   # 640
        H = cfg.resolution_y   # 480
        self.W = W
        self.H = H

        fov_x = math.radians(cfg.fov_x_deg)   # 87° → 1.5184 rad
        fov_y = math.radians(cfg.fov_y_deg)   # 58° → 1.0123 rad

        fx = (W / 2.0) / math.tan(fov_x / 2.0)
        fy = (H / 2.0) / math.tan(fov_y / 2.0)
        cx = W / 2.0
        cy = H / 2.0

        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._fov_x = fov_x
        self._fov_y = fov_y

        x_idx = np.arange(W, dtype=np.float64)
        y_idx = np.arange(H, dtype=np.float64)

        # _X_factor[x] = (x - cx) / fx  →  X_mm = Z_m * _X_factor * 1000
        # _Y_factor[y] = (y - cy) / fy  →  Y_mm = Z_m * _Y_factor * 1000
        self._X_factor = ((x_idx - cx) / fx)[np.newaxis, :]   # (1, W)
        self._Y_factor = ((y_idx - cy) / fy)[:, np.newaxis]   # (H, 1)

    # ── Public API ────────────────────────────────────────────────────────────

    def project(self, depth_m: np.ndarray, altitude_mm: float | None = None):
        """
        Project a depth array to a gridded elevation patch.

        Parameters
        ----------
        depth_m     : (H, W) float32 array of Z values in metres.
                      NaN marks invalid (zero-depth) pixels.
        altitude_mm : camera altitude above reference plane (mm).

        Returns
        -------
        patch   : (ny, nx) float32 elevation array, mm.  NaN where no data.
        x_grid  : (nx,) float32 — X coordinate of each column centre, mm.
        y_grid  : (ny,) float32 — Y coordinate of each row centre, mm.
        """
        return self.project_depth(depth_m, altitude_mm)

    def project_depth(self,
                      data_depth_m: np.ndarray,
                      altitude_mm: float | None = None):
        """
        Project a raw Z map (metres) to a gridded elevation patch.

        Parameters
        ----------
        data_depth_m : (H, W) float32/float64 array of Z values in metres.
                       NaN marks invalid pixels.
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
        X_mm    = Z_m * self._X_factor * 1000.0   # (H, W)
        Y_mm    = Z_m * self._Y_factor * 1000.0   # (H, W)
        elev_mm = altitude_mm - Z_m * 1000.0      # (H, W)

        # Discard invalid pixels and physically implausible depths
        min_z_m = (self.cfg.min_range_mm or 0.0) / 1000.0
        max_z_m = 3.0 * altitude_mm / 1000.0
        valid   = np.isfinite(Z_m) & (Z_m > min_z_m) & (Z_m < max_z_m)
        x_pts    = X_mm[valid]
        y_pts    = Y_mm[valid]
        elev_pts = elev_mm[valid]

        x_grid, y_grid = self._footprint_grid(altitude_mm)

        if x_pts.size == 0:
            patch = np.full((y_grid.size, x_grid.size), np.nan, dtype=np.float32)
            return patch, x_grid.astype(np.float32), y_grid.astype(np.float32)

        x_edges = _cell_edges(x_grid)
        y_edges = _cell_edges(y_grid)

        result = stats.binned_statistic_2d(
            x_pts, y_pts, elev_pts,
            statistic="median",
            bins=[x_edges, y_edges],
        )
        # scipy returns (nx, ny); transpose to (ny, nx) for image convention
        patch = result.statistic.T.astype(np.float64)

        sigma_cells = self.cfg.smoothing_sigma_mm / self.cfg.output_grid_res_mm
        if sigma_cells > 0:
            patch = _gaussian_filter_nan(patch, sigma_cells)

        return patch.astype(np.float32), x_grid.astype(np.float32), y_grid.astype(np.float32)

    def footprint_mm(self, altitude_mm: float | None = None) -> tuple[float, float]:
        """Return the (width, height) in mm of the scan footprint on a flat surface."""
        if altitude_mm is None:
            altitude_mm = self.default_altitude_mm
        x_grid, y_grid = self._footprint_grid(altitude_mm)
        return float(x_grid[-1] - x_grid[0]), float(y_grid[-1] - y_grid[0])

    # ── Private helpers ───────────────────────────────────────────────────────

    def _footprint_grid(self, altitude_mm: float):
        """Build regular (x_grid, y_grid) axes covering the scan footprint."""
        res = self.cfg.output_grid_res_mm
        half_x = altitude_mm * math.tan(self._fov_x / 2.0)
        half_y = altitude_mm * math.tan(self._fov_y / 2.0)
        x_grid = np.arange(-half_x, half_x + res, res)
        y_grid = np.arange(-half_y, half_y + res, res)
        return x_grid, y_grid


# ── Module-level helpers ──────────────────────────────────────────────────────

def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Convert an array of evenly-spaced cell centres to bin edges."""
    half = (centers[1] - centers[0]) / 2.0
    return np.concatenate([[centers[0] - half], centers + half])


def _gaussian_filter_nan(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian filter that correctly handles NaN cells (normalised convolution)."""
    mask   = np.isnan(arr)
    filled  = np.where(mask, 0.0, arr)
    weights = np.where(mask, 0.0, 1.0)

    smoothed        = ndimage.gaussian_filter(filled,  sigma)
    weight_smoothed = ndimage.gaussian_filter(weights, sigma)

    with np.errstate(invalid="ignore"):
        result = smoothed / weight_smoothed

    result[weight_smoothed < 1e-6] = np.nan
    return result


if __name__ == "__main__":
    # Quick smoke test — no camera required
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from config import load_config

    cfg = load_config()
    model = RealSenseProjectionModel(cfg.realsense, altitude_mm=100.0)
    W, H = cfg.realsense.resolution_x, cfg.realsense.resolution_y
    synthetic = np.full((H, W), 0.1, dtype=np.float32)   # flat surface at 100 mm
    patch, xg, yg = model.project_depth(synthetic, altitude_mm=100.0)
    print(f"patch shape: {patch.shape}  valid: {int(np.isfinite(patch).sum())}  "
          f"elev range: [{float(np.nanmin(patch)):.1f}, {float(np.nanmax(patch)):.1f}] mm")
    fw, fh = model.footprint_mm(100.0)
    print(f"footprint at 100 mm altitude: {fw:.1f} × {fh:.1f} mm")
