"""
STL → rasterised Digital Elevation Model for TRN localization.

Workflow
--------
1. Load (or re-use cached) DEM::

       dem = load_or_rasterize("mars_output.stl", res_mm=1.0)

2. Extract predicted elevation patches for a particle cloud::

       patches = dem.sample_patch_batch(
           particles,          # (N, 2) gantry (x_mm, y_mm)
           half_x_mm, half_y_mm,
           corner_tl, corner_br,
           nx_out, ny_out,
       )  # → (N, ny_out, nx_out) float32

Coordinate conventions
----------------------
* DEM ``data[i, j]`` indexes along the STL X axis (i) and Y axis (j).
* ``corner_tl`` (gantry mm) is assumed to correspond to the STL minimum-XY
  corner (DEM index [0, 0]).
* ``corner_br`` (gantry mm) corresponds to the STL maximum-XY corner
  (DEM index [nx-1, ny-1]).

Camera ↔ gantry axis mapping (adjustable via ``x_sign`` / ``y_sign``):
* Camera X (horizontal, x_grid)  ↔  Gantry Y (default x_sign=+1)
* Camera Y (vertical,   y_grid)  ↔  Gantry X (default y_sign=+1)
  If localization does not converge try flipping one or both signs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import map_coordinates


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class RasterDEM:
    """
    Rasterised surface height map derived from the terrain STL.

    Attributes
    ----------
    data      : (nx, ny) float32 array.  data[i, j] = Z at STL point
                (x_min + i*res, y_min + j*res).  NaN where the mesh has
                no surface (e.g. outside the bounding box).
    nx, ny    : DEM dimensions.
    res       : raster resolution (mm per pixel).
    stl_x_min, stl_y_min : STL origin offsets (mm).  Used only for
                reference; gantry↔DEM mapping uses the calibration corners.
    """
    data:      np.ndarray   # (nx, ny) float32
    nx:        int
    ny:        int
    res:       float
    stl_x_min: float
    stl_y_min: float

    # ── Single-particle patch ─────────────────────────────────────────────────

    def sample_patch(
        self,
        cx_gantry: float,
        cy_gantry: float,
        half_x_mm: float,
        half_y_mm: float,
        corner_tl: Tuple[float, float],
        corner_br:  Tuple[float, float],
        nx_out: int,
        ny_out: int,
        x_sign: float = 1.0,
        y_sign: float = 1.0,
    ) -> np.ndarray:
        """
        Extract a (ny_out, nx_out) elevation patch centred at a single
        gantry position.  Convenience wrapper around sample_patch_batch.
        """
        center = np.array([[cx_gantry, cy_gantry]])
        batch  = self.sample_patch_batch(
            center, half_x_mm, half_y_mm,
            corner_tl, corner_br, nx_out, ny_out,
            x_sign=x_sign, y_sign=y_sign,
        )
        return batch[0]

    # ── Vectorised batch patch extraction ─────────────────────────────────────

    def sample_patch_batch(
        self,
        centers: np.ndarray,
        half_x_mm: float,
        half_y_mm: float,
        corner_tl: Tuple[float, float],
        corner_br:  Tuple[float, float],
        nx_out: int,
        ny_out: int,
        x_sign: float = 1.0,
        y_sign: float = 1.0,
    ) -> np.ndarray:
        """
        Extract predicted elevation patches for N candidate positions at once.

        Parameters
        ----------
        centers   : (N, 2) array of gantry (x_mm, y_mm) positions.
        half_x_mm : half-width of the footprint in gantry X mm
                    (corresponds to camera x_grid half-extent).
        half_y_mm : half-height of the footprint in gantry Y mm
                    (corresponds to camera y_grid half-extent).
        corner_tl : gantry (x, y) mm when sensor is at the STL TL corner.
        corner_br : gantry (x, y) mm when sensor is at the STL BR corner.
        nx_out    : number of columns in the output patch (camera X axis).
        ny_out    : number of rows    in the output patch (camera Y axis).
        x_sign    : flip camera-X→gantry mapping (+1 or -1).
        y_sign    : flip camera-Y→gantry mapping (+1 or -1).

        Returns
        -------
        patches : (N, ny_out, nx_out) float32 — NaN where out-of-bounds or
                  no surface in DEM.
        """
        centers = np.asarray(centers, dtype=np.float64)
        N = len(centers)

        span_x = corner_br[0] - corner_tl[0]   # signed mm
        span_y = corner_br[1] - corner_tl[1]   # signed mm

        # Pixels per mm in each DEM axis
        px_per_mm_x = (self.nx - 1) / span_x   # may be negative if span < 0
        px_per_mm_y = (self.ny - 1) / span_y

        # DEM centre pixel for each particle
        i_ctr = (centers[:, 0] - corner_tl[0]) * px_per_mm_x   # (N,)
        j_ctr = (centers[:, 1] - corner_tl[1]) * px_per_mm_y   # (N,)

        # Camera-axis offsets → gantry-axis offsets → DEM pixel offsets
        #
        # Physical camera orientation (as mounted):
        #   Camera X (horizontal, wide axis)  ↔  Gantry X  (DEM i-axis)
        #   Camera Y (vertical,  narrow axis) ↔  Gantry Y  (DEM j-axis)
        #   Camera top (−Y in image)          →  Gantry −Y  (y_sign = +1)
        #
        # camera X position x_vals[c] drives DEM i-offset → varies along cols (axis 2)
        # camera Y position y_vals[r] drives DEM j-offset → varies along rows (axis 1)
        x_vals = np.linspace(-half_x_mm, half_x_mm, nx_out)  # (nx_out,)  camera X
        y_vals = np.linspace(-half_y_mm, half_y_mm, ny_out)  # (ny_out,)  camera Y

        di_col = x_sign * x_vals * px_per_mm_x   # (nx_out,)  DEM i offset per col
        dj_row = y_sign * y_vals * px_per_mm_y   # (ny_out,)  DEM j offset per row

        # Build full (N, ny_out, nx_out) coordinate arrays.
        # Use np.broadcast_to so no extra memory is allocated until ravel() copies.
        i_raw = (i_ctr[:, np.newaxis, np.newaxis]
                 + di_col[np.newaxis, np.newaxis, :])          # (N, 1, nx_out)
        j_raw = (j_ctr[:, np.newaxis, np.newaxis]
                 + dj_row[np.newaxis, :, np.newaxis])          # (N, ny_out, 1)
        i_coords = np.broadcast_to(i_raw, (N, ny_out, nx_out)).ravel()
        j_coords = np.broadcast_to(j_raw, (N, ny_out, nx_out)).ravel()

        # One vectorised bilinear interpolation call for all N×ny×nx points
        values = map_coordinates(
            self.data.astype(np.float64),
            [i_coords, j_coords],
            order=1,
            mode="constant",
            cval=np.nan,
        )
        return values.reshape(N, ny_out, nx_out).astype(np.float32)


# ── Loader / rasterizer ───────────────────────────────────────────────────────

def load_or_rasterize(stl_path: str, res_mm: float = 1.0) -> RasterDEM:
    """
    Return a RasterDEM for *stl_path*.

    On the first call the STL is rasterised and the result is cached in
    ``<stl_path>.dem.npy`` + ``<stl_path>.dem.json``.  Subsequent calls
    load the cache instantly (unless the STL has been modified).

    Parameters
    ----------
    stl_path : absolute or relative path to the STL file.
    res_mm   : raster resolution in mm per pixel (default 1.0).

    Raises
    ------
    ImportError  if ``trimesh`` is not installed.
    FileNotFoundError if *stl_path* does not exist.
    """
    if not os.path.isfile(stl_path):
        raise FileNotFoundError(f"STL not found: {stl_path}")

    cache_npy  = stl_path + ".dem.npy"
    cache_json = stl_path + ".dem.json"

    stl_mtime = os.path.getmtime(stl_path)

    # ── Try cache ─────────────────────────────────────────────────────────────
    if os.path.isfile(cache_npy) and os.path.isfile(cache_json):
        try:
            with open(cache_json) as f:
                meta = json.load(f)
            if (abs(meta["stl_mtime"] - stl_mtime) < 1.0
                    and abs(meta["res_mm"] - res_mm) < 1e-6):
                data = np.load(cache_npy)
                return RasterDEM(
                    data=data,
                    nx=data.shape[0],
                    ny=data.shape[1],
                    res=res_mm,
                    stl_x_min=meta["stl_x_min"],
                    stl_y_min=meta["stl_y_min"],
                )
        except Exception:
            pass  # cache corrupt or outdated → fall through to rasterize

    # ── Rasterize ─────────────────────────────────────────────────────────────
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "trimesh is required for STL rasterization.\n"
            "Install it with:  pip install trimesh"
        ) from exc

    mesh = trimesh.load(stl_path, force="mesh")
    (x_min, y_min, z_min), (x_max, y_max, z_max) = mesh.bounds

    nx = max(2, int(round((x_max - x_min) / res_mm)) + 1)
    ny = max(2, int(round((y_max - y_min) / res_mm)) + 1)

    x_range = np.linspace(x_min, x_max, nx)
    y_range = np.linspace(y_min, y_max, ny)

    # Build ray origins above the mesh (using meshgrid with ij indexing so
    # the resulting array is indexed as data[i_x, j_y])
    xg, yg = np.meshgrid(x_range, y_range, indexing="ij")   # (nx, ny)
    z_top  = float(z_max) + 1.0

    n_rays   = nx * ny
    origins  = np.column_stack([
        xg.ravel(),
        yg.ravel(),
        np.full(n_rays, z_top, dtype=np.float64),
    ])
    dirs = np.zeros((n_rays, 3))
    dirs[:, 2] = -1.0

    heights = np.full(n_rays, np.nan, dtype=np.float32)

    # intersects_location with multiple_hits=False returns the topmost surface
    # for a downward ray (first intersection encountered from above).
    try:
        locs, idx_ray, _ = mesh.ray.intersects_location(
            origins, dirs, multiple_hits=False
        )
        if len(locs) > 0:
            heights[idx_ray] = locs[:, 2].astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Ray casting failed: {exc}") from exc

    data = heights.reshape(nx, ny)

    # ── Save cache ────────────────────────────────────────────────────────────
    try:
        np.save(cache_npy, data)
        with open(cache_json, "w") as f:
            json.dump({
                "stl_mtime": stl_mtime,
                "res_mm":    res_mm,
                "nx":        nx,
                "ny":        ny,
                "stl_x_min": float(x_min),
                "stl_y_min": float(y_min),
                "stl_x_max": float(x_max),
                "stl_y_max": float(y_max),
            }, f, indent=2)
    except Exception:
        pass   # cache write failure is non-fatal

    return RasterDEM(
        data=data,
        nx=nx,
        ny=ny,
        res=res_mm,
        stl_x_min=float(x_min),
        stl_y_min=float(y_min),
    )
