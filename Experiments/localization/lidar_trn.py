"""
Top-level Terrain-Relative Navigation (TRN) class.

Ties together:
  * RasterDEM        — reference height map from STL
  * RealSenseProjectionModel — converts depth frames to elevation patches
  * ParticleFilter   — SIR filter over gantry (x, y)
  * ncc_batch        — vectorised NCC scoring

Typical usage (waypoint-triggered, from TrackRunner measure_cb)
---------------------------------------------------------------
    trn = LidarTRN(dem, projection_model, sim_cfg, altitude_mm,
                   corner_tl, corner_br)
    trn.reset(init_pos=(gantry_x, gantry_y), mode=InitMode.WARM)

    # At each waypoint (gantry is Idle):
    result = trn.update(depth_m, gantry_pos=(gantry_x, gantry_y))
    # result.x_est_mm, result.y_est_mm — estimated position
    # result.particles, result.weights — for visualisation

Static one-shot test
--------------------
    result = trn.update(depth_m, gantry_pos=(gantry_x, gantry_y))
    # Opens visualiser, shows particle cloud on DEM, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .ncc import ncc_batch
from .particle_filter import InitMode, PFConfig, ParticleFilter
from .stl_dem import RasterDEM


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class TRNResult:
    """All outputs from a single TRN update step."""
    x_est_mm:        float          # weighted-mean X estimate (gantry mm)
    y_est_mm:        float          # weighted-mean Y estimate (gantry mm)
    ncc_score:       float          # NCC of the best-weight particle
    ess:             float          # effective sample size
    particles:       np.ndarray     # (N, 2) current particle cloud
    weights:         np.ndarray     # (N,)  normalised weights
    measured_patch:  np.ndarray     # (ny, nx) float32 — from RealSense
    predicted_patch: np.ndarray     # (ny, nx) float32 — at best particle
    x_grid:          np.ndarray     # (nx,)  camera X axis (mm)
    y_grid:          np.ndarray     # (ny,)  camera Y axis (mm)


# ── Main TRN class ────────────────────────────────────────────────────────────

class LidarTRN:
    """
    Terrain-Relative Navigation engine.

    Parameters
    ----------
    dem              : RasterDEM loaded from STL.
    projection_model : RealSenseProjectionModel instance.
    sim_cfg          : SimulationConfig (from experiment_config.yml).
    altitude_mm      : nominal camera altitude above terrain (mm).
    corner_tl        : gantry (x, y) mm at the TL corner of the DEM.
    corner_br        : gantry (x, y) mm at the BR corner of the DEM.
    x_sign           : +1 or -1 to flip camera X ↔ gantry mapping.
    y_sign           : +1 or -1 to flip camera Y ↔ gantry mapping.
    """

    def __init__(
        self,
        dem:              RasterDEM,
        projection_model,
        sim_cfg,
        altitude_mm:      float,
        corner_tl:        Tuple[float, float],
        corner_br:        Tuple[float, float],
        x_sign:           float = 1.0,
        y_sign:           float = 1.0,
    ):
        self._dem         = dem
        self._model       = projection_model
        self._altitude_mm = altitude_mm
        self._corner_tl   = corner_tl
        self._corner_br   = corner_br
        self._x_sign      = x_sign
        self._y_sign      = y_sign

        pf_cfg = PFConfig(
            n_particles         = int(sim_cfg.n_particles         or 500),
            process_noise_x_mm  = float(sim_cfg.process_noise_x_mm  or 5.0),
            process_noise_y_mm  = float(sim_cfg.process_noise_y_mm  or 5.0),
            initial_spread_x_mm = float(sim_cfg.initial_spread_x_mm or 30.0),
            initial_spread_y_mm = float(sim_cfg.initial_spread_y_mm or 30.0),
            x_min = float(min(corner_tl[0], corner_br[0])),
            x_max = float(max(corner_tl[0], corner_br[0])),
            y_min = float(min(corner_tl[1], corner_br[1])),
            y_max = float(max(corner_tl[1], corner_br[1])),
        )
        self._pf              = ParticleFilter(pf_cfg)
        self._prev_gantry_pos: Optional[Tuple[float, float]] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self,
              init_pos: Optional[Tuple[float, float]],
              mode:     InitMode = InitMode.WARM) -> None:
        """
        Reinitialise the particle cloud.

        Call this once at the start of each run (or static test).

        Parameters
        ----------
        init_pos : (x_mm, y_mm) seed position.
                   Ignored for COLD mode; None forces COLD mode.
        mode     : WARM = Gaussian around init_pos.
                   COLD = Gaussian around map centre.
        """
        self._pf.initialize(init_pos, mode)
        self._prev_gantry_pos = init_pos

    def update(self,
               depth_m:     np.ndarray,
               gantry_pos:  Tuple[float, float]) -> TRNResult:
        """
        Process one depth frame and return the updated position estimate.

        This is called synchronously (blocks until complete).  At N=500
        particles the typical wall time is 0.5–3 s depending on hardware.

        Parameters
        ----------
        depth_m    : (H, W) float32 depth map in metres; NaN = invalid.
        gantry_pos : (x_mm, y_mm) gantry encoder reading at capture time.
                     Used as the motion-model input (Δ from last update).

        Returns
        -------
        TRNResult with position estimate, particle cloud, and debug patches.
        """
        if not self._pf.initialized:
            self.reset(gantry_pos, InitMode.WARM)

        # ── Motion model ──────────────────────────────────────────────────────
        if self._prev_gantry_pos is not None:
            dx = gantry_pos[0] - self._prev_gantry_pos[0]
            dy = gantry_pos[1] - self._prev_gantry_pos[1]
        else:
            dx, dy = 0.0, 0.0
        self._prev_gantry_pos = gantry_pos

        self._pf.propagate(dx, dy)

        # ── Measurement: project depth frame to elevation patch ────────────────
        measured_patch, x_grid, y_grid = self._model.project_depth(
            depth_m, self._altitude_mm
        )
        half_x = float((x_grid[-1] - x_grid[0]) / 2.0)
        half_y = float((y_grid[-1] - y_grid[0]) / 2.0)
        nx_out = len(x_grid)
        ny_out = len(y_grid)

        # ── Score all particles at once ────────────────────────────────────────
        particles = self._pf.particles   # (N, 2) copy
        pred_batch = self._dem.sample_patch_batch(
            particles, half_x, half_y,
            self._corner_tl, self._corner_br,
            nx_out, ny_out,
            x_sign=self._x_sign,
            y_sign=self._y_sign,
        )   # (N, ny_out, nx_out)

        ncc_scores = ncc_batch(measured_patch, pred_batch)   # (N,) float32
        safe_scores = np.where(np.isfinite(ncc_scores), ncc_scores, 0.0)

        # ── Filter update ──────────────────────────────────────────────────────
        self._pf.update(safe_scores)
        self._pf.resample_if_needed()

        x_est, y_est = self._pf.estimate()
        ess = self._pf.ess()

        # ── Best-particle predicted patch for visualisation ────────────────────
        best_idx    = int(np.argmax(self._pf.weights))
        best_pred   = pred_batch[best_idx]
        best_ncc    = float(safe_scores[best_idx])

        return TRNResult(
            x_est_mm       = x_est,
            y_est_mm       = y_est,
            ncc_score      = best_ncc,
            ess            = ess,
            particles      = self._pf.particles,
            weights        = self._pf.weights,
            measured_patch = measured_patch,
            predicted_patch= best_pred,
            x_grid         = x_grid,
            y_grid         = y_grid,
        )
