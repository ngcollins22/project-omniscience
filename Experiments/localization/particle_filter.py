"""
Sequential Importance Resampling (SIR) particle filter for 2-D position.

State space: (x_mm, y_mm) in gantry machine coordinates.

Typical usage
-------------
    cfg = PFConfig(n_particles=500, ...)
    pf  = ParticleFilter(cfg)
    pf.initialize(center=(gantry_x, gantry_y), mode=InitMode.WARM)

    # At each measurement:
    pf.propagate(dx=gantry_delta_x, dy=gantry_delta_y)
    pf.update(ncc_scores)       # (N,) array from ncc_batch()
    pf.resample_if_needed()
    x_est, y_est = pf.estimate()
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class InitMode(Enum):
    WARM = "warm"   # particles centred at a supplied position
    COLD = "cold"   # particles spread over the full map


@dataclass
class PFConfig:
    """Particle filter hyper-parameters."""
    n_particles:         int   = 500
    process_noise_x_mm:  float = 5.0
    process_noise_y_mm:  float = 5.0
    initial_spread_x_mm: float = 30.0    # warm-start 1-sigma spread
    initial_spread_y_mm: float = 30.0
    likelihood_k:        float = 2.0     # weight = max(0, NCC)^k
    x_min:               float = 0.0
    x_max:               float = 400.0
    y_min:               float = 0.0
    y_max:               float = 800.0


class ParticleFilter:
    """
    2-D SIR particle filter.

    Attributes (read-only after each step)
    ----------------------------------------
    particles : (N, 2) float64 — current particle positions (x_mm, y_mm)
    weights   : (N,)  float64 — normalised importance weights
    """

    def __init__(self, cfg: PFConfig,
                 rng: Optional[np.random.Generator] = None):
        self.cfg   = cfg
        self._rng  = rng if rng is not None else np.random.default_rng()
        self._particles: np.ndarray = np.empty((cfg.n_particles, 2))
        self._weights:   np.ndarray = np.full(cfg.n_particles, 1.0 / cfg.n_particles)
        self._initialized = False

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self,
                   center: Optional[Tuple[float, float]],
                   mode:   InitMode = InitMode.WARM) -> None:
        """
        (Re-)seed the particle cloud.

        Parameters
        ----------
        center : (x_mm, y_mm) seed position.  Ignored for COLD mode (uses
                 map centre).  If None, forces COLD mode.
        mode   : WARM — Gaussian around *center* with initial_spread.
                 COLD — Gaussian around map centre with spread = map_size/4.
        """
        N = self.cfg.n_particles

        if mode == InitMode.COLD or center is None:
            cx = (self.cfg.x_min + self.cfg.x_max) / 2.0
            cy = (self.cfg.y_min + self.cfg.y_max) / 2.0
            sx = (self.cfg.x_max - self.cfg.x_min) / 4.0
            sy = (self.cfg.y_max - self.cfg.y_min) / 4.0
        else:
            cx, cy = center
            sx = self.cfg.initial_spread_x_mm
            sy = self.cfg.initial_spread_y_mm

        x = self._rng.normal(cx, sx, N)
        y = self._rng.normal(cy, sy, N)
        x = np.clip(x, self.cfg.x_min, self.cfg.x_max)
        y = np.clip(y, self.cfg.y_min, self.cfg.y_max)
        self._particles = np.column_stack([x, y])
        self._weights   = np.full(N, 1.0 / N)
        self._initialized = True

    # ── Prediction step ───────────────────────────────────────────────────────

    def propagate(self, dx: float, dy: float) -> None:
        """
        Move all particles by (dx, dy) plus Gaussian process noise.

        *dx*, *dy* are the gantry encoder delta since the last update.
        """
        N = self.cfg.n_particles
        noise = self._rng.standard_normal((N, 2))
        self._particles[:, 0] += dx + noise[:, 0] * self.cfg.process_noise_x_mm
        self._particles[:, 1] += dy + noise[:, 1] * self.cfg.process_noise_y_mm
        self._particles[:, 0] = np.clip(self._particles[:, 0],
                                        self.cfg.x_min, self.cfg.x_max)
        self._particles[:, 1] = np.clip(self._particles[:, 1],
                                        self.cfg.y_min, self.cfg.y_max)

    # ── Update step ───────────────────────────────────────────────────────────

    def update(self, ncc_scores: np.ndarray) -> None:
        """
        Multiply weights by the NCC-derived likelihoods and normalise.

        NaN scores (insufficient valid pixels) are treated as likelihood 0.

        Parameters
        ----------
        ncc_scores : (N,) array of NCC values in [-1, 1] or NaN.
        """
        k = self.cfg.likelihood_k
        safe_scores = np.where(np.isfinite(ncc_scores), ncc_scores, 0.0)
        likelihoods = np.maximum(0.0, safe_scores) ** k

        # Guard: if every particle has zero likelihood keep a flat prior
        if likelihoods.sum() < 1e-300:
            likelihoods = np.ones(len(self._particles))

        self._weights *= likelihoods
        total = self._weights.sum()
        if total < 1e-300:
            self._weights[:] = 1.0 / len(self._weights)
        else:
            self._weights /= total

    # ── Estimate ──────────────────────────────────────────────────────────────

    def estimate(self) -> Tuple[float, float]:
        """Weighted-mean position estimate."""
        x = float(np.dot(self._weights, self._particles[:, 0]))
        y = float(np.dot(self._weights, self._particles[:, 1]))
        return x, y

    def ess(self) -> float:
        """Effective Sample Size = 1 / Σwi²."""
        return float(1.0 / np.sum(self._weights ** 2))

    # ── Resampling ────────────────────────────────────────────────────────────

    def resample_if_needed(self, threshold_frac: float = 0.5) -> None:
        """Systematic resampling when ESS < threshold_frac × N."""
        if self.ess() < threshold_frac * len(self._particles):
            self._systematic_resample()

    def _systematic_resample(self) -> None:
        N = len(self._particles)
        positions  = (np.arange(N) + self._rng.random()) / N
        cumsum     = np.cumsum(self._weights)
        indices    = np.searchsorted(cumsum, positions)
        indices    = np.clip(indices, 0, N - 1)
        self._particles = self._particles[indices].copy()
        self._weights   = np.full(N, 1.0 / N)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def particles(self) -> np.ndarray:
        return self._particles.copy()

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def initialized(self) -> bool:
        return self._initialized
