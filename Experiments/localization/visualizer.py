"""
TRN Visualizer — rich matplotlib figure embedded in a Tkinter Toplevel.

Two display modes
-----------------
static  (``mode="static"``)
    Full 4-panel layout.  Call ``show_result()`` once after creation.
    Intended for the "TRN Static Test" button.

live    (``mode="live"``)
    2-panel layout (DEM overview + latest measured patch).
    Call ``update_result()`` after each waypoint during a run.

Panel layout (static mode)
--------------------------

  ┌────────────────────────┬──────────────────────┐
  │ DEM overview           │ 3-D point cloud       │
  │  · terrain colourmap   │  · coloured by elev   │
  │  · particle cloud      │  · interactive 3-D    │
  │  · GT pos  (green ✕)   │                       │
  │  · EST pos (purple ✕)  │                       │
  │  · footprint rects     │                       │
  ├────────────────────────┼──────────────────────┤
  │ Measured patch         │ Predicted patch       │
  │  NCC = 0.87  (title)   │  at estimated pos     │
  └────────────────────────┴──────────────────────┘
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import tkinter as tk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk,
    )
    from matplotlib import cm as _cm
    from matplotlib.patches import Rectangle
    _MPL_OK = True
except ImportError:
    _MPL_OK = False


class TRNVisualizer:
    """
    Tkinter Toplevel with an embedded matplotlib figure.

    Parameters
    ----------
    parent    : Tk root or any Tk widget.
    dem       : RasterDEM (used for the overhead image).
    corner_tl : gantry (x, y) mm at TL corner of DEM.
    corner_br : gantry (x, y) mm at BR corner of DEM.
    mode      : "static" (4-panel, opened once) or
                "live"   (2-panel, updated each waypoint).
    """

    def __init__(self, parent, dem, corner_tl, corner_br,
                 mode: str = "static"):
        if not _MPL_OK:
            raise ImportError("matplotlib is required for TRNVisualizer")

        self._dem       = dem
        self._corner_tl = corner_tl
        self._corner_br = corner_br
        self._mode      = mode

        # Build window
        self.win = tk.Toplevel(parent)
        self.win.title("TRN — " + ("Static Test" if mode == "static"
                                    else "Live Tracking"))
        self.win.configure(bg="#1e1e2e")

        # Precompute DEM image (terrain colourmap, normalised)
        self._dem_img = _dem_to_rgba(dem.data)

        # Build figure
        if mode == "static":
            fig_w, fig_h = 14, 9
        else:
            fig_w, fig_h = 8, 6

        self._fig = Figure(figsize=(fig_w, fig_h), dpi=100, facecolor="#1e1e2e")
        self._build_axes(mode)

        # History for live mode
        self._est_trail_x:  list[float] = []
        self._est_trail_y:  list[float] = []
        self._gt_trail_x:   list[float] = []
        self._gt_trail_y:   list[float] = []

        # Embed in Tk
        canvas = FigureCanvasTkAgg(self._fig, master=self.win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self.win)
        toolbar.update()
        toolbar.pack(fill=tk.X)
        self._canvas = canvas

    # ── Axis construction ─────────────────────────────────────────────────────

    def _build_axes(self, mode: str) -> None:
        if mode == "static":
            gs = self._fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
            self._ax_dem    = self._fig.add_subplot(gs[0, 0])
            self._ax_cloud  = self._fig.add_subplot(gs[0, 1], projection="3d")
            self._ax_meas   = self._fig.add_subplot(gs[1, 0])
            self._ax_pred   = self._fig.add_subplot(gs[1, 1])
        else:
            gs = self._fig.add_gridspec(1, 2, wspace=0.3)
            self._ax_dem   = self._fig.add_subplot(gs[0, 0])
            self._ax_meas  = self._fig.add_subplot(gs[0, 1])
            self._ax_cloud = None
            self._ax_pred  = None

        for ax in [self._ax_dem, self._ax_meas,
                   getattr(self, "_ax_pred", None)]:
            if ax is not None:
                ax.set_facecolor("#181825")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#313244")
                ax.tick_params(colors="#585b70", labelsize=7)
                ax.xaxis.label.set_color("#a6adc8")
                ax.yaxis.label.set_color("#a6adc8")
                ax.title.set_color("#cdd6f4")

        if self._ax_cloud is not None:
            self._ax_cloud.set_facecolor("#181825")
            self._ax_cloud.tick_params(colors="#585b70", labelsize=6)

    # ── Static show ───────────────────────────────────────────────────────────

    def show_result(
        self,
        result,
        gantry_pos_mm: Optional[Tuple[float, float]] = None,
        depth_m: Optional[np.ndarray] = None,
        projection_model=None,
    ) -> None:
        """
        Render all panels for a single TRNResult snapshot.

        Parameters
        ----------
        result          : TRNResult from LidarTRN.update().
        gantry_pos_mm   : (x_mm, y_mm) true gantry position (for GT marker).
        depth_m         : raw depth map (m) for the 3-D point cloud panel.
        projection_model: RealSenseProjectionModel (needed for point cloud).
        """
        self._draw_dem(result, gantry_pos_mm)
        self._draw_patches(result)
        if (self._ax_cloud is not None and depth_m is not None
                and projection_model is not None):
            self._draw_cloud(depth_m, projection_model, result)
        self._canvas.draw_idle()

    # ── Live update ───────────────────────────────────────────────────────────

    def update_result(
        self,
        result,
        gantry_pos_mm: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Update the DEM panel and measured-patch panel for live mode."""
        if gantry_pos_mm is not None:
            self._gt_trail_x.append(gantry_pos_mm[0])
            self._gt_trail_y.append(gantry_pos_mm[1])
        self._est_trail_x.append(result.x_est_mm)
        self._est_trail_y.append(result.y_est_mm)

        self._draw_dem(result, gantry_pos_mm)
        self._draw_patches(result)
        self._canvas.draw_idle()

    # ── DEM panel ─────────────────────────────────────────────────────────────

    def _draw_dem(self, result, gantry_pos_mm) -> None:
        ax = self._ax_dem
        ax.cla()
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#585b70", labelsize=7)
        ax.title.set_color("#cdd6f4")

        tl = self._corner_tl
        br = self._corner_br
        x_lo, x_hi = sorted([tl[0], br[0]])
        y_lo, y_hi = sorted([tl[1], br[1]])

        ax.imshow(
            self._dem_img,
            extent=[y_lo, y_hi, x_hi, x_lo],   # [left, right, bottom, top]
            aspect="auto",
            origin="upper",
        )
        ax.set_xlabel("Gantry Y (mm)", fontsize=7)
        ax.set_ylabel("Gantry X (mm)", fontsize=7)

        # Particle cloud
        pts = result.particles
        w   = result.weights
        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-12)
        ax.scatter(pts[:, 1], pts[:, 0], c=w_norm, cmap="plasma",
                   s=4, alpha=0.6, linewidths=0, zorder=3)

        # Trails (live mode)
        if self._est_trail_x:
            ax.plot(self._est_trail_y, self._est_trail_x,
                    "-", color="#cba6f7", lw=1, alpha=0.7, zorder=4)
        if self._gt_trail_x:
            ax.plot(self._gt_trail_y, self._gt_trail_x,
                    "-", color="#a6e3a1", lw=1, alpha=0.7, zorder=4)

        # Footprint half-extents (mm)
        if len(result.x_grid) > 1:
            hx = float((result.x_grid[-1] - result.x_grid[0]) / 2)
            hy = float((result.y_grid[-1] - result.y_grid[0]) / 2)
        else:
            hx = hy = 10.0

        # Estimated position
        ex, ey = result.x_est_mm, result.y_est_mm
        ax.plot(ey, ex, "x", color="#cba6f7", ms=10, mew=2, zorder=6,
                label=f"Est ({ex:.0f}, {ey:.0f})")
        ax.add_patch(Rectangle(
            (ey - hy, ex - hx), 2 * hy, 2 * hx,
            linewidth=1, edgecolor="#cba6f7", facecolor="none", zorder=5,
        ))

        # Ground truth
        if gantry_pos_mm is not None:
            gx, gy = gantry_pos_mm
            ax.plot(gy, gx, "x", color="#a6e3a1", ms=10, mew=2, zorder=6,
                    label=f"GT  ({gx:.0f}, {gy:.0f})")
            ax.add_patch(Rectangle(
                (gy - hy, gx - hx), 2 * hy, 2 * hx,
                linewidth=1, edgecolor="#a6e3a1", facecolor="none",
                linestyle="--", zorder=5,
            ))

        ax.set_title(
            f"DEM  NCC={result.ncc_score:.3f}  ESS={result.ess:.0f}",
            fontsize=8, color="#cdd6f4",
        )
        ax.legend(fontsize=6, loc="upper right",
                  facecolor="#313244", edgecolor="#585b70",
                  labelcolor="#cdd6f4")

    # ── Patch panels ─────────────────────────────────────────────────────────

    def _draw_patches(self, result) -> None:
        measured  = result.measured_patch
        predicted = result.predicted_patch
        xg = result.x_grid
        yg = result.y_grid

        # Shared colour scale
        both   = [v for v in [measured, predicted] if v is not None]
        finite = np.concatenate([b[np.isfinite(b)].ravel() for b in both])
        if len(finite) > 0:
            vmin, vmax = float(np.percentile(finite, 2)), float(np.percentile(finite, 98))
        else:
            vmin, vmax = 0.0, 1.0

        extent = [float(xg[0]), float(xg[-1]), float(yg[-1]), float(yg[0])]

        for ax, data, title in [
            (self._ax_meas, measured,  f"Measured  (alt={self._ax_meas})"),
            (getattr(self, "_ax_pred", None), predicted, "Predicted @ est"),
        ]:
            if ax is None or data is None:
                continue
            ax.cla()
            ax.set_facecolor("#181825")
            ax.tick_params(colors="#585b70", labelsize=7)

            disp = data.copy()
            disp[~np.isfinite(disp)] = vmin
            ax.imshow(disp, extent=extent, aspect="auto", origin="upper",
                      cmap="terrain", vmin=vmin, vmax=vmax)
            ax.set_xlabel("Camera X (mm)", fontsize=7, color="#a6adc8")
            ax.set_ylabel("Camera Y (mm)", fontsize=7, color="#a6adc8")

        self._ax_meas.set_title("Measured patch", fontsize=8, color="#cdd6f4")
        if self._ax_pred is not None:
            self._ax_pred.set_title(
                f"Predicted @ est  NCC={result.ncc_score:.3f}",
                fontsize=8, color="#cdd6f4",
            )

    # ── 3-D point cloud ───────────────────────────────────────────────────────

    def _draw_cloud(self, depth_m, projection_model, result) -> None:
        ax = self._ax_cloud
        if ax is None:
            return
        ax.cla()
        ax.set_facecolor("#181825")

        altitude_mm = getattr(projection_model, "default_altitude_mm", 100.0)
        Z_m = np.asarray(depth_m, dtype=np.float64)
        X_mm = Z_m * projection_model._X_factor * 1000.0
        Y_mm = Z_m * projection_model._Y_factor * 1000.0
        E_mm = altitude_mm - Z_m * 1000.0

        valid = (np.isfinite(Z_m) & (Z_m > 0) & (Z_m < altitude_mm * 3 / 1000.0))
        x_pts = X_mm[valid].ravel()
        y_pts = Y_mm[valid].ravel()
        e_pts = E_mm[valid].ravel()

        if len(x_pts) > 0:
            # Subsample for speed
            step = max(1, len(x_pts) // 4000)
            ax.scatter(
                x_pts[::step], y_pts[::step], e_pts[::step],
                c=e_pts[::step], cmap="terrain",
                s=0.8, alpha=0.7, linewidths=0,
            )
        ax.set_xlabel("X (mm)", fontsize=6, color="#a6adc8")
        ax.set_ylabel("Y (mm)", fontsize=6, color="#a6adc8")
        ax.set_zlabel("Elev (mm)", fontsize=6, color="#a6adc8")
        ax.set_title("Point cloud", fontsize=8, color="#cdd6f4")
        ax.tick_params(colors="#585b70", labelsize=6)


# ── Helper ────────────────────────────────────────────────────────────────────

def _dem_to_rgba(data: np.ndarray) -> np.ndarray:
    """Convert DEM height array to RGBA image using terrain colourmap."""
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return np.zeros((*data.shape, 4), dtype=np.uint8)
    lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
    normed = np.clip((data - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    normed[~np.isfinite(data)] = 0.0
    if _MPL_OK:
        rgba = (_cm.terrain(normed) * 255).astype(np.uint8)
    else:
        rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    return rgba
