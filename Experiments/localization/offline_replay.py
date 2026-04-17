"""
Offline TRN replay tool.

Loads a recorded LiDAR session, feeds depth frames through LidarTRN in
chronological order, and produces a summary figure showing estimated vs
ground-truth paths on the DEM, along with per-frame NCC scores and
position error.

Usage
-----
    python -m Experiments.localization.offline_replay \\
        --session  sessions/lidar_20250416_120000 \\
        --stl      mars_output.stl \\
        --corner_tl  "gx_mm,gy_mm" \\
        --corner_br  "gx_mm,gy_mm" \\
        [--res       1.0]          \\
        [--n_particles 500]        \\
        [--mode      warm|cold]    \\
        [--output    replay.png]   \\
        [--show]

The session directory must contain:
    session.csv   — columns: timestamp_s, color_path, depth_path, lat_deg, lon_deg
    frames/       — depth PNGs (uint16 mm, 0 = invalid)

lat_deg / lon_deg in the CSV are used as ground truth if the replay tool can
convert them to gantry mm (requires --corner_tl, --corner_br and the map
lat/lon bounds stored alongside the session).  If bounds are unavailable,
ground truth is plotted in lat/lon space instead.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

# Make repo root importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _parse_pair(s: str):
    parts = s.replace("(", "").replace(")", "").split(",")
    return float(parts[0].strip()), float(parts[1].strip())


def _load_depth_png(path: str) -> np.ndarray:
    """Load a 16-bit depth PNG (mm uint16, 0=invalid) → float32 metres."""
    try:
        import cv2
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        depth_m = raw.astype(np.float32) / 1000.0
        depth_m[depth_m == 0] = np.nan
        return depth_m
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(
        description="Offline TRN replay — runs the particle filter on a "
                    "saved session and plots the estimated vs ground-truth path."
    )
    ap.add_argument("--session",     required=True, help="Path to session directory")
    ap.add_argument("--stl",         required=True, help="Path to mars_output.stl")
    ap.add_argument("--corner_tl",   required=True, help='"x_mm,y_mm" of TL corner')
    ap.add_argument("--corner_br",   required=True, help='"x_mm,y_mm" of BR corner')
    ap.add_argument("--res",         type=float, default=1.0,
                    help="DEM raster resolution mm/px (default 1.0)")
    ap.add_argument("--n_particles", type=int,   default=500)
    ap.add_argument("--mode",        choices=["warm", "cold"], default="warm")
    ap.add_argument("--output",      default=None, help="Save figure to this path")
    ap.add_argument("--show",        action="store_true",
                    help="Display interactive figure")
    args = ap.parse_args()

    corner_tl = _parse_pair(args.corner_tl)
    corner_br = _parse_pair(args.corner_br)

    # ── Load session CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(args.session, "session.csv")
    if not os.path.isfile(csv_path):
        print(f"[ERROR] session.csv not found in {args.session}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    print(f"[INFO ] Loaded {len(rows)} rows from session.csv")

    # ── Load DEM ──────────────────────────────────────────────────────────────
    from Experiments.localization.stl_dem import load_or_rasterize
    print(f"[INFO ] Loading DEM from {args.stl} at {args.res} mm/px …")
    dem = load_or_rasterize(args.stl, res_mm=args.res)
    print(f"[INFO ] DEM shape: {dem.nx} × {dem.ny} pixels")

    # ── Build projection model ────────────────────────────────────────────────
    from config import load_config
    from Experiments.simulation.realsense_projection import RealSenseProjectionModel
    cfg = load_config()
    altitude_mm = float(cfg.gantry.camera_altitude_mm or 100.0)
    model = RealSenseProjectionModel(cfg.realsense, altitude_mm)

    # ── Build TRN ─────────────────────────────────────────────────────────────
    from Experiments.localization.lidar_trn import LidarTRN
    from Experiments.localization.particle_filter import InitMode, PFConfig
    sim_cfg = cfg.simulation
    # Override n_particles from CLI
    class _SimCfgOverride:
        n_particles          = args.n_particles
        process_noise_x_mm   = sim_cfg.process_noise_x_mm
        process_noise_y_mm   = sim_cfg.process_noise_y_mm
        initial_spread_x_mm  = sim_cfg.initial_spread_x_mm
        initial_spread_y_mm  = sim_cfg.initial_spread_y_mm

    trn = LidarTRN(dem, model, _SimCfgOverride(), altitude_mm, corner_tl, corner_br)
    init_mode = InitMode.WARM if args.mode == "warm" else InitMode.COLD

    # ── Replay ────────────────────────────────────────────────────────────────
    est_positions = []   # (x_mm, y_mm)
    gt_positions  = []   # (x_mm, y_mm) from CSV gantry cols (may be NaN)
    ncc_scores    = []
    timestamps    = []

    first = True
    for i, row in enumerate(rows):
        depth_rel = row.get("depth_path", "").strip()
        if not depth_rel:
            continue
        depth_abs = os.path.join(args.session, depth_rel)
        depth_m   = _load_depth_png(depth_abs)
        if depth_m is None:
            print(f"[WARN ] Could not load depth frame {i}: {depth_abs}")
            continue

        # Ground truth from CSV (stored as x_mm, y_mm if available)
        try:
            gt_x = float(row.get("x_mm") or row.get("gantry_x_mm", "nan"))
            gt_y = float(row.get("y_mm") or row.get("gantry_y_mm", "nan"))
        except (ValueError, TypeError):
            gt_x = gt_y = float("nan")

        if first:
            init_pos = (gt_x, gt_y) if np.isfinite(gt_x) else None
            trn.reset(init_pos, init_mode)
            first = False

        gantry_pos = (gt_x if np.isfinite(gt_x) else 0.0,
                      gt_y if np.isfinite(gt_y) else 0.0)

        try:
            result = trn.update(depth_m, gantry_pos)
        except Exception as exc:
            print(f"[WARN ] TRN update failed at frame {i}: {exc}")
            continue

        est_positions.append((result.x_est_mm, result.y_est_mm))
        gt_positions.append((gt_x, gt_y))
        ncc_scores.append(result.ncc_score)
        try:
            timestamps.append(float(row["timestamp_s"]))
        except (KeyError, ValueError):
            timestamps.append(float(i))

        print(f"[{i+1:4d}] est=({result.x_est_mm:7.1f},{result.y_est_mm:7.1f}) mm  "
              f"NCC={result.ncc_score:.3f}  ESS={result.ess:.0f}")

    if not est_positions:
        print("[ERROR] No frames processed.", file=sys.stderr)
        sys.exit(1)

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        if not (args.show):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from Experiments.localization.visualizer import _dem_to_rgba
    except ImportError:
        print("[ERROR] matplotlib not installed — cannot produce plot.", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#1e1e2e")
    for ax in axes:
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#585b70")
        for sp in ax.spines.values():
            sp.set_edgecolor("#313244")

    tl, br = corner_tl, corner_br
    x_lo, x_hi = sorted([tl[0], br[0]])
    y_lo, y_hi = sorted([tl[1], br[1]])

    # Panel 1: DEM with paths
    dem_img = _dem_to_rgba(dem.data)
    axes[0].imshow(dem_img,
                   extent=[y_lo, y_hi, x_hi, x_lo],
                   aspect="auto", origin="upper")
    est_x = [p[0] for p in est_positions]
    est_y = [p[1] for p in est_positions]
    axes[0].plot(est_y, est_x, "-o", color="#cba6f7", ms=3, lw=1.5,
                 label="Estimated")

    gt_x_v = [p[0] for p in gt_positions if np.isfinite(p[0])]
    gt_y_v = [p[1] for p in gt_positions if np.isfinite(p[1])]
    if gt_x_v:
        axes[0].plot(gt_y_v, gt_x_v, "-o", color="#a6e3a1", ms=3, lw=1.5,
                     label="Ground truth")
    axes[0].set_xlabel("Gantry Y (mm)", color="#a6adc8")
    axes[0].set_ylabel("Gantry X (mm)", color="#a6adc8")
    axes[0].set_title("Path on DEM", color="#cdd6f4")
    axes[0].legend(fontsize=8, facecolor="#313244", labelcolor="#cdd6f4",
                   edgecolor="#585b70")

    # Panel 2: NCC scores over time
    axes[1].plot(timestamps, ncc_scores, "-", color="#89b4fa", lw=1.5)
    axes[1].axhline(0, color="#585b70", lw=0.5, ls="--")
    axes[1].set_xlabel("Time (s)", color="#a6adc8")
    axes[1].set_ylabel("NCC score", color="#a6adc8")
    axes[1].set_title("Per-frame NCC", color="#cdd6f4")
    axes[1].set_ylim(-1.05, 1.05)

    # Panel 3: position error
    errors = []
    for (ex, ey), (gx, gy) in zip(est_positions, gt_positions):
        if np.isfinite(gx) and np.isfinite(gy):
            errors.append(np.hypot(ex - gx, ey - gy))
        else:
            errors.append(float("nan"))
    if any(np.isfinite(e) for e in errors):
        axes[2].plot(timestamps, errors, "-", color="#fab387", lw=1.5)
        valid_err = [e for e in errors if np.isfinite(e)]
        axes[2].set_title(
            f"Position error  (mean={np.mean(valid_err):.1f} mm)",
            color="#cdd6f4",
        )
    else:
        axes[2].text(0.5, 0.5, "No ground truth available",
                     transform=axes[2].transAxes,
                     ha="center", va="center", color="#585b70")
        axes[2].set_title("Position error", color="#cdd6f4")
    axes[2].set_xlabel("Time (s)", color="#a6adc8")
    axes[2].set_ylabel("Error (mm)",  color="#a6adc8")

    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=120, facecolor="#1e1e2e")
        print(f"[OK   ] Figure saved → {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
