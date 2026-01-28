import math
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from org.orekit.utils import *
from org.orekit.bodies import *
from mars_constellation import *
from org.orekit.time import *

from matplotlib import cm

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import re

def plot_across_mars_path(constellation,
                          times,
                          inertial_pvs,
                          sat_ids,
                          path,
                          latencies,  # seconds, len = len(path) + 1
                          t_idx=0,
                          mars_texture_path="mars_viking_full.jpg",
                          lat1=0.0, lon1=0.0,
                          lat2=0.0, lon2=180.0):
    """
    constellation : Constellation
        Holds mars_shape, mars_fixed, mars_inertial.
    times : list[AbsoluteDate]
    inertial_pvs : Dict[int, List[PVCoordinates]]
        Same structure as in compute_across_mars_latency.
    sat_ids : list[int]
        Satellite IDs (indices into latencies / inertial_pvs).
    path : list[int]
        Satellite IDs returned by A* (start -> ... -> goal).
    latencies : list[float]
        Leg latencies in seconds, ordered:
          [gp1->path[0], path[0]->path[1], ..., path[-2]->path[-1], path[-1]->gp2]
    """

    # ---- Basic checks ----
    if path and len(latencies) != len(path) + 1:
        raise ValueError(
            f"Expected len(latencies) = len(path) + 1, got "
            f"{len(latencies)} vs {len(path) + 1}"
        )

    mars_shape = constellation.mars_shape
    body = constellation.mars_fixed
    inertial = constellation.mars_inertial
    date = times[t_idx]

    # --- Ground points -> inertial frame ---
    gp1 = GeodeticPoint(math.radians(lat1), math.radians(lon1), 0.0)
    gp2 = GeodeticPoint(math.radians(lat2), math.radians(lon2), 0.0)

    body_to_inertial = body.getTransformTo(inertial, date).toStaticTransform()

    pos_gp1_body = mars_shape.transform(gp1)
    pos_gp2_body = mars_shape.transform(gp2)

    pos_gp1_inertial = body_to_inertial.transformPosition(pos_gp1_body)
    pos_gp2_inertial = body_to_inertial.transformPosition(pos_gp2_body)

    r_gp1 = np.array([pos_gp1_inertial.getX(),
                      pos_gp1_inertial.getY(),
                      pos_gp1_inertial.getZ()])
    r_gp2 = np.array([pos_gp2_inertial.getX(),
                      pos_gp2_inertial.getY(),
                      pos_gp2_inertial.getZ()])

    # --- Find nearest satellite to each ground point ---
    min_dist1 = float("inf")
    min_dist2 = float("inf")
    sat_idx_gp1 = None
    sat_idx_gp2 = None

    sat_positions = {}

    for sid in sat_ids:
        pv_sat = inertial_pvs[sid][t_idx]
        r_sat = np.array([pv_sat.getPosition().getX(),
                          pv_sat.getPosition().getY(),
                          pv_sat.getPosition().getZ()])
        sat_positions[sid] = r_sat
        d1 = np.linalg.norm(r_sat - r_gp1)
        d2 = np.linalg.norm(r_sat - r_gp2)
        if d1 < min_dist1:
            min_dist1 = d1
            sat_idx_gp1 = sid
        if d2 < min_dist2:
            min_dist2 = d2
            sat_idx_gp2 = sid

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # --- Mars sphere ---
    R = mars_shape.getEquatorialRadius()
    n_lat = 80
    n_lon = 160
    lon_vals = np.linspace(-np.pi, np.pi, n_lon)
    lat_vals = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

    X = R * np.cos(lat_grid) * np.cos(lon_grid)
    Y = R * np.cos(lat_grid) * np.sin(lon_grid)
    Z = R * np.sin(lat_grid)

    if mars_texture_path is not None:
        img = plt.imread(mars_texture_path)
        h, w = img.shape[:2]
        u = (lon_grid + np.pi) / (2 * np.pi) * (w - 1)
        v = (np.pi / 2 - lat_grid) / np.pi * (h - 1)
        u_idx = u.astype(int)
        v_idx = v.astype(int)

        facecolors = img[v_idx, u_idx]
        if facecolors.max() > 1.0:
            facecolors = facecolors / 255.0

        ax.plot_surface(X, Y, Z,
                        rstride=1, cstride=1,
                        facecolors=facecolors,
                        linewidth=0, antialiased=False)
    else:
        ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0, alpha=0.5)

    # --- All satellites (background) ---
    all_sat_array = np.array([sat_positions[sid] for sid in sat_ids])
    ax.scatter(all_sat_array[:, 0],
               all_sat_array[:, 1],
               all_sat_array[:, 2],
               s=10,
               color="0.7")  # light gray

    # --- Ground points ---
    ax.scatter([r_gp1[0]], [r_gp1[1]], [r_gp1[2]], s=50, marker="^", color="black")
    ax.scatter([r_gp2[0]], [r_gp2[1]], [r_gp2[2]], s=50, marker="^", color="black")
    ax.text(r_gp1[0], r_gp1[1], r_gp1[2], "G1", fontsize=8, color="black")
    ax.text(r_gp2[0], r_gp2[1], r_gp2[2], "G2", fontsize=8, color="black")

    # --- Path sats + arrows + latency labels ---
    if path and len(path) > 0:
        path_coords = np.array([sat_positions[sid] for sid in path])

        # Highlight path sats
        ax.scatter(path_coords[:, 0],
                   path_coords[:, 1],
                   path_coords[:, 2],
                   s=40,
                   color="black")

        # Label sats with ID
        for sid in path:
            r = sat_positions[sid]
            ax.text(r[0], r[1], r[2], str(sid), fontsize=12, color="black")

        # Build segments: G1 -> path[0], path[i-1] -> path[i], path[-1] -> G2
        segments = []
        segment_labels_ms = []

        segments.append((r_gp1, sat_positions[path[0]]))
        segment_labels_ms.append(latencies[0] * 1e3)

        for i in range(1, len(path)):
            p0 = sat_positions[path[i - 1]]
            p1 = sat_positions[path[i]]
            segments.append((p0, p1))
            segment_labels_ms.append(latencies[i] * 1e3)

        segments.append((sat_positions[path[-1]], r_gp2))
        segment_labels_ms.append(latencies[-1] * 1e3)

        # Use a categorical colormap for legs
        cmap = cm.get_cmap("tab10")
        n_segs = len(segments)

        for seg_idx, ((p0, p1), lat_ms) in enumerate(zip(segments, segment_labels_ms)):
            color = cmap(seg_idx % 10)
            vec = p1 - p0

            # Arrow for this leg
            ax.quiver(
                p0[0], p0[1], p0[2],
                vec[0], vec[1], vec[2],
                arrow_length_ratio=0.1,
                linewidth=2,
                color=color
            )

            # Latency label at midpoint
            mid = 0.5 * (p0 + p1)
            ax.text(
                mid[0], mid[1], mid[2],
                f"{lat_ms:.1f} ms",
                fontsize=12,
                color=color
            )

    # --- Equal aspect ---
    xs = np.concatenate([X.flatten(), all_sat_array[:, 0], [r_gp1[0], r_gp2[0]]])
    ys = np.concatenate([Y.flatten(), all_sat_array[:, 1], [r_gp1[1], r_gp2[1]]])
    zs = np.concatenate([Z.flatten(), all_sat_array[:, 2], [r_gp1[2], r_gp2[2]]])

    max_range = np.array([xs.max() - xs.min(),
                          ys.max() - ys.min(),
                          zs.max() - zs.min()]).max() / 2.0

    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    total_latency_s = float(sum(latencies))
    total_latency_ms = total_latency_s * 1e3

    ax.set_title(
        f"Mars cross-planet path for "
        f"{getattr(constellation, 'config', 'Unknown Constellation')} "
        f"at t_idx={t_idx}\n"
        f"One-way latency: {total_latency_ms:.3f} ms"
    )

    ax.set_box_aspect((1, 1, 1))

    plt.show()


def plot_ground_tracks_on_basemap(tracks,
                                  basemap_path: str,
                                  constellation=None,
                                  title: str | None = None,
                                  silent: bool = False):
    """
    Load the Mars basemap and overlay satellite ground tracks.

    tracks: dict[sat_id] -> {"lat": [...], "lon": [...]}
    basemap_path: path to the Mars equirectangular JPG/PNG.
    constellation: optional Constellation object to build a nice title from.
    title: explicit title override. If None and constellation is provided,
           a default title is generated from its config.
    """
    img = load_mars_basemap(basemap_path)

    # ----- build title -----
    if title is None and constellation is not None:
        cfg = constellation.config
        name = getattr(cfg, "name", "Constellation")
        incl = getattr(cfg, "inclination_deg", None)
        h = getattr(cfg, "altitude_km", None)
        p = getattr(cfg, "planes", None)
        t = getattr(cfg, "total_sats", None)
        f = getattr(cfg, "phasing", None)

        parts = [name]
        if incl is not None:
            parts.append(f"i={incl:.1f}°, h={h:.0f} km")
        if p is not None and t is not None and f is not None:
            parts.append(f"Walker Delta: t={t}, p={p}, f={f}")
        # Use " | " in the visible title
        title = " | ".join(parts)

    if title is None:
        title = "Mars Ground Tracks"

    # Global equirectangular: lon ∈ [-180,180], lat ∈ [-90,90]
    extent = (-180.0, 180.0, -90.0, 90.0)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.imshow(img,
              origin="upper",
              extent=extent,
              aspect="auto")

    # Axes & grid
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # ----- plot each satellite track, breaking at wrap-around -----
    for sat_id, tr in tracks.items():
        lats = tr["lat"]
        lons = tr["lon"]
        if len(lats) == 0:
            continue

        seg_lats = [lats[0]]
        seg_lons = [lons[0]]

        first_segment = True
        for i in range(1, len(lons)):
            lon_prev = lons[i - 1]
            lon_curr = lons[i]

            # If we jump across the ±180° boundary, start a new segment
            if abs(lon_curr - lon_prev) > 180.0:
                ax.plot(
                    seg_lons,
                    seg_lats,
                    linewidth=1.0,
                    alpha=0.8,
                    label=f"Sat {sat_id}" if first_segment else None,
                )
                first_segment = False
                seg_lats = [lats[i]]
                seg_lons = [lons[i]]
            else:
                seg_lats.append(lats[i])
                seg_lons.append(lons[i])

        ax.plot(
            seg_lons,
            seg_lats,
            linewidth=1.0,
            alpha=0.8,
            label=f"Sat {sat_id}" if first_segment else None,
        )

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)

    ax.legend(loc="upper right", fontsize="x-small", ncol=2)

    plt.tight_layout()

    if silent:
        # Build a safe filename (strip/replace illegal chars for Windows)
        # Illegal on Windows: \ / : * ? " < > |
        unsafe = title
        safe = re.sub(r'[\\/:*?"<>|]', "_", unsafe)
        safe = safe.replace(" ", "_")
        fig_path = Path(f"{safe}_ground_tracks.png")
        plt.savefig(fig_path)
        plt.close(fig)
        print(f"Saved ground tracks to {fig_path}")
    else:
        plt.show()


def load_mars_basemap(path: str) -> np.ndarray:
    """
    Load a Mars equirectangular basemap image (e.g. MOLA/Viking JPG).

    Parameters
    ----------
    path : str
        Path to the JPG/PNG file.

    Returns
    -------
    img : np.ndarray
        Image array as returned by matplotlib.pyplot.imread().
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Basemap file not found: {p}")

    # Disable Pillow decompression bomb protection for this trusted image
    Image.MAX_IMAGE_PIXELS = None

    with Image.open(str(p)) as im:
        # Optionally force RGB; some scientific maps can be paletted/greyscale
        im = im.convert("RGB")
        img = np.array(im)

    if img.ndim not in (2, 3):
        raise ValueError(f"Unexpected image dimensions for basemap: {img.shape}")
    
    return img


def show_mars_basemap(img: np.ndarray,
                      extent=(-180.0, 180.0, -90.0, 90.0),
                      title: str = "Mars Equirectangular Basemap") -> None:
    """
    Display a Mars basemap image with longitude/latitude axes.

    Parameters
    ----------
    img : np.ndarray
        Image array from load_mars_basemap().
    extent : tuple
        (lon_min, lon_max, lat_min, lat_max) in degrees. For a global
        equirectangular Mars map, (-180, 180, -90, 90) is appropriate.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Note: origin='upper' matches typical geotiff/JPEG orientation
    ax.imshow(img,
              origin="upper",
              extent=extent,
              aspect="auto")

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)

    # Optional: grid + ticks
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()


def show_mars_basemap_from_file(path: str,
                                extent=(-180.0, 180.0, -90.0, 90.0),
                                title: str = "Mars Equirectangular Basemap") -> None:
    """
    Convenience wrapper: load basemap from file and display it.
    """
    img = load_mars_basemap(path)
    show_mars_basemap(img, extent=extent, title=title)

def plot_pdop_p95_map_on_basemap(lat_vals: np.ndarray,
                                 lon_vals: np.ndarray,
                                 p95_pdop: np.ndarray,
                                 basemap_path: str,
                                 title: str = "PDOP P95 (95th percentile)", 
                                 silent: bool = False):
    """
    Overlay a PDOP P95 map on the Mars basemap.

    lat_vals: 1D array of latitudes (deg)
    lon_vals: 1D array of longitudes (deg)
    p95_pdop: 2D array [n_lat, n_lon] of PDOP P95 values (NaN where no coverage)
    """
    img = load_mars_basemap(basemap_path)

    # Full Mars extent for background
    full_extent = (-180.0, 180.0, -90.0, 90.0)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Basemap
    ax.imshow(img,
              origin="upper",
              extent=full_extent,
              aspect="auto")

    # Restrict color layer to the ROI: lon_vals x lat_vals
    Lon, Lat = np.meshgrid(lon_vals, lat_vals)

    # pcolormesh expects the same shape; our p95_pdop is (n_lat, n_lon)
    cmap = plt.get_cmap("viridis")

    # Use a masked array to hide NaNs
    p95_masked = np.ma.masked_invalid(p95_pdop)

    pcm = ax.pcolormesh(
        Lon,
        Lat,
        p95_masked,
        cmap=cmap,
        shading="auto",
        alpha=0.6,
    )

    # Axes / labels
    ax.set_xlim(-180, 180)
    # Focus on ±45 deg latitude, but keep full map visible if you want:
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)

    cbar = fig.colorbar(pcm, ax=ax, label="PDOP (95th percentile)")
    cbar.ax.set_ylabel("PDOP P95")

    plt.tight_layout()
    if silent:
        # save as file instead of showing
        fig_path = Path(f"{title.replace(' ', '_')}_p95_map.png")
        plt.savefig(fig_path)
        plt.close(fig)
    if not silent:
        plt.show()
