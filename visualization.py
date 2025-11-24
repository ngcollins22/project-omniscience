import math
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from org.orekit.utils import PVCoordinates
from org.orekit.bodies import GeodeticPoint

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import re

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
