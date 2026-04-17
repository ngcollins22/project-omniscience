"""
omni_replay.py  —  OMNIScience combined session replay
=======================================================
Two-panel video:

    ┌──────────────┬──────────────────────────────────────────────┐
    │  Query frame │  Full reference map                          │
    │  (left half  │  - Green  trail = GT gantry path             │
    │   of         │  - Cyan   trail = EST camera-vision path     │
    │   drawMatches│  - Green  lines = ORB inlier matches         │
    │   output)    │  - Magenta rect = query footprint            │
    └──────────────┴──────────────────────────────────────────────┘

cv2.drawMatches is used for the right panel so match lines are drawn
from query keypoints straight to their matched locations on the reference.
The lat/lon trails are then composited on top.

Colour conventions:
    Green  — GT ground truth (gantry)
    Cyan   — EST camera vision estimate
    Magenta — ORB homography footprint

Usage
-----
    python omni_replay.py  CSV_PATH_OR_SESSION_DIR  REFERENCE_MAP  [options]

Options
    --fps N          Output FPS (default: inferred)
    --speed S        Speed multiplier (default 1.0)
    --out PATH       Output video (default SESSION_DIR/omni_replay.mp4)
    --ref-scale S    Scale reference before ORB (default 1.0)
    --nfeatures N    ORB nfeatures (default 50000)
    --ratio T        Lowe ratio threshold (default 0.7)
    --min-matches N  Min matches for homography (default 10)
    --query-scale S  Scale query frames before ORB (default 1.0)
    --out-width W    Output video width px (default 1920)
    --trail-dot R    Trail dot radius on ref map px (default 4)
    --trail-alpha A  Trail persistence per frame 0-1 (default 1.0)

Example
-------
    python omni_replay.py sessions/20260416_120000 mars_12k_color.jpg \\
        --ref-scale 0.5 --speed 2.0
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ── Colours (BGR for cv2) ─────────────────────────────────────────────────────
# Green = GT ground truth
COL_GT_DIM_BGR     = (  0,  80,   0)
COL_GT_BRIGHT_BGR  = (136, 255,   0)   # #00ff88 in BGR
COL_GT_LINE_BGR    = ( 68, 204,  68)

# Cyan = EST camera vision
COL_EST_DIM_BGR    = (120,  40,  30)
COL_EST_BRIGHT_BGR = (255, 212,   0)   # #00d4ff in BGR
COL_EST_LINE_BGR   = (204, 136,  68)

# ORB match lines — green (same family as GT, "correctly located on map")
COL_MATCH_BGR      = (  0, 255,   0)

# Homography outline — magenta
COL_OUTLINE_BGR    = (255,   0, 255)

# HUD
COL_HUD_BG         = ( 30,  15,  15)   # BGR
COL_HUD_TEXT       = (220, 220, 220)


# ── Utilities ─────────────────────────────────────────────────────────────────

def safe_float(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        # aggressively strip whitespace from every header name
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for r in reader:
            rows.append({k.strip(): v.strip() for k, v in r.items()})
    return rows


def infer_fps(rows):
    ts = []
    for r in rows:
        try:
            ts.append(float(r["timestamp_s"]))
        except (KeyError, ValueError):
            pass
    if len(ts) < 2:
        return 5.0
    diffs = sorted([ts[i+1]-ts[i] for i in range(len(ts)-1) if ts[i+1] > ts[i]])
    median = diffs[len(diffs)//2] if diffs else 0
    return round(1.0/median, 1) if median > 0 else 5.0


# ── Equirectangular lat/lon → pixel ───────────────────────────────────────────

def ll_to_px(lat, lon, w, h):
    """lon -180→+180 = left→right,  lat +90→-90 = top→bottom."""
    x = int((lon + 180.0) / 360.0 * w)
    y = int((90.0 - lat)  / 180.0 * h)
    return x, y


# ── Reference: one-time ORB ───────────────────────────────────────────────────

def build_reference(ref_path, ref_scale, nfeatures):
    print(f"[INFO] Loading reference: {ref_path}")
    img = cv2.imread(str(ref_path))
    if img is None:
        sys.exit(f"[ERROR] Cannot read: {ref_path}")
    print(f"[INFO] Reference: {img.shape[1]}x{img.shape[0]}")
    if ref_scale != 1.0:
        nw = int(img.shape[1] * ref_scale)
        nh = int(img.shape[0] * ref_scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        print(f"[INFO] Scaled to {nw}x{nh}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"[INFO] ORB on reference (nfeatures={nfeatures}) …")
    t0 = time.time()
    orb_r = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb_r.detectAndCompute(gray, None)
    print(f"[OK  ] {len(kp)} keypoints  ({time.time()-t0:.1f}s)")
    if des is None or len(kp) == 0:
        sys.exit("[ERROR] No keypoints found in reference.")
    return img, gray, kp, des          # img is BGR


# ── Per-frame ORB ─────────────────────────────────────────────────────────────

def run_orb(query_bgr, ref_gray, ref_kp, ref_des, orb, bf, ratio, min_matches):
    """
    Rotate query 90° CW, run ORB + knnMatch + homography.
    Returns (query_rot, kp1, good, inlier_mask, dst_poly).
    """
    query_rot  = cv2.rotate(query_bgr, cv2.ROTATE_90_CLOCKWISE)
    query_gray = cv2.cvtColor(query_rot, cv2.COLOR_BGR2GRAY)
    kp1, des1  = orb.detectAndCompute(query_gray, None)

    if des1 is None or len(kp1) == 0:
        return query_rot, [], [], None, None

    knn  = bf.knnMatch(des1, ref_des, k=2)
    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]
    good = sorted(good, key=lambda x: x.distance)

    dst_poly    = None
    inlier_mask = None

    if len(good) >= min_matches:
        src = np.float32([kp1[m.queryIdx].pt  for m in good]).reshape(-1,1,2)
        dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if M is not None:
            inlier_mask = mask.ravel().tolist()
            hq, wq = query_gray.shape
            corners = np.float32([[0,0],[0,hq-1],[wq-1,hq-1],[wq-1,0]]).reshape(-1,1,2)
            dst_poly = cv2.perspectiveTransform(corners, M)

    return query_rot, kp1, good, inlier_mask, dst_poly


# ── Trail drawing on BGR image ────────────────────────────────────────────────

def draw_trail(img, history, w, h, col_dim, col_bright, col_line, dot_r):
    """Paint full history trail onto img (BGR, in-place)."""
    n = len(history)
    if n == 0:
        return
    pts = []
    for i, (lat, lon) in enumerate(history):
        if math.isnan(lat) or math.isnan(lon):
            continue
        px, py = ll_to_px(lat, lon, w, h)
        a  = (i + 1) / max(n, 1)
        b  = int(col_dim[0] + a * (col_bright[0] - col_dim[0]))
        g  = int(col_dim[1] + a * (col_bright[1] - col_dim[1]))
        r  = int(col_dim[2] + a * (col_bright[2] - col_dim[2]))
        rd = max(2, int(dot_r * a))
        cv2.circle(img, (px, py), rd, (b, g, r), -1, cv2.LINE_AA)
        pts.append((px, py))
    if len(pts) >= 2:
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i+1], col_line, 1, cv2.LINE_AA)


# ── HUD ───────────────────────────────────────────────────────────────────────

def draw_hud(img, row, idx, total, elapsed, n_kp, n_good, n_inliers):
    gt_lat = safe_float(row.get("lat_deg"))
    gt_lon = safe_float(row.get("lon_deg"))
    es_lat = safe_float(row.get("estimated_latitude_deg"))
    es_lon = safe_float(row.get("estimated_longitude_deg"))

    def fll(v):
        return f"{v:+.3f}" if not math.isnan(v) else "n/a"

    if not (math.isnan(gt_lat) or math.isnan(es_lat)):
        err_str = f"{math.hypot(es_lat-gt_lat, es_lon-gt_lon):.3f}deg"
    else:
        err_str = "n/a"

    lines = [
        f"Frame {idx+1}/{total}  t={elapsed:.1f}s",
        f"GT   lat {fll(gt_lat)}  lon {fll(gt_lon)}",
        f"EST  lat {fll(es_lat)}  lon {fll(es_lon)}",
        f"Err {err_str}",
        f"kp:{n_kp}  matches:{n_good}  inliers:{n_inliers}",
    ]

    lh, pad = 17, 6
    bw = 340
    bh = len(lines) * lh + pad * 2
    roi = img[pad:pad+bh, pad:pad+bw]
    cv2.addWeighted(np.full_like(roi, COL_HUD_BG), 0.78, roi, 0.22, 0, dst=roi)
    for i, line in enumerate(lines):
        ty = pad*2 + i*lh
        cv2.putText(img, line, (pad*2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0,0,0),      2, cv2.LINE_AA)
        cv2.putText(img, line, (pad*2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, COL_HUD_TEXT,  1, cv2.LINE_AA)


# ── Legend ────────────────────────────────────────────────────────────────────

def draw_legend(img):
    """Small colour legend in the bottom-left corner of the right panel."""
    items = [
        (COL_GT_BRIGHT_BGR,  "GT path (gantry)"),
        (COL_EST_BRIGHT_BGR, "EST path (camera vision)"),
        (COL_MATCH_BGR,      "ORB inlier matches"),
        (COL_OUTLINE_BGR,    "Query footprint"),
    ]
    h, w = img.shape[:2]
    lh, pad = 16, 6
    bw = 210
    bh = len(items) * lh + pad * 2
    y0 = h - bh - pad
    roi = img[y0:y0+bh, pad:pad+bw]
    cv2.addWeighted(np.full_like(roi, COL_HUD_BG), 0.72, roi, 0.28, 0, dst=roi)
    for i, (col, label) in enumerate(items):
        ty = y0 + pad + i * lh + lh // 2
        cv2.circle(img, (pad*2 + 6, ty), 5, col, -1, cv2.LINE_AA)
        cv2.putText(img, label, (pad*2 + 16, ty + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_HUD_TEXT, 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def render_video(args):
    # ── Paths ─────────────────────────────────────────────────────────────────
    inp      = Path(args.input).resolve()
    csv_path = (inp / "session_with_estimates.csv") if inp.is_dir() else inp
    if not csv_path.exists():
        sys.exit(f"[ERROR] CSV not found: {csv_path}")
    session_dir = csv_path.parent

    ref_path = Path(args.reference).resolve()
    if not ref_path.exists():
        sys.exit(f"[ERROR] Reference not found: {ref_path}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    print(f"[INFO] CSV: {csv_path}")
    rows = load_csv(csv_path)
    if not rows:
        sys.exit("[ERROR] CSV is empty.")
    total = len(rows)
    print(f"[INFO] {total} rows")
    print(f"[INFO] Columns: {list(rows[0].keys())}")

    has_est = ("estimated_latitude_deg"  in rows[0] and
               "estimated_longitude_deg" in rows[0])
    if not has_est:
        print("[WARN] No estimated_latitude_deg / estimated_longitude_deg — "
              "EST trail will not be drawn.")
    else:
        # Quick check: how many rows actually have valid EST values?
        n_valid = sum(
            1 for r in rows
            if not math.isnan(safe_float(r.get("estimated_latitude_deg")))
        )
        print(f"[INFO] EST values: {n_valid}/{total} rows have valid data")

    # ── FPS ───────────────────────────────────────────────────────────────────
    rec_fps = infer_fps(rows)
    out_fps = max(1.0, (args.fps if args.fps else rec_fps) * args.speed)
    print(f"[INFO] Recorded ~{rec_fps:.1f} fps  ->  {out_fps:.1f} fps (x{args.speed})")

    # ── Output ────────────────────────────────────────────────────────────────
    out_path = Path(args.out) if args.out else session_dir / "omni_replay.mp4"
    print(f"[INFO] Output -> {out_path}")

    # ── Build reference ───────────────────────────────────────────────────────
    ref_color, ref_gray, ref_kp, ref_des = build_reference(
        ref_path, args.ref_scale, args.nfeatures)
    ref_h, ref_w = ref_gray.shape

    orb = cv2.ORB_create(nfeatures=args.nfeatures)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ── Layout ────────────────────────────────────────────────────────────────
    # cv2.drawMatches produces [query | ref] side by side.
    # We want the total width = args.out_width.
    # The ref panel occupies ref_w/(ref_w + query_w) of that.
    # We'll size the ref display to args.out_width * ref_frac, then derive
    # query display width from what's left.
    #
    # Strategy: scale the reference to a fixed display height, then the query
    # panel is whatever drawMatches makes it (query_rot width scaled to same height).

    # Display height = driven by reference aspect scaled to out_width * 0.72
    ref_disp_w  = int(args.out_width * 0.72)
    ref_scale_d = ref_disp_w / ref_w
    ref_disp_h  = max(1, int(ref_h * ref_scale_d))
    video_h     = ref_disp_h

    # Probe query size from first valid frame
    first_bgr = None
    for row in rows:
        fp = session_dir / row.get("filepath","").replace("\\","/")
        if fp.exists():
            first_bgr = cv2.imread(str(fp))
            if first_bgr is not None:
                break
    if first_bgr is None:
        sys.exit("[ERROR] No frame images found.")

    if args.query_scale != 1.0:
        first_bgr = cv2.resize(first_bgr,
            (int(first_bgr.shape[1]*args.query_scale),
             int(first_bgr.shape[0]*args.query_scale)),
            interpolation=cv2.INTER_AREA)

    # After 90° CW rotation the query becomes (orig_w, orig_h) -> (orig_h, orig_w)
    probe_rot = cv2.rotate(first_bgr, cv2.ROTATE_90_CLOCKWISE)
    q_scale_d = video_h / probe_rot.shape[0]
    query_disp_w = max(1, int(probe_rot.shape[1] * q_scale_d))
    query_disp_h = video_h

    video_w = query_disp_w + 2 + ref_disp_w
    print(f"[INFO] Layout: query={query_disp_w}x{query_disp_h}  "
          f"ref={ref_disp_w}x{ref_disp_h}")
    print(f"[INFO] Output frame: {video_w}x{video_h}")

    # Pre-scale the reference for display
    ref_disp = cv2.resize(ref_color, (ref_disp_w, ref_disp_h),
                          interpolation=cv2.INTER_AREA)

    # Pre-scale ref keypoints to display coordinates
    ref_kp_disp = [cv2.KeyPoint(kp.pt[0]*ref_scale_d, kp.pt[1]*ref_scale_d,
                                kp.size*ref_scale_d, kp.angle,
                                kp.response, kp.octave, kp.class_id)
                   for kp in ref_kp]

    # ── Trail state ───────────────────────────────────────────────────────────
    gt_history  = []
    est_history = []

    # ── Video writer ──────────────────────────────────────────────────────────
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             out_fps, (video_w, video_h))
    if not writer.isOpened():
        sys.exit(f"[ERROR] Cannot open VideoWriter: {out_path}")

    try:
        t0_ts = float(rows[0]["timestamp_s"])
    except (KeyError, ValueError):
        t0_ts = 0.0

    t_wall = time.time()
    print(f"[INFO] Rendering {total} frames …")

    for idx, row in enumerate(rows):
        # ── Accumulate trails ─────────────────────────────────────────────────
        gt_lat = safe_float(row.get("lat_deg"))
        gt_lon = safe_float(row.get("lon_deg"))
        if not (math.isnan(gt_lat) or math.isnan(gt_lon)):
            gt_history.append((gt_lat, gt_lon))

        if has_est:
            es_lat = safe_float(row.get("estimated_latitude_deg"))
            es_lon = safe_float(row.get("estimated_longitude_deg"))
            if not (math.isnan(es_lat) or math.isnan(es_lon)):
                est_history.append((es_lat, es_lon))

        # ── Load + ORB ────────────────────────────────────────────────────────
        fp = session_dir / row.get("filepath","").replace("\\","/")
        query_bgr = cv2.imread(str(fp)) if fp.exists() else None

        if query_bgr is not None and args.query_scale != 1.0:
            query_bgr = cv2.resize(
                query_bgr,
                (int(query_bgr.shape[1]*args.query_scale),
                 int(query_bgr.shape[0]*args.query_scale)),
                interpolation=cv2.INTER_AREA)

        if query_bgr is not None:
            query_rot, kp1, good, inlier_mask, dst_poly = run_orb(
                query_bgr, ref_gray, ref_kp, ref_des,
                orb, bf, args.ratio, args.min_matches)
        else:
            query_rot   = np.zeros((query_disp_h, query_disp_w, 3), dtype=np.uint8)
            kp1, good, inlier_mask, dst_poly = [], [], None, None

        # ── Build ref panel: drawMatches → ref panel with lines ───────────────
        # drawMatches needs ref keypoints in ORB-scale coords (ref_kp),
        # but we want the display to show the display-scaled ref image.
        # Solution: scale query keypoints UP by ref_scale_d so they land
        # correctly when drawMatches lays out [query_disp | ref_disp].
        #
        # Scale query image to display height first
        q_h, q_w = query_rot.shape[:2]
        q_scale_to_disp = query_disp_h / q_h
        query_scaled = cv2.resize(query_rot,
                                  (max(1, int(q_w * q_scale_to_disp)), query_disp_h),
                                  interpolation=cv2.INTER_AREA)
        qs_h, qs_w = query_scaled.shape[:2]

        # Scale query keypoints to match the scaled query image
        kp1_disp = [cv2.KeyPoint(kp.pt[0]*q_scale_to_disp, kp.pt[1]*q_scale_to_disp,
                                 kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                    for kp in kp1]

        # Scale inlier match dst coords into display-ref space
        # (ref_kp_disp already has display coords)
        draw_params = dict(
            matchColor=COL_MATCH_BGR,
            singlePointColor=(80, 80, 80),   # dim grey for unmatched kp
            matchesMask=inlier_mask,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )
        vis = cv2.drawMatches(
            query_scaled, kp1_disp,
            ref_disp.copy(), ref_kp_disp,
            good, None, **draw_params)
        # vis shape: (max(qs_h, ref_disp_h),  qs_w + ref_disp_w, 3)
        # The ref portion starts at x = qs_w

        # Extract the ref panel from vis (right half)
        ref_panel = vis[:ref_disp_h, qs_w:qs_w + ref_disp_w].copy()

        # Extract the query panel from vis (left half), ensure exact size
        query_panel = vis[:query_disp_h, :qs_w].copy()
        if query_panel.shape[1] != query_disp_w:
            query_panel = cv2.resize(query_panel, (query_disp_w, query_disp_h))

        # ── Draw homography footprint on ref panel ────────────────────────────
        if dst_poly is not None:
            dst_poly_disp = dst_poly * ref_scale_d
            cv2.polylines(ref_panel, [np.int32(dst_poly_disp)],
                          True, COL_OUTLINE_BGR, 2, cv2.LINE_AA)

        # ── Draw lat/lon trails on ref panel ──────────────────────────────────
        draw_trail(ref_panel, gt_history,
                   ref_disp_w, ref_disp_h,
                   COL_GT_DIM_BGR, COL_GT_BRIGHT_BGR, COL_GT_LINE_BGR,
                   args.trail_dot)

        draw_trail(ref_panel, est_history,
                   ref_disp_w, ref_disp_h,
                   COL_EST_DIM_BGR, COL_EST_BRIGHT_BGR, COL_EST_LINE_BGR,
                   args.trail_dot)

        # ── HUD + legend ──────────────────────────────────────────────────────
        n_inliers = sum(inlier_mask) if inlier_mask else 0
        try:
            elapsed = float(row["timestamp_s"]) - t0_ts
        except (KeyError, ValueError):
            elapsed = 0.0

        draw_hud(ref_panel, row, idx, total, elapsed,
                 len(kp1), len(good), n_inliers)
        draw_legend(ref_panel)

        # ── Composite final frame ─────────────────────────────────────────────
        canvas = np.zeros((video_h, video_w, 3), dtype=np.uint8)
        canvas[:, 0:query_disp_w]                           = query_panel
        canvas[:, query_disp_w:query_disp_w+2]              = (60, 60, 80)
        canvas[:, query_disp_w+2:query_disp_w+2+ref_disp_w] = ref_panel

        writer.write(canvas)

        if (idx+1) % max(1, total//40) == 0 or idx == total-1:
            pct  = (idx+1)/total*100
            wall = time.time()-t_wall
            eta  = (wall/(idx+1))*(total-idx-1)
            bar  = "\u2588"*int(pct/5)+"\u2591"*(20-int(pct/5))
            print(f"  [{bar}] {idx+1}/{total} ({pct:.0f}%)  "
                  f"{wall:.0f}s  ETA {eta:.0f}s", end="\r")

    writer.release()
    wall = time.time()-t_wall
    print(f"\n[OK  ] {out_path}")
    print(f"       {total} frames | {out_fps:.1f} fps | {video_w}x{video_h} | "
          f"{wall:.1f}s ({wall/total:.2f}s/frame)")
    print(f"       GT trail: {len(gt_history)} pts | EST trail: {len(est_history)} pts")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="OMNIScience combined lat/lon trail + ORB match video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("input",     help="session.csv or session directory")
    p.add_argument("reference", help="Reference map image")
    p.add_argument("--fps",         type=float, metavar="N")
    p.add_argument("--speed",       type=float, default=1.0,   metavar="S")
    p.add_argument("--out",         metavar="PATH")
    p.add_argument("--ref-scale",   type=float, default=1.0,   metavar="S",
                   help="Scale reference before ORB (default 1.0)")
    p.add_argument("--nfeatures",   type=int,   default=50000, metavar="N")
    p.add_argument("--ratio",       type=float, default=0.7,   metavar="T")
    p.add_argument("--min-matches", type=int,   default=10,    metavar="N")
    p.add_argument("--query-scale", type=float, default=1.0,   metavar="S")
    p.add_argument("--out-width",   type=int,   default=1920,  metavar="W")
    p.add_argument("--trail-dot",   type=int,   default=4,     metavar="R",
                   help="Trail dot radius px (default 4)")
    p.add_argument("--trail-alpha", type=float, default=1.0,   metavar="A",
                   help="Trail persistence per frame (default 1.0 = never fade)")
    render_video(p.parse_args())


if __name__ == "__main__":
    main()