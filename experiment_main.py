"""
OMNIScience Experiment Control Panel

Pipeline:
    1. Generate ground tracks (orekit)
    2. Connect to Arduino -> gantry
    3. Connect to Intel RealSense D435  -OR-  ArduCam (mutually exclusive — different physical setups)
    4. Run experiment sequence: move gantry -> capture data -> repeat
    5. Post-process: align timestamps, store structured data
"""

import sys
import os
import csv
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np

# PIL for converting numpy arrays -> Tkinter-displayable images
from PIL import Image, ImageTk

from config import load_config

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib import cm as _mpl_cm
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

_SIM_PATH = os.path.join(os.path.dirname(__file__), "Experiments", "simulation")
if _SIM_PATH not in sys.path:
    sys.path.insert(0, _SIM_PATH)

try:
    from realsense_projection import RealSenseProjectionModel
    _PROJECTION_AVAILABLE = True
except ImportError:
    _PROJECTION_AVAILABLE = False

try:
    from gantry import GantryWorker as _GantryWorker
    _GRBL_AVAILABLE = True
except ImportError:
    _GRBL_AVAILABLE = False

try:
    from ground_track import (
        MapCalibration, DemRegionBounds, GroundTrack,
        generate_synthetic_pass, generate_synthetic_pass_normalized,
        load_csv as _load_gt_csv,
        TrackRunner,
    )
    _GT_AVAILABLE = True
except ImportError:
    _GT_AVAILABLE = False

# ── ORBStuff import ───────────────────────────────────────────────────────────
try:
    from Experiments.ORBStuff import getPosePixelCords
    _ORB_AVAILABLE = True
except ImportError:
    _ORB_AVAILABLE = False

# ── Display constants ─────────────────────────────────────────────────────────
RS_W         = 640   # RealSense D435 native resolution
RS_H         = 480
ARDUCAM_W    = 640
ARDUCAM_H    = 480
POLL_MS      = 50    # GUI refresh interval (20 Hz)

# ── ORB map display size ──────────────────────────────────────────────────────
ORB_MAP_W    = 720   # width of the basemap canvas in the UI
ORB_MAP_H    = 360   # height (2:1 matches a standard equirectangular Mars map)

# ── Recording defaults ────────────────────────────────────────────────────────
DEFAULT_RECORD_FPS   = 5       # frames per second saved during recording
SESSIONS_ROOT        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")

# ── Map coordinate constants ──────────────────────────────────────────────────
# Top-left corner of the physical map = (lat=90°, lon=0°)
# Bottom-right corner of the physical map = (lat=-90°, lon=360°)
MAP_TL_LAT =  90.0
MAP_TL_LON =   0.0
MAP_BR_LAT = -90.0
MAP_BR_LON = 360.0


# ── Background worker: reads RealSense D435 frames off the main thread ────────
class RealSenseWorker(threading.Thread):
    def __init__(self, pipeline, frame_q: queue.Queue):
        super().__init__(daemon=True)
        self.pipeline   = pipeline
        self.frame_q    = frame_q
        self._stop      = threading.Event()
        self._colorizer = rs.colorizer()
        self._align     = rs.align(rs.stream.color)

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                aligned      = self._align.process(frames)
                depth_frame  = aligned.get_depth_frame()
                color_frame  = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_colorized = np.asanyarray(
                    self._colorizer.colorize(depth_frame).get_data()
                )   # (H, W, 3) uint8

                color_rgb = np.asanyarray(color_frame.get_data())[:, :, ::-1].copy()
                # color_frame is BGR from the SDK; reverse channel order → RGB

                depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
                depth_m   = depth_raw / 1000.0        # uint16 mm → float32 m
                depth_m[depth_m == 0] = np.nan        # mark invalid pixels

                if not self.frame_q.full():
                    self.frame_q.put_nowait((depth_colorized, color_rgb, depth_m))
            except Exception as e:
                self.frame_q.put_nowait(("error", str(e)))
                break


# ── Background worker: runs ORB on a single frame then signals done ───────────
class ORBWorker(threading.Thread):
    """
    Receives one grayscale/BGR frame, rotates it 90° CW, calls
    getPosePixelCords(), and puts the result onto result_q.

    result_q items are either:
        ("ok",  [cx, cy], [ref_w, ref_h])   – success
        ("err", error_string)               – failure
    """

    def __init__(self, frame: np.ndarray, result_q: queue.Queue):
        super().__init__(daemon=True)
        self.frame    = frame
        self.result_q = result_q

    def run(self):
        try:
            # Rotate 90° clockwise: transpose then flip horizontally
            rotated = cv2.rotate(self.frame, cv2.ROTATE_90_CLOCKWISE)
            # getPosePixelCords expects a grayscale numpy array
            if rotated.ndim == 3:
                gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
            else:
                gray = rotated
            result = getPosePixelCords(gray)
            if result is None:
                self.result_q.put_nowait(("err", "getPosePixelCords returned None"))
                return
            center, ref_res = result
            self.result_q.put_nowait(("ok", center, ref_res))
        except Exception as e:
            self.result_q.put_nowait(("err", str(e)))


# ── Main application ──────────────────────────────────────────────────────────
class ExperimentApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("OMNIScience Experiment Control Panel")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)

        self._cfg = load_config()

        # Device handles
        self._rs_pipeline  = None
        self._rs_worker    = None
        self._rs_frame_q   = queue.Queue(maxsize=2)
        self._rs_serial    = ""        # serial number shown after connect
        self._arducam      = None

        # Gantry
        self._gantry_worker  = None
        self._gantry_resp_q  = queue.Queue()
        self._gantry_state_var = tk.StringVar(value="—")
        self._gantry_x_var     = tk.StringVar(value="—")
        self._gantry_y_var     = tk.StringVar(value="—")
        self._gantry_z_var     = tk.StringVar(value="—")
        self._jog_step_var     = tk.DoubleVar(value=1.0)

        # Ground track / map calibration
        self._map_corner_br   = None              # (x_mm, y_mm)
        self._map_corner_tl   = None              # (x_mm, y_mm)
        self._map_br_var      = tk.StringVar(value="—")
        self._map_tl_var      = tk.StringVar(value="—")
        self._map_size_var    = tk.StringVar(value="—")
        self._gt_source_var   = tk.StringVar(value="synthetic")
        self._gt_csv_path_var = tk.StringVar(value="")
        self._gt_n_var        = tk.StringVar(value="50")
        self._gt_duration_var = tk.StringVar(value="120")
        self._gt_cycles_var   = tk.StringVar(value="1.5")
        self._gt_speed_var    = tk.StringVar(value="3000")
        self._gt_dry_run_var  = tk.BooleanVar(value=True)
        self._ground_track    = None
        self._track_runner    = None

        # Snap & Visualize state
        self._last_depth_m   = None   # most-recent Z map (float32, metres, H×W)
        self._last_grayscale = None   # most-recent grayscale image (uint8, H×W)
        self._last_depth_rgb = None   # most-recent depth colormap image (uint8, H×W×3)
        self._snap_window    = None   # Toplevel for debug snapshot (replaced on each snap)

        # ── ArduCam focus state ───────────────────────────────────────────────
        self._arducam_autofocus_var = tk.BooleanVar(value=True)
        self._arducam_focus_var     = tk.StringVar(value="500")

        # ── ArduCam FPS control ───────────────────────────────────────────────
        self._record_fps_var     = tk.IntVar(value=DEFAULT_RECORD_FPS)
        self._last_capture_time  = 0.0   # monotonic time of last saved frame

        # ── Recording state ───────────────────────────────────────────────────
        self._recording          = False
        self._session_dir        = None   # e.g. ./sessions/20260406_143022/
        self._frames_dir         = None   # session_dir/frames/
        self._session_csv_path   = None   # session_dir/session.csv
        self._session_csv_file   = None   # open file handle
        self._session_csv_writer = None   # csv.writer
        self._record_frame_count = 0
        self._record_count_var   = tk.StringVar(value="0")
        self._record_status_var  = tk.StringVar(value="Idle")
        self._session_path_var   = tk.StringVar(value="—")

        # ── ORB localization state ────────────────────────────────────────────
        self._orb_running      = False
        self._orb_worker       = None
        self._orb_result_q     = queue.Queue()
        self._orb_history_map  = []             # list of (map_px_x, map_px_y)
        self._orb_history_gantry = []           # list of (gantry_mm_x, gantry_mm_y)
        self._orb_ref_img_path = tk.StringVar(value="Experiments/mars_4k_color.jpg")
        self._orb_map_photo    = None
        self._orb_est_lat_var  = tk.StringVar(value="—")
        self._orb_est_lon_var  = tk.StringVar(value="—")
        self._orb_est_gx_var   = tk.StringVar(value="—")
        self._orb_est_gy_var   = tk.StringVar(value="—")
        self._orb_gt_gx_var    = tk.StringVar(value="—")
        self._orb_gt_gy_var    = tk.StringVar(value="—")
        self._orb_status_var   = tk.StringVar(value="Idle")
        self._orb_frame_count  = 0
        self._orb_count_var    = tk.StringVar(value="0")

        # ── ORB map video recording state ─────────────────────────────────────
        self._orb_video_recording   = False          # toggle flag
        self._orb_video_writer      = None           # cv2.VideoWriter or None
        self._orb_video_path        = None           # path to current video file
        self._orb_video_fps_var     = tk.IntVar(value=10)   # target FPS for video
        self._orb_video_frame_count = 0
        self._orb_video_count_var   = tk.StringVar(value="0")
        self._orb_video_status_var  = tk.StringVar(value="Idle")
        self._last_orb_video_time   = 0.0            # throttle video frame capture

        # Per-device status StringVars, populated in _build_status_bar
        self._status_color: dict[str, tk.StringVar] = {}
        self._status_text:  dict[str, tk.StringVar] = {}

        self._build_ui()
        self._after_id = self.after(POLL_MS, self._poll_cameras)

    # ── Coordinate conversion helpers ─────────────────────────────────────────

    def _gantry_mm_to_lat_lon(self, gx_mm: float, gy_mm: float):
        """
        Convert a gantry position (mm) to (lat_deg, lon_deg) using the same
        axis mapping and coordinate convention as the live ORB pipeline:

            _gantry_mm_to_canvas:   canvas-X ← gy_mm,  canvas-Y ← gx_mm
            _ref_px_to_lon_lat:     lon = fx*360 - 180  (-180° to +180°)
                                    lat = 90 - fy*180   (+90° to -90°)

        Returns (lat_deg, lon_deg) or (nan, nan) if calibration is absent.
        """
        if self._map_corner_tl is None or self._map_corner_br is None:
            return float("nan"), float("nan")

        tl_x, tl_y = self._map_corner_tl
        br_x, br_y = self._map_corner_br
        span_x = br_x - tl_x
        span_y = br_y - tl_y

        if abs(span_x) < 1e-6 or abs(span_y) < 1e-6:
            return float("nan"), float("nan")

        # Mirror _gantry_mm_to_canvas exactly:
        #   canvas X  ← gy_mm  (so gy drives longitude)
        #   canvas Y  ← gx_mm  (so gx drives latitude)
        fx = (gy_mm - tl_y) / span_y   # fraction across canvas X  →  longitude
        fy = (gx_mm - tl_x) / span_x   # fraction across canvas Y  →  latitude

        # Mirror _ref_px_to_lon_lat exactly:
        #   lon = fx * 360 - 180   →  -180° to +180°
        #   lat = 90 - fy * 180    →  +90° to -90°
        lon_deg = fx * 360.0 - 180.0
        lat_deg = 90.0 - fy * 180.0

        return lat_deg, lon_deg

    # ── Top-level layout ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.minsize(700, 400)
        self._build_status_bar()
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)
        self._build_notebook()
        self._build_log()

    # ── Scrollable tab factory ────────────────────────────────────────────────

    def _make_scrollable_tab(self) -> tuple:
        outer = tk.Frame(self._nb, bg="#1e1e2e")
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        sc = tk.Canvas(outer, bg="#1e1e2e", highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=sc.yview)
        sc.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky="ns")
        sc.grid(row=0, column=0, sticky="nsew")

        inner = tk.Frame(sc, bg="#1e1e2e")
        win_id = sc.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_cfg(e):
            sc.configure(scrollregion=sc.bbox("all"))

        def _on_canvas_cfg(e):
            sc.itemconfig(win_id, width=e.width)

        inner.bind("<Configure>", _on_inner_cfg)
        sc.bind("<Configure>", _on_canvas_cfg)

        def _on_wheel(e):
            if e.num == 4:
                sc.yview_scroll(-1, "units")
            elif e.num == 5:
                sc.yview_scroll(1, "units")
            else:
                sc.yview_scroll(int(-1 * (e.delta / 120)), "units")

        for w in (sc, inner):
            w.bind("<MouseWheel>", _on_wheel)
            w.bind("<Button-4>",   _on_wheel)
            w.bind("<Button-5>",   _on_wheel)

        return outer, inner

    # ── Status bar ───────────────────────────────────────────────────────────

    def _build_status_bar(self):
        bar = tk.Frame(self, bg="#11111b", padx=10, pady=6)
        bar.pack(fill=tk.X)

        tk.Label(bar, text="OMNIScience",
                 bg="#11111b", fg="#cdd6f4",
                 font=("Consolas", 12, "bold")).pack(side=tk.LEFT, padx=(0, 24))

        for device in ("LiDAR", "ArduCam", "Gantry"):
            self._status_color[device] = tk.StringVar(value="#f38ba8")
            self._status_text[device]  = tk.StringVar(value="Disconnected")
            self._make_status_indicator(bar, device)

    def _make_status_indicator(self, parent, device: str):
        frame = tk.Frame(parent, bg="#11111b")
        frame.pack(side=tk.LEFT, padx=14)

        canvas = tk.Canvas(frame, width=10, height=10,
                           bg="#11111b", highlightthickness=0)
        canvas.pack(side=tk.LEFT, padx=(0, 5))
        oval = canvas.create_oval(1, 1, 9, 9,
                                  fill=self._status_color[device].get(),
                                  outline="")

        def _refresh(*_):
            canvas.itemconfig(oval, fill=self._status_color[device].get())
        self._status_color[device].trace_add("write", _refresh)

        tk.Label(frame, text=f"{device}:", bg="#11111b", fg="#a6adc8",
                 font=("Consolas", 9)).pack(side=tk.LEFT)
        tk.Label(frame, textvariable=self._status_text[device],
                 bg="#11111b", fg="#cdd6f4",
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=(3, 0))

    # ── Notebook ──────────────────────────────────────────────────────────────

    def _build_notebook(self):
        style = ttk.Style()
        style.configure("TNotebook",     background="#1e1e2e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#313244", foreground="#cdd6f4",
                        padding=[12, 5], font=("Consolas", 9))
        style.map("TNotebook.Tab",
                  background=[("selected", "#89b4fa"), ("active", "#45475a")],
                  foreground=[("selected", "#1e1e2e")])

        outer = tk.Frame(self, bg="#1e1e2e")
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        self._nb = ttk.Notebook(outer)
        self._nb.pack(fill=tk.BOTH, expand=True)

        self._nb.add(self._tab_run(),     text="  Run  ")
        self._nb.add(self._tab_lidar(),   text="  LiDAR  ")
        self._nb.add(self._tab_arducam(), text="  ArduCam  ")
        self._nb.add(self._tab_gantry(),  text="  Gantry  ")

    # ── Run tab ───────────────────────────────────────────────────────────────

    def _tab_run(self) -> tk.Frame:
        tab, tab_content = self._make_scrollable_tab()

        cal_sec = self._section(tab_content, "Map Calibration")
        tk.Label(cal_sec,
                 text="Align ArduCam centre over the map corner, then click Record.",
                 bg="#181825", fg="#a6adc8", font=("Consolas", 8)
                 ).pack(anchor="w", padx=10, pady=(6, 2))

        btn_row = tk.Frame(cal_sec, bg="#181825")
        btn_row.pack(anchor="w", padx=10, pady=(2, 4))
        tk.Button(btn_row, text="Record Bottom-Right",
                  command=self._record_corner_br,
                  **self._btn(fg="#89b4fa")).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(btn_row, text="Record Top-Left",
                  command=self._record_corner_tl,
                  **self._btn(fg="#89b4fa")).pack(side=tk.LEFT)

        pos_row = tk.Frame(cal_sec, bg="#181825")
        pos_row.pack(anchor="w", padx=10, pady=(0, 6))
        for label, var in [("BR:", self._map_br_var),
                           ("TL:", self._map_tl_var),
                           ("Map:", self._map_size_var)]:
            tk.Label(pos_row, text=label, bg="#181825", fg="#585b70",
                     font=("Consolas", 8)).pack(side=tk.LEFT)
            tk.Label(pos_row, textvariable=var, bg="#181825", fg="#cdd6f4",
                     font=("Consolas", 8), width=22, anchor="w"
                     ).pack(side=tk.LEFT, padx=(0, 12))

        gt_sec = self._section(tab_content, "Ground Track")

        src_row = tk.Frame(gt_sec, bg="#181825")
        src_row.pack(anchor="w", padx=10, pady=(6, 4))
        for text, val in [("Synthetic pass", "synthetic"), ("Load CSV", "csv")]:
            tk.Radiobutton(src_row, text=text, variable=self._gt_source_var,
                           value=val, command=self._on_gt_source_change,
                           bg="#181825", fg="#cdd6f4", selectcolor="#313244",
                           activebackground="#181825", font=("Consolas", 9)
                           ).pack(side=tk.LEFT, padx=(0, 16))

        sub_container = tk.Frame(gt_sec, bg="#181825")
        sub_container.pack(anchor="w", padx=10, pady=(0, 2))

        self._gt_synth_frame = tk.Frame(sub_container, bg="#181825")
        self._gt_synth_frame.pack(anchor="w")
        for label, var, w in [("Duration (s):", self._gt_duration_var, 6),
                               ("N waypoints:", self._gt_n_var, 5),
                               ("Sine cycles:", self._gt_cycles_var, 4)]:
            tk.Label(self._gt_synth_frame, text=label, bg="#181825", fg="#a6adc8",
                     font=("Consolas", 9)).pack(side=tk.LEFT)
            tk.Entry(self._gt_synth_frame, textvariable=var, width=w,
                     bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                     font=("Consolas", 9), relief=tk.FLAT
                     ).pack(side=tk.LEFT, padx=(3, 14))

        self._gt_csv_frame = tk.Frame(sub_container, bg="#181825")
        tk.Label(self._gt_csv_frame, text="File:", bg="#181825", fg="#a6adc8",
                 font=("Consolas", 9)).pack(side=tk.LEFT)
        tk.Entry(self._gt_csv_frame, textvariable=self._gt_csv_path_var, width=38,
                 bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                 font=("Consolas", 9), relief=tk.FLAT
                 ).pack(side=tk.LEFT, padx=(3, 6))
        tk.Button(self._gt_csv_frame, text="Browse",
                  command=self._browse_gt_csv,
                  **self._btn()).pack(side=tk.LEFT)

        load_row = tk.Frame(gt_sec, bg="#181825")
        load_row.pack(anchor="w", padx=10, pady=(4, 6))
        tk.Button(load_row, text="Generate / Load",
                  command=self._load_ground_track,
                  **self._btn(fg="#a6e3a1")).pack(side=tk.LEFT)
        self._gt_status_var = tk.StringVar(value="No track loaded.")
        tk.Label(load_row, textvariable=self._gt_status_var,
                 bg="#181825", fg="#585b70", font=("Consolas", 8)
                 ).pack(side=tk.LEFT, padx=(12, 0))

        run_sec = self._section(tab_content, "Run Settings")

        settings_row = tk.Frame(run_sec, bg="#181825")
        settings_row.pack(anchor="w", padx=10, pady=(6, 4))
        tk.Label(settings_row, text="Speed mm/min:", bg="#181825", fg="#a6adc8",
                 font=("Consolas", 9)).pack(side=tk.LEFT)
        tk.Entry(settings_row, textvariable=self._gt_speed_var, width=7,
                 bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                 font=("Consolas", 9), relief=tk.FLAT
                 ).pack(side=tk.LEFT, padx=(4, 16))
        tk.Checkbutton(settings_row, text="Dry Run",
                       variable=self._gt_dry_run_var,
                       bg="#181825", fg="#cdd6f4", selectcolor="#313244",
                       activebackground="#181825", font=("Consolas", 9)
                       ).pack(side=tk.LEFT)

        ctrl_row = tk.Frame(run_sec, bg="#181825")
        ctrl_row.pack(anchor="w", padx=10, pady=(2, 8))
        self._start_btn = tk.Button(
            ctrl_row, text="▶  Start",
            command=self._start_experiment,
            bg="#1e6640", fg="#a6e3a1", activebackground="#2a7a50",
            font=("Consolas", 9, "bold"), relief=tk.FLAT, padx=12, pady=5, width=12)
        self._start_btn.pack(side=tk.LEFT, padx=(0, 6))
        self._stop_btn = tk.Button(
            ctrl_row, text="■  Stop",
            command=self._stop_experiment,
            bg="#6e2020", fg="#f38ba8", activebackground="#7e3030",
            font=("Consolas", 9, "bold"), relief=tk.FLAT, padx=12, pady=5, width=12,
            state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT)

        return tab

    # ── LiDAR tab ─────────────────────────────────────────────────────────────

    def _tab_lidar(self) -> tk.Frame:
        tab, tab_content = self._make_scrollable_tab()

        conn = self._section(tab_content, "Connection")

        tk.Button(conn, text="Enumerate",  command=self._rs_enumerate,
                  **self._btn()).grid(row=0, column=0, padx=(8, 4), pady=6)
        tk.Button(conn, text="Connect",    command=self._connect_rs,
                  **self._btn(fg="#a6e3a1")).grid(row=0, column=1, padx=4)
        tk.Button(conn, text="Disconnect", command=self._disconnect_rs,
                  **self._btn(fg="#f38ba8")).grid(row=0, column=2, padx=(4, 8))

        self._rs_serial_var = tk.StringVar(value="—")
        tk.Label(conn, text="Device:", bg="#181825", fg="#a6adc8",
                 font=("Consolas", 9)).grid(row=0, column=3, padx=(12, 4), pady=6)
        tk.Label(conn, textvariable=self._rs_serial_var,
                 bg="#181825", fg="#cdd6f4",
                 font=("Consolas", 9)).grid(row=0, column=4, padx=(0, 8), pady=6)

        settings = self._section(tab_content, "Settings")

        _r = self._cfg.realsense
        fields = [
            ("Emitter Power", str(_r.emitter_power), "0 – 360"),
            ("Exposure (µs)",  str(_r.exposure_us),  "0 = auto"),
        ]
        self._rs_settings: dict[str, tk.StringVar] = {}
        for i, (label, default, hint) in enumerate(fields):
            tk.Label(settings, text=label + ":", bg="#181825", fg="#a6adc8",
                     font=("Consolas", 9), anchor="e", width=16
                     ).grid(row=i, column=0, padx=(8, 4), pady=3, sticky="e")
            var = tk.StringVar(value=default)
            self._rs_settings[label] = var
            tk.Entry(settings, textvariable=var, width=8,
                     bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                     font=("Consolas", 9), relief=tk.FLAT
                     ).grid(row=i, column=1, padx=4, pady=3, sticky="w")
            tk.Label(settings, text=hint, bg="#181825", fg="#585b70",
                     font=("Consolas", 8)
                     ).grid(row=i, column=2, padx=4, sticky="w")

        tk.Button(settings, text="Apply Settings", command=self._apply_rs_settings,
                  **self._btn()).grid(row=len(fields), column=0, columnspan=3,
                                     padx=8, pady=(6, 4), sticky="w")

        preview = self._section(tab_content, "Preview")
        preview.columnconfigure(0, weight=1)
        preview.columnconfigure(1, weight=1)

        tk.Label(preview, text="Depth Map", bg="#181825", fg="#89b4fa",
                 font=("Consolas", 8)).grid(row=0, column=0, pady=(4, 2))
        tk.Label(preview, text="Color (RGB)", bg="#181825", fg="#89b4fa",
                 font=("Consolas", 8)).grid(row=0, column=1, pady=(4, 2))

        self._depth_canvas = tk.Canvas(preview, width=RS_W, height=RS_H,
                                       bg="#11111b", highlightthickness=1,
                                       highlightbackground="#313244")
        self._depth_canvas.grid(row=1, column=0, padx=6, pady=(0, 6))

        self._gray_canvas = tk.Canvas(preview, width=RS_W, height=RS_H,
                                      bg="#11111b", highlightthickness=1,
                                      highlightbackground="#313244")
        self._gray_canvas.grid(row=1, column=1, padx=6, pady=(0, 6))

        snap_sec = self._section(tab_content, "Snap & Visualize")

        tk.Label(snap_sec, text="Altitude (mm):", bg="#181825", fg="#a6adc8",
                 font=("Consolas", 9)).grid(row=0, column=0, padx=(8, 4), pady=6)

        self._snap_alt_var = tk.StringVar(value=str(self._cfg.gantry.camera_altitude_mm))
        tk.Entry(snap_sec, textvariable=self._snap_alt_var, width=8,
                 bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                 font=("Consolas", 9), relief=tk.FLAT
                 ).grid(row=0, column=1, padx=4, pady=6)

        tk.Button(snap_sec, text="Snap & Plot 3D", command=self._snap_and_plot,
                  **self._btn(fg="#89b4fa")).grid(row=0, column=2, padx=(8, 8), pady=6)

        return tab

    # ── ArduCam tab ───────────────────────────────────────────────────────────

    def _tab_arducam(self) -> tk.Frame:
        tab = tk.Frame(self._nb, bg="#1e1e2e")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)

        _scroll_canvas = tk.Canvas(tab, bg="#1e1e2e", highlightthickness=0)
        _scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL,
                                   command=_scroll_canvas.yview)
        _scroll_canvas.configure(yscrollcommand=_scrollbar.set)
        _scrollbar.grid(row=0, column=1, sticky="ns")
        _scroll_canvas.grid(row=0, column=0, sticky="nsew")

        inner = tk.Frame(_scroll_canvas, bg="#1e1e2e")
        _inner_id = _scroll_canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(event):
            _scroll_canvas.configure(scrollregion=_scroll_canvas.bbox("all"))

        def _on_canvas_configure(event):
            _scroll_canvas.itemconfig(_inner_id, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        _scroll_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            if event.num == 4:
                _scroll_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                _scroll_canvas.yview_scroll(1, "units")
            else:
                _scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        _scroll_canvas.bind("<MouseWheel>", _on_mousewheel)
        _scroll_canvas.bind("<Button-4>",   _on_mousewheel)
        _scroll_canvas.bind("<Button-5>",   _on_mousewheel)
        inner.bind("<MouseWheel>", _on_mousewheel)
        inner.bind("<Button-4>",   _on_mousewheel)
        inner.bind("<Button-5>",   _on_mousewheel)

        tab_content = inner

        # ── Connection ────────────────────────────────────────────────────
        conn = self._section(tab_content, "Connection")
        tk.Label(conn, text="Camera index:", bg="#181825", fg="#a6adc8",
                 font=("Consolas", 9)).grid(row=0, column=0, padx=(8, 4), pady=6)

        self._arducam_idx_var = tk.StringVar(value=str(self._cfg.arducam.camera_index))
        tk.Entry(conn, textvariable=self._arducam_idx_var, width=4,
                 bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                 font=("Consolas", 9), relief=tk.FLAT
                 ).grid(row=0, column=1, padx=4)
        tk.Button(conn, text="Connect",    command=self._connect_arducam,
                  **self._btn(fg="#a6e3a1")).grid(row=0, column=2, padx=4)
        tk.Button(conn, text="Disconnect", command=self._disconnect_arducam,
                  **self._btn(fg="#f38ba8")).grid(row=0, column=3, padx=(4, 8))

        # ── Recording section ─────────────────────────────────────────────
        rec_sec = self._section(tab_content, "Recording")

        rec_ctrl_row = tk.Frame(rec_sec, bg="#181825")
        rec_ctrl_row.pack(fill=tk.X, padx=10, pady=(8, 4))

        self._rec_toggle_btn = tk.Button(
            rec_ctrl_row, text="⏺  Start Recording",
            command=self._toggle_recording,
            bg="#1e6640", fg="#a6e3a1", activebackground="#2a7a50",
            font=("Consolas", 9, "bold"), relief=tk.FLAT, padx=12, pady=5, width=18)
        self._rec_toggle_btn.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(rec_ctrl_row, text="Status:", bg="#181825", fg="#585b70",
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        self._rec_status_lbl = tk.Label(
            rec_ctrl_row, textvariable=self._record_status_var,
            bg="#181825", fg="#585b70",
            font=("Consolas", 8, "bold"), width=10, anchor="w")
        self._rec_status_lbl.pack(side=tk.LEFT, padx=(3, 20))

        tk.Label(rec_ctrl_row, text="Saved:", bg="#181825", fg="#585b70",
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        tk.Label(rec_ctrl_row, textvariable=self._record_count_var,
                 bg="#181825", fg="#cdd6f4",
                 font=("Consolas", 8, "bold"), width=7, anchor="w"
                 ).pack(side=tk.LEFT, padx=(3, 4))
        tk.Label(rec_ctrl_row, text="frames", bg="#181825", fg="#585b70",
                 font=("Consolas", 8)).pack(side=tk.LEFT)

        fps_row = tk.Frame(rec_sec, bg="#181825")
        fps_row.pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Label(fps_row, text="Capture FPS:", bg="#181825", fg="#a6adc8",
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=(0, 6))

        self._fps_slider = tk.Scale(
            fps_row, from_=1, to=30,
            orient=tk.HORIZONTAL, length=200,
            variable=self._record_fps_var,
            command=self._on_fps_slider,
            bg="#181825", fg="#cdd6f4",
            troughcolor="#313244", activebackground="#45475a",
            highlightthickness=0, relief=tk.FLAT,
            font=("Consolas", 8))
        self._fps_slider.pack(side=tk.LEFT, padx=(0, 6))

        self._fps_entry = tk.Entry(
            fps_row, textvariable=self._record_fps_var, width=4,
            bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
            font=("Consolas", 9), relief=tk.FLAT)
        self._fps_entry.pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(fps_row, text="fps", bg="#181825", fg="#585b70",
                 font=("Consolas", 8)).pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(fps_row,
                 text="(GUI preview always runs at 20 Hz regardless)",
                 bg="#181825", fg="#585b70",
                 font=("Consolas", 8, "italic")).pack(side=tk.LEFT)

        path_row = tk.Frame(rec_sec, bg="#181825")
        path_row.pack(fill=tk.X, padx=10, pady=(0, 8))

        tk.Label(path_row, text="Session:", bg="#181825", fg="#585b70",
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        tk.Label(path_row, textvariable=self._session_path_var,
                 bg="#181825", fg="#89b4fa",
                 font=("Consolas", 8), anchor="w"
                 ).pack(side=tk.LEFT, padx=(6, 0))

        # ── Focus controls ────────────────────────────────────────────────
        focus_sec = self._section(tab_content, "Focus")

        focus_row = tk.Frame(focus_sec, bg="#181825")
        focus_row.pack(fill=tk.X, padx=10, pady=(8, 4))

        self._af_btn = tk.Button(
            focus_row, text="⟳  Autofocus: ON",
            command=self._toggle_autofocus,
            bg="#1e4466", fg="#89b4fa", activebackground="#2a5070",
            font=("Consolas", 9, "bold"), relief=tk.FLAT, padx=10, pady=4)
        self._af_btn.pack(side=tk.LEFT, padx=(0, 16))

        tk.Button(focus_row, text="Trigger AF",
                  command=self._trigger_autofocus,
                  **self._btn(fg="#89b4fa")).pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(focus_row, text="Manual focus:", bg="#181825", fg="#a6adc8",
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=(0, 6))

        self._focus_slider = tk.Scale(
            focus_row, from_=0, to=1023,
            orient=tk.HORIZONTAL, length=220,
            variable=self._arducam_focus_var,
            command=self._on_focus_slider,
            bg="#181825", fg="#cdd6f4",
            troughcolor="#313244", activebackground="#45475a",
            highlightthickness=0, relief=tk.FLAT,
            font=("Consolas", 8))
        self._focus_slider.pack(side=tk.LEFT, padx=(0, 6))

        tk.Entry(focus_row, textvariable=self._arducam_focus_var, width=5,
                 bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                 font=("Consolas", 9), relief=tk.FLAT
                 ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(focus_row, text="Set",
                  command=self._apply_manual_focus,
                  **self._btn()).pack(side=tk.LEFT)

        focus_info = tk.Frame(focus_sec, bg="#181825")
        focus_info.pack(fill=tk.X, padx=10, pady=(0, 8))
        tk.Label(focus_info, text="Current:", bg="#181825", fg="#585b70",
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        self._focus_readout_var = tk.StringVar(value="—")
        tk.Label(focus_info, textvariable=self._focus_readout_var,
                 bg="#181825", fg="#cdd6f4",
                 font=("Consolas", 8, "bold"), width=6, anchor="w"
                 ).pack(side=tk.LEFT, padx=(3, 16))
        tk.Label(focus_info,
                 text="Range 0–1023.  Autofocus must be OFF for manual focus to take effect.",
                 bg="#181825", fg="#585b70",
                 font=("Consolas", 8, "italic")).pack(side=tk.LEFT)

        # ── Preview ───────────────────────────────────────────────────────
        preview = self._section(tab_content, "Preview")
        tk.Label(preview, text="RGB Feed", bg="#181825", fg="#89b4fa",
                 font=("Consolas", 8)).pack(pady=(4, 2))
        self._arducam_canvas = tk.Canvas(preview, width=ARDUCAM_W, height=ARDUCAM_H,
                                         bg="#11111b", highlightthickness=1,
                                         highlightbackground="#313244")
        self._arducam_canvas.pack(padx=6, pady=(0, 6))

        # ── ORB Localization ──────────────────────────────────────────────
        orb_sec = self._section(tab_content, "ORB Localization")

        ref_row = tk.Frame(orb_sec, bg="#181825")
        ref_row.pack(fill=tk.X, padx=10, pady=(8, 4))

        tk.Label(ref_row, text="Reference map:", bg="#181825", fg="#a6adc8",
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=(0, 6))
        tk.Entry(ref_row, textvariable=self._orb_ref_img_path, width=36,
                 bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                 font=("Consolas", 9), relief=tk.FLAT
                 ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(ref_row, text="Browse",
                  command=self._browse_orb_ref,
                  **self._btn()).pack(side=tk.LEFT, padx=(0, 16))

        self._orb_start_btn = tk.Button(
            ref_row, text="▶  Start ORB",
            command=self._start_orb,
            bg="#1e6640", fg="#a6e3a1", activebackground="#2a7a50",
            font=("Consolas", 9, "bold"), relief=tk.FLAT, padx=10, pady=4)
        self._orb_start_btn.pack(side=tk.LEFT, padx=(0, 4))

        self._orb_stop_btn = tk.Button(
            ref_row, text="■  Stop",
            command=self._stop_orb,
            bg="#6e2020", fg="#f38ba8", activebackground="#7e3030",
            font=("Consolas", 9, "bold"), relief=tk.FLAT, padx=10, pady=4,
            state=tk.DISABLED)
        self._orb_stop_btn.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(ref_row, text="Clear History",
                  command=self._orb_clear_history,
                  **self._btn(fg="#fab387")).pack(side=tk.LEFT)

        info_row = tk.Frame(orb_sec, bg="#181825")
        info_row.pack(fill=tk.X, padx=10, pady=(0, 6))

        tk.Label(info_row, text="Status:", bg="#181825", fg="#585b70",
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        self._orb_status_lbl = tk.Label(
            info_row, textvariable=self._orb_status_var,
            bg="#181825", fg="#a6adc8",
            font=("Consolas", 8, "bold"), width=14, anchor="w")
        self._orb_status_lbl.pack(side=tk.LEFT, padx=(3, 16))

        tk.Label(info_row, text="Frames:", bg="#181825", fg="#585b70",
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        tk.Label(info_row, textvariable=self._orb_count_var,
                 bg="#181825", fg="#cdd6f4",
                 font=("Consolas", 8), width=6, anchor="w"
                 ).pack(side=tk.LEFT, padx=(3, 20))

        readouts = [
            ("Est Lon:",  self._orb_est_lon_var,  "#89b4fa"),
            ("Est Lat:",  self._orb_est_lat_var,  "#89b4fa"),
            ("Est X mm:", self._orb_est_gx_var,   "#cba6f7"),
            ("Est Y mm:", self._orb_est_gy_var,   "#cba6f7"),
            ("GT X mm:",  self._orb_gt_gx_var,    "#a6e3a1"),
            ("GT Y mm:",  self._orb_gt_gy_var,    "#a6e3a1"),
        ]
        for lbl_text, var, fg_col in readouts:
            tk.Label(info_row, text=lbl_text, bg="#181825", fg="#585b70",
                     font=("Consolas", 8)).pack(side=tk.LEFT, padx=(0, 2))
            tk.Label(info_row, textvariable=var,
                     bg="#181825", fg=fg_col,
                     font=("Consolas", 8, "bold"), width=9, anchor="w"
                     ).pack(side=tk.LEFT, padx=(0, 10))

        # ── ORB Map Video Recording controls ─────────────────────────────
        vid_row = tk.Frame(orb_sec, bg="#181825")
        vid_row.pack(fill=tk.X, padx=10, pady=(0, 6))

        self._orb_vid_btn = tk.Button(
            vid_row, text="⏺  Record Map Video",
            command=self._toggle_orb_video,
            bg="#1e6640", fg="#a6e3a1", activebackground="#2a7a50",
            font=("Consolas", 9, "bold"), relief=tk.FLAT, padx=10, pady=4, width=20)
        self._orb_vid_btn.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(vid_row, text="Video FPS:", bg="#181825", fg="#a6adc8",
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(vid_row, from_=1, to=30, orient=tk.HORIZONTAL, length=120,
                 variable=self._orb_video_fps_var,
                 bg="#181825", fg="#cdd6f4", troughcolor="#313244",
                 activebackground="#45475a", highlightthickness=0, relief=tk.FLAT,
                 font=("Consolas", 8)
                 ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(vid_row, textvariable=self._orb_video_fps_var, width=3,
                 bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                 font=("Consolas", 9), relief=tk.FLAT
                 ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(vid_row, text="fps", bg="#181825", fg="#585b70",
                 font=("Consolas", 8)).pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(vid_row, text="Status:", bg="#181825", fg="#585b70",
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        self._orb_vid_status_lbl = tk.Label(
            vid_row, textvariable=self._orb_video_status_var,
            bg="#181825", fg="#585b70",
            font=("Consolas", 8, "bold"), width=10, anchor="w")
        self._orb_vid_status_lbl.pack(side=tk.LEFT, padx=(3, 12))

        tk.Label(vid_row, text="Frames:", bg="#181825", fg="#585b70",
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        tk.Label(vid_row, textvariable=self._orb_video_count_var,
                 bg="#181825", fg="#cdd6f4",
                 font=("Consolas", 8), width=6, anchor="w"
                 ).pack(side=tk.LEFT, padx=(3, 0))

        vid_info = tk.Frame(orb_sec, bg="#181825")
        vid_info.pack(fill=tk.X, padx=10, pady=(0, 4))
        tk.Label(vid_info,
                 text="Video is saved to session_dir/orb_map_<timestamp>.mp4  "
                      "(ORB does not need to be running to record)",
                 bg="#181825", fg="#585b70",
                 font=("Consolas", 8, "italic")).pack(anchor="w")

        # ── Map canvas ────────────────────────────────────────────────────
        map_frame = tk.Frame(orb_sec, bg="#181825")
        map_frame.pack(padx=10, pady=(0, 10))

        self._orb_map_canvas = tk.Canvas(
            map_frame,
            width=ORB_MAP_W, height=ORB_MAP_H,
            bg="#11111b", highlightthickness=1,
            highlightbackground="#313244")
        self._orb_map_canvas.pack()

        self._orb_draw_placeholder()

        return tab

    # ── Gantry tab ────────────────────────────────────────────────────────────

    def _tab_gantry(self) -> tk.Frame:
        tab, tab_content = self._make_scrollable_tab()

        conn = self._section(tab_content, "Connection")

        tk.Label(conn, text="Port:", bg="#181825", fg="#a6adc8",
                 font=("Consolas", 9)).grid(row=0, column=0, padx=(8, 4), pady=6)
        self._gantry_port_var = tk.StringVar(
            value=self._cfg.gantry.serial_port or "COM3")
        tk.Entry(conn, textvariable=self._gantry_port_var, width=10,
                 bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                 font=("Consolas", 9), relief=tk.FLAT
                 ).grid(row=0, column=1, padx=4, pady=6)

        tk.Label(conn, text="Baud:", bg="#181825", fg="#a6adc8",
                 font=("Consolas", 9)).grid(row=0, column=2, padx=(8, 4), pady=6)
        self._gantry_baud_var = tk.StringVar(
            value=str(self._cfg.gantry.baud_rate or 115200))
        tk.Entry(conn, textvariable=self._gantry_baud_var, width=8,
                 bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                 font=("Consolas", 9), relief=tk.FLAT
                 ).grid(row=0, column=3, padx=4, pady=6)

        tk.Button(conn, text="Connect",    command=self._connect_gantry,
                  **self._btn(fg="#a6e3a1")).grid(row=0, column=4, padx=4)
        tk.Button(conn, text="Disconnect", command=self._disconnect_gantry,
                  **self._btn(fg="#f38ba8")).grid(row=0, column=5, padx=(4, 8))

        pos = self._section(tab_content, "Position")

        tk.Label(pos, text="State:", bg="#181825", fg="#a6adc8",
                 font=("Consolas", 9)).grid(row=0, column=0, padx=(8, 4), pady=8)
        self._gantry_state_lbl = tk.Label(
            pos, textvariable=self._gantry_state_var,
            bg="#181825", fg="#cdd6f4",
            font=("Consolas", 9, "bold"), width=8, anchor="w")
        self._gantry_state_lbl.grid(row=0, column=1, padx=(0, 8))

        for col, (axis, var) in enumerate(
            (("X", self._gantry_x_var),
             ("Y", self._gantry_y_var),
             ("Z", self._gantry_z_var)),
            start=1
        ):
            c = col * 2
            tk.Label(pos, text=f"{axis}:", bg="#181825", fg="#a6adc8",
                     font=("Consolas", 9)).grid(row=0, column=c, padx=(8, 2))
            tk.Label(pos, textvariable=var, bg="#181825", fg="#89b4fa",
                     font=("Consolas", 9, "bold"), width=10, anchor="e"
                     ).grid(row=0, column=c + 1, padx=(0, 4))

        jog = self._section(tab_content, "Jog")

        step_row = tk.Frame(jog, bg="#181825")
        step_row.pack(padx=8, pady=(6, 4), anchor="w")

        tk.Label(step_row, text="Step (mm):", bg="#181825", fg="#a6adc8",
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=(0, 6))
        self._jog_step_btns = {}
        for step in (0.1, 1.0, 10.0, 100.0):
            btn = tk.Button(
                step_row, text=f"{step:g}",
                command=lambda s=step: self._set_jog_step(s),
                font=("Consolas", 9), relief=tk.FLAT, padx=8, pady=3)
            btn.pack(side=tk.LEFT, padx=2)
            self._jog_step_btns[step] = btn

        tk.Label(step_row, text="   Feed (mm/min):", bg="#181825", fg="#a6adc8",
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=(8, 4))
        self._jog_feed_var = tk.StringVar(value="1000")
        tk.Entry(step_row, textvariable=self._jog_feed_var, width=6,
                 bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                 font=("Consolas", 9), relief=tk.FLAT).pack(side=tk.LEFT)

        self._update_jog_step_btns()

        dpad_outer = tk.Frame(jog, bg="#181825")
        dpad_outer.pack(padx=8, pady=(2, 4), anchor="w")

        JB = dict(font=("Consolas", 10, "bold"), relief=tk.FLAT,
                  padx=14, pady=8, bg="#313244", activebackground="#45475a")

        dpad = tk.Frame(dpad_outer, bg="#181825")
        dpad.pack(side=tk.LEFT, padx=(0, 24))

        tk.Button(dpad, text="+Y", fg="#cdd6f4",
                  command=lambda: self._jog("Y", +1), **JB
                  ).grid(row=0, column=1, padx=2, pady=2)
        tk.Button(dpad, text="-X", fg="#cdd6f4",
                  command=lambda: self._jog("X", -1), **JB
                  ).grid(row=1, column=0, padx=2, pady=2)
        tk.Label(dpad, text="·", bg="#181825", fg="#585b70",
                 font=("Consolas", 16)).grid(row=1, column=1, padx=2, pady=2)
        tk.Button(dpad, text="+X", fg="#cdd6f4",
                  command=lambda: self._jog("X", +1), **JB
                  ).grid(row=1, column=2, padx=2, pady=2)
        tk.Button(dpad, text="-Y", fg="#cdd6f4",
                  command=lambda: self._jog("Y", -1), **JB
                  ).grid(row=2, column=1, padx=2, pady=2)

        zpad = tk.Frame(dpad_outer, bg="#181825")
        zpad.pack(side=tk.LEFT)
        tk.Button(zpad, text="+Z", fg="#cba6f7",
                  command=lambda: self._jog("Z", +1), **JB).pack(pady=2)
        tk.Button(zpad, text="-Z", fg="#cba6f7",
                  command=lambda: self._jog("Z", -1), **JB).pack(pady=2)

        tk.Label(jog,
                 text="Arrow keys:  ← → = X   ↑ ↓ = Y   PgUp PgDn = Z",
                 bg="#181825", fg="#585b70",
                 font=("Consolas", 8, "italic")
                 ).pack(padx=10, pady=(0, 6), anchor="w")

        goto_sec = self._section(tab_content, "Go To")

        coord_row = tk.Frame(goto_sec, bg="#181825")
        coord_row.pack(padx=8, pady=(6, 2), anchor="w")

        for axis, var_name in (("X", "_goto_x_var"),
                               ("Y", "_goto_y_var"),
                               ("Z", "_goto_z_var")):
            setattr(self, var_name, tk.StringVar(value=""))
            tk.Label(coord_row, text=f"{axis}:", bg="#181825", fg="#a6adc8",
                     font=("Consolas", 9)).pack(side=tk.LEFT, padx=(0, 2))
            tk.Entry(coord_row, textvariable=getattr(self, var_name), width=8,
                     bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                     font=("Consolas", 9), relief=tk.FLAT
                     ).pack(side=tk.LEFT, padx=(0, 10))

        self._goto_rapid_var = tk.BooleanVar(value=True)
        self._goto_mode_btn = tk.Button(
            coord_row, text="G0  Rapid",
            command=self._toggle_goto_mode,
            bg="#1e6640", fg="#a6e3a1", activebackground="#2a7a50",
            font=("Consolas", 9, "bold"), relief=tk.FLAT, padx=10, pady=4)
        self._goto_mode_btn.pack(side=tk.LEFT, padx=(4, 8))

        tk.Button(coord_row, text="Go", command=self._goto,
                  **self._btn(fg="#89b4fa")).pack(side=tk.LEFT)

        feed_row = tk.Frame(goto_sec, bg="#181825")
        feed_row.pack(padx=8, pady=(0, 6), anchor="w")

        tk.Label(feed_row, text="Feed (G1 only):", bg="#181825", fg="#a6adc8",
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=(0, 4))
        self._goto_feed_var = tk.StringVar(value="1000")
        self._goto_feed_entry = tk.Entry(
            feed_row, textvariable=self._goto_feed_var, width=6,
            bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
            font=("Consolas", 9), relief=tk.FLAT, state=tk.DISABLED)
        self._goto_feed_entry.pack(side=tk.LEFT)
        tk.Label(feed_row, text=" mm/min", bg="#181825", fg="#585b70",
                 font=("Consolas", 8)).pack(side=tk.LEFT)

        mach = self._section(tab_content, "Machine")
        mach_row = tk.Frame(mach, bg="#181825")
        mach_row.pack(padx=8, pady=(6, 2), anchor="w")

        tk.Button(mach_row, text="Home  $H",
                  command=lambda: self._gantry_cmd(lambda w: w.home()),
                  **self._btn(fg="#cba6f7")).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(mach_row, text="Kill Alarm  $X",
                  command=lambda: self._gantry_cmd(lambda w: w.send("$X")),
                  **self._btn(fg="#fab387")).pack(side=tk.LEFT, padx=4)
        tk.Button(mach_row, text="Feed Hold  !",
                  command=lambda: self._gantry_cmd(lambda w: w.feed_hold()),
                  **self._btn(fg="#fab387")).pack(side=tk.LEFT, padx=4)
        tk.Button(mach_row, text="Cycle Start  ~",
                  command=lambda: self._gantry_cmd(lambda w: w.cycle_start()),
                  **self._btn(fg="#a6e3a1")).pack(side=tk.LEFT, padx=4)
        tk.Button(mach_row, text="Soft Reset  ^X",
                  command=lambda: self._gantry_cmd(lambda w: w.soft_reset()),
                  **self._btn(fg="#f38ba8")).pack(side=tk.LEFT, padx=4)

        origin_row = tk.Frame(mach, bg="#181825")
        origin_row.pack(padx=8, pady=(0, 6), anchor="w")

        tk.Label(origin_row, text="Set origin:", bg="#181825", fg="#a6adc8",
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(origin_row, text="Zero All  G92",
                  command=lambda: self._gantry_cmd(
                      lambda w: w.send("G92 X0 Y0 Z0")),
                  **self._btn(fg="#cdd6f4")).pack(side=tk.LEFT, padx=(0, 2))
        for axis in ("X", "Y", "Z"):
            tk.Button(origin_row, text=f"Zero {axis}",
                      command=lambda a=axis: self._gantry_cmd(
                          lambda w, ax=a: w.send(f"G92 {ax}0")),
                      **self._btn(fg="#a6adc8")).pack(side=tk.LEFT, padx=2)
        tk.Button(origin_row, text="Clear G92",
                  command=lambda: self._gantry_cmd(
                      lambda w: w.send("G92.1")),
                  **self._btn(fg="#585b70")).pack(side=tk.LEFT, padx=(8, 0))

        return tab

    # ── Log (below notebook) ──────────────────────────────────────────────────

    def _build_log(self):
        self._log_visible = True

        header = tk.Frame(self, bg="#11111b")
        header.pack(fill=tk.X, padx=8, pady=(2, 0))

        tk.Label(header, text="  Log", bg="#11111b", fg="#89b4fa",
                 font=("Consolas", 9, "bold")).pack(side=tk.LEFT)

        self._log_toggle_btn = tk.Button(
            header, text="▼ hide",
            command=self._toggle_log,
            bg="#11111b", fg="#585b70",
            activebackground="#11111b", activeforeground="#a6adc8",
            font=("Consolas", 8), relief=tk.FLAT, bd=0,
            cursor="hand2")
        self._log_toggle_btn.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8)

        self._log_frame = tk.Frame(self, bg="#11111b")
        self._log_frame.pack(fill=tk.X, padx=8, pady=(0, 6))

        self._log_widget = scrolledtext.ScrolledText(
            self._log_frame, height=5, bg="#11111b", fg="#a6adc8",
            font=("Consolas", 8), relief=tk.FLAT,
            state=tk.DISABLED, wrap=tk.WORD,
            insertbackground="#cdd6f4")
        self._log_widget.pack(fill=tk.X, padx=4, pady=4)

    def _toggle_log(self) -> None:
        if self._log_visible:
            self._log_frame.pack_forget()
            self._log_toggle_btn.config(text="▲ show")
            self._log_visible = False
        else:
            self._log_frame.pack(fill=tk.X, padx=8, pady=(0, 6))
            self._log_toggle_btn.config(text="▼ hide")
            self._log_visible = True

    # ── Shared widget helpers ─────────────────────────────────────────────────

    def _section(self, parent, title: str) -> tk.LabelFrame:
        f = tk.LabelFrame(parent, text=f"  {title}  ",
                          bg="#181825", fg="#89b4fa",
                          font=("Consolas", 9, "bold"),
                          relief=tk.FLAT, highlightthickness=1,
                          highlightbackground="#313244")
        f.pack(fill=tk.X, padx=8, pady=6, anchor="nw")
        return f

    def _btn(self, fg="#cdd6f4") -> dict:
        return dict(bg="#313244", fg=fg, activebackground="#45475a",
                    font=("Consolas", 9), relief=tk.FLAT, padx=8, pady=4)

    def _set_status(self, device: str, connected: bool):
        self._status_color[device].set("#a6e3a1" if connected else "#f38ba8")
        self._status_text[device].set("Connected" if connected else "Disconnected")

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log_widget.configure(state=tk.NORMAL)
        self._log_widget.insert(tk.END, f"[{ts}] {msg}\n")
        self._log_widget.see(tk.END)
        self._log_widget.configure(state=tk.DISABLED)

    # ── Recording logic ───────────────────────────────────────────────────────

    def _toggle_recording(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if self._arducam is None:
            self._log("[WARN ] Recording: ArduCam not connected.")
            return

        session_name = time.strftime("%Y%m%d_%H%M%S")
        self._session_dir = os.path.join(SESSIONS_ROOT, session_name)
        self._frames_dir  = os.path.join(self._session_dir, "frames")
        os.makedirs(self._frames_dir, exist_ok=True)

        self._session_csv_path = os.path.join(self._session_dir, "session.csv")

        try:
            self._session_csv_file   = open(self._session_csv_path, "w", newline="")
            self._session_csv_writer = csv.writer(self._session_csv_file)
            # ── NEW: added lat_deg and lon_deg columns ────────────────────
            self._session_csv_writer.writerow(
                ["timestamp_s", "filepath",
                 "x_mm", "y_mm", "z_mm",
                 "lat_deg", "lon_deg"])
        except OSError as e:
            self._log(f"[ERROR] Could not open CSV files: {e}")
            return

        self._recording          = True
        self._record_frame_count = 0
        self._last_capture_time  = 0.0
        self._record_count_var.set("0")
        self._record_status_var.set("● Recording")
        self._rec_status_lbl.configure(fg="#f38ba8")
        self._session_path_var.set(self._session_dir)
        self._rec_toggle_btn.configure(
            text="■  Stop Recording",
            bg="#6e2020", fg="#f38ba8", activebackground="#7e3030")

        if self._map_corner_tl is None or self._map_corner_br is None:
            self._log("[WARN ] Recording: map not calibrated — lat/lon columns will be NaN.")

        self._log(f"[OK   ] Recording started → {self._session_dir}")

    def _stop_recording(self) -> None:
        self._recording = False

        for fh in (self._session_csv_file,):
            if fh is not None:
                try:
                    fh.flush()
                    fh.close()
                except OSError:
                    pass
        self._session_csv_file   = None
        self._session_csv_writer = None

        self._record_status_var.set("Idle")
        self._rec_status_lbl.configure(fg="#585b70")
        self._rec_toggle_btn.configure(
            text="⏺  Start Recording",
            bg="#1e6640", fg="#a6e3a1", activebackground="#2a7a50")

        self._log(
            f"[OK   ] Recording stopped — {self._record_frame_count} frames saved "
            f"to {self._session_dir}"
        )

    def _maybe_capture_frame(self, frame_bgr: np.ndarray) -> None:
        """
        Called every GUI poll with the latest ArduCam frame (BGR).
        Saves a JPEG and writes one row to session.csv, throttled to target FPS.

        CSV columns: timestamp_s, filepath, x_mm, y_mm, z_mm, lat_deg, lon_deg
        lat/lon are derived from the current gantry position using the calibrated
        map corners (TL = 90°lat/0°lon, BR = -90°lat/360°lon).
        """
        if not self._recording:
            return

        now = time.monotonic()
        target_interval = 1.0 / max(self._record_fps_var.get(), 1)
        if (now - self._last_capture_time) < target_interval:
            return

        self._last_capture_time = now
        wall_ts = time.time()

        fname    = f"{wall_ts:.6f}.jpg"
        rel_path = os.path.join("frames", fname)
        abs_path = os.path.join(self._frames_dir, fname)
        try:
            cv2.imwrite(abs_path, frame_bgr)
        except Exception as e:
            self._log(f"[ERROR] Frame save failed: {e}")
            return

        # ── Gantry position ───────────────────────────────────────────────
        gx = gy = gz = float("nan")
        if self._gantry_worker is not None:
            try:
                pos, _ = self._gantry_worker.get_position()
                gx, gy, gz = pos["x"], pos["y"], pos["z"]
            except Exception:
                pass

        # ── Lat/lon from gantry position ──────────────────────────────────
        lat_deg, lon_deg = self._gantry_mm_to_lat_lon(gx, gy)

        try:
            self._session_csv_writer.writerow([
                f"{wall_ts:.6f}", rel_path,
                f"{gx:.4f}", f"{gy:.4f}", f"{gz:.4f}",
                f"{lat_deg:.6f}", f"{lon_deg:.6f}",
            ])
        except Exception as e:
            self._log(f"[ERROR] session.csv write failed: {e}")

        self._record_frame_count += 1
        self._record_count_var.set(str(self._record_frame_count))

    # ── FPS control ───────────────────────────────────────────────────────────

    def _on_fps_slider(self, value) -> None:
        try:
            clamped = max(1, min(30, int(float(value))))
            self._record_fps_var.set(clamped)
        except (ValueError, TypeError):
            pass

    # ── ORB map video recording ───────────────────────────────────────────────

    def _toggle_orb_video(self) -> None:
        if self._orb_video_recording:
            self._stop_orb_video()
        else:
            self._start_orb_video()

    def _start_orb_video(self) -> None:
        """
        Open a cv2.VideoWriter that will capture the ORB map canvas each time
        _orb_redraw_map() is called (throttled to the selected video FPS).

        The output file is placed in the current session directory if a
        recording session is active, otherwise in SESSIONS_ROOT directly.
        """
        if not _CV2_AVAILABLE:
            self._log("[ERROR] ORB video: OpenCV not available.")
            return

        # Choose output directory
        out_dir = self._session_dir if self._session_dir else SESSIONS_ROOT
        os.makedirs(out_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        self._orb_video_path = os.path.join(out_dir, f"orb_map_{ts}.mp4")

        fps = max(1, min(30, self._orb_video_fps_var.get()))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self._orb_video_writer = cv2.VideoWriter(
            self._orb_video_path, fourcc, fps,
            (ORB_MAP_W, ORB_MAP_H))

        if not self._orb_video_writer.isOpened():
            self._log(f"[ERROR] ORB video: could not open VideoWriter for '{self._orb_video_path}'")
            self._orb_video_writer = None
            return

        self._orb_video_recording   = True
        self._orb_video_frame_count = 0
        self._last_orb_video_time   = 0.0   # force first frame immediately
        self._orb_video_count_var.set("0")
        self._orb_video_status_var.set("● Recording")
        self._orb_vid_status_lbl.configure(fg="#f38ba8")
        self._orb_vid_btn.configure(
            text="■  Stop Map Video",
            bg="#6e2020", fg="#f38ba8", activebackground="#7e3030")

        self._log(f"[OK   ] ORB map video recording started → {self._orb_video_path}")

    def _stop_orb_video(self) -> None:
        self._orb_video_recording = False
        if self._orb_video_writer is not None:
            self._orb_video_writer.release()
            self._orb_video_writer = None

        self._orb_video_status_var.set("Idle")
        self._orb_vid_status_lbl.configure(fg="#585b70")
        self._orb_vid_btn.configure(
            text="⏺  Record Map Video",
            bg="#1e6640", fg="#a6e3a1", activebackground="#2a7a50")

        self._log(
            f"[OK   ] ORB map video stopped — "
            f"{self._orb_video_frame_count} frames → {self._orb_video_path}"
        )

    def _render_orb_map_to_array(self) -> np.ndarray:
        """
        Re-render the ORB map directly into a (ORB_MAP_H × ORB_MAP_W × 3) uint8
        numpy array (RGB).  Mirrors exactly what _orb_redraw_map() draws onto the
        Tk canvas, but works entirely in PIL/numpy — no screen capture needed.
        """
        # ── Background / basemap ─────────────────────────────────────────
        if self._orb_map_photo is not None and hasattr(self, "_orb_ref_img_bgr"):
            # Use the cached resized basemap we stored when loading
            base = self._orb_basemap_pil.copy()
        else:
            # Plain dark background
            base = Image.new("RGB", (ORB_MAP_W, ORB_MAP_H), color=(13, 13, 26))

        from PIL import ImageDraw
        draw = ImageDraw.Draw(base)

        n = len(self._orb_history_map)

        # ── Estimated-position trail (blue) ──────────────────────────────
        for i, (px, py) in enumerate(self._orb_history_map):
            alpha_frac = (i + 1) / max(n, 1)
            r = int(30  + alpha_frac * 80)
            g = int(40  + alpha_frac * 140)
            b = int(120 + alpha_frac * 135)
            r_dot = max(2, int(3 * alpha_frac))
            draw.ellipse([px - r_dot, py - r_dot, px + r_dot, py + r_dot],
                         fill=(r, g, b))

        if len(self._orb_history_map) >= 2:
            flat = [coord for pt in self._orb_history_map for coord in pt]
            pts  = list(zip(flat[0::2], flat[1::2]))
            draw.line(pts, fill=(68, 136, 204), width=1)

        # ── Ground-truth trail (green) ────────────────────────────────────
        n_gt = len(self._orb_history_gantry)
        gt_canvas_pts = []
        for i, (gx_mm, gy_mm) in enumerate(self._orb_history_gantry):
            px, py = self._gantry_mm_to_canvas(gx_mm, gy_mm)
            if px is None:
                continue
            alpha_frac = (i + 1) / max(n_gt, 1)
            r_dot  = max(2, int(3 * alpha_frac))
            shade  = int(80 + alpha_frac * 175)
            draw.ellipse([px - r_dot, py - r_dot, px + r_dot, py + r_dot],
                         fill=(0, shade, 0))
            gt_canvas_pts.append((px, py))

        if len(gt_canvas_pts) >= 2:
            draw.line(gt_canvas_pts, fill=(68, 204, 68), width=1)

        # ── Current estimated position marker (cyan cross + circle) ──────
        if self._orb_history_map:
            cx, cy = self._orb_history_map[-1]
            R = 6
            draw.ellipse([cx - R, cy - R, cx + R, cy + R],
                         outline=(255, 255, 255), fill=(0, 212, 255), width=1)
            # cross arms (extend R px beyond the circle edge)
            draw.line([(cx - R*2, cy), (cx - R, cy)], fill=(0, 212, 255), width=1)
            draw.line([(cx + R,   cy), (cx + R*2, cy)], fill=(0, 212, 255), width=1)
            draw.line([(cx, cy - R*2), (cx, cy - R)], fill=(0, 212, 255), width=1)
            draw.line([(cx, cy + R),   (cx, cy + R*2)], fill=(0, 212, 255), width=1)
            draw.text((cx + R + 4, cy - R - 4), "EST", fill=(0, 212, 255))

        # ── Current ground-truth marker (green circle) ────────────────────
        if self._orb_history_gantry:
            gx_mm, gy_mm = self._orb_history_gantry[-1]
            px, py = self._gantry_mm_to_canvas(gx_mm, gy_mm)
            if px is not None:
                R = 6
                draw.ellipse([px - R, py - R, px + R, py + R],
                             outline=(255, 255, 255), fill=(0, 255, 136), width=1)
                draw.text((px + R + 4, py - R - 4), "GT", fill=(0, 255, 136))

        return np.array(base)   # RGB uint8

    def _maybe_capture_orb_video_frame(self) -> None:
        """
        Render the ORB map to a numpy array and write it to the VideoWriter,
        throttled to the chosen video FPS.
        Called from _orb_redraw_map() every time the map is redrawn.
        No screen capture — pixels come straight from the same data the canvas uses.
        """
        if not self._orb_video_recording or self._orb_video_writer is None:
            return

        now = time.monotonic()
        fps = max(1, min(30, self._orb_video_fps_var.get()))
        if (now - self._last_orb_video_time) < (1.0 / fps):
            return

        self._last_orb_video_time = now

        try:
            frame_rgb = self._render_orb_map_to_array()          # RGB
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self._log(f"[WARN ] ORB video render failed: {e}")
            return

        self._orb_video_writer.write(frame_bgr)
        self._orb_video_frame_count += 1
        self._orb_video_count_var.set(str(self._orb_video_frame_count))

    # ── RealSense D435 connection ─────────────────────────────────────────────

    def _rs_enumerate(self):
        if not _RS_AVAILABLE:
            self._log("[ERROR] pyrealsense2 not installed — pip install pyrealsense2")
            return
        try:
            ctx     = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                self._log("[WARN ] No RealSense devices found.")
                return
            for dev in devices:
                name   = dev.get_info(rs.camera_info.name)
                serial = dev.get_info(rs.camera_info.serial_number)
                fw     = dev.get_info(rs.camera_info.firmware_version)
                self._log(f"[INFO ] {name}  serial={serial}  fw={fw}")
        except Exception as e:
            self._log(f"[ERROR] Enumerate failed: {e}")

    def _connect_rs(self):
        if not _RS_AVAILABLE:
            self._log("[ERROR] pyrealsense2 not installed — pip install pyrealsense2")
            return
        if self._rs_pipeline is not None:
            self._log("[WARN ] RealSense already connected.")
            return

        cfg_rs = self._cfg.realsense
        self._log("[INFO ] Starting RealSense pipeline...")
        try:
            pipeline = rs.pipeline()
            rs_cfg   = rs.config()
            rs_cfg.enable_stream(rs.stream.depth, cfg_rs.resolution_x, cfg_rs.resolution_y,
                                 rs.format.z16,  cfg_rs.fps)
            rs_cfg.enable_stream(rs.stream.color, cfg_rs.resolution_x, cfg_rs.resolution_y,
                                 rs.format.bgr8, cfg_rs.fps)
            profile  = pipeline.start(rs_cfg)
            dev      = profile.get_device()
            serial   = dev.get_info(rs.camera_info.serial_number)
            name     = dev.get_info(rs.camera_info.name)
            fw       = dev.get_info(rs.camera_info.firmware_version)
            self._log(f"[OK   ] {name}  serial={serial}  fw={fw}")
            self._rs_pipeline = pipeline
            self._rs_serial   = serial
            self._rs_serial_var.set(serial)
            self._set_status("LiDAR", True)
        except Exception as e:
            self._log(f"[ERROR] {e}")
            return

        self._rs_frame_q = queue.Queue(maxsize=2)
        self._rs_worker  = RealSenseWorker(self._rs_pipeline, self._rs_frame_q)
        self._rs_worker.start()

    def _disconnect_rs(self):
        if self._rs_worker:
            self._rs_worker.stop()
            self._rs_worker = None
        if self._rs_pipeline:
            try:
                self._rs_pipeline.stop()
            except Exception:
                pass
            self._rs_pipeline = None
            self._log("[INFO ] RealSense disconnected.")
        self._rs_serial_var.set("—")
        self._set_status("LiDAR", False)

    def _apply_rs_settings(self):
        if self._rs_pipeline is None:
            self._log("[WARN ] Not connected — connect first.")
            return
        try:
            power    = int(self._rs_settings["Emitter Power"].get())
            exposure = int(self._rs_settings["Exposure (µs)"].get())
        except ValueError:
            self._log("[ERROR] Settings must be integers.")
            return
        try:
            sensor = self._rs_pipeline.get_active_profile().get_device().first_depth_sensor()
            sensor.set_option(rs.option.emitter_enabled, 1)
            sensor.set_option(rs.option.laser_power, power)
            if exposure == 0:
                sensor.set_option(rs.option.enable_auto_exposure, 1)
                self._log(f"[OK   ] Emitter power: {power}, exposure: auto")
            else:
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                sensor.set_option(rs.option.exposure, exposure)
                self._log(f"[OK   ] Emitter power: {power}, exposure: {exposure} µs")
        except Exception as e:
            self._log(f"[ERROR] {e}")

    # ── ArduCam connection ────────────────────────────────────────────────────

    def _connect_arducam(self):
        if not _CV2_AVAILABLE:
            self._log("[ERROR] OpenCV not available.")
            return
        idx = int(self._arducam_idx_var.get())
        self._log(f"[INFO ] Opening camera index {idx}...")
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            self._log(f"[ERROR] Could not open camera index {idx}.")
            return
        self._arducam = cap
        self._log(f"[OK   ] ArduCam opened (index {idx}).")
        self._set_status("ArduCam", True)

        af_val    = cap.get(cv2.CAP_PROP_AUTOFOCUS)
        focus_val = cap.get(cv2.CAP_PROP_FOCUS)
        af_on = bool(af_val)
        self._arducam_autofocus_var.set(af_on)
        if af_on:
            self._af_btn.config(text="⟳  Autofocus: ON",
                                bg="#1e4466", fg="#89b4fa",
                                activebackground="#2a5070")
        else:
            self._af_btn.config(text="⟳  Autofocus: OFF",
                                bg="#313244", fg="#585b70",
                                activebackground="#45475a")
        if focus_val >= 0:
            self._arducam_focus_var.set(str(int(focus_val)))
            self._focus_readout_var.set(str(int(focus_val)))
        self._log(f"[INFO ] Camera initial state — autofocus={'on' if af_on else 'off'}  "
                  f"focus={int(focus_val)}")

    def _disconnect_arducam(self):
        if self._recording:
            self._stop_recording()
        if self._orb_running:
            self._stop_orb()
        if self._orb_video_recording:
            self._stop_orb_video()
        if self._arducam:
            self._arducam.release()
            self._arducam = None
            self._log("[INFO ] ArduCam disconnected.")
        self._set_status("ArduCam", False)

    # ── ArduCam focus helpers ─────────────────────────────────────────────────

    def _toggle_autofocus(self) -> None:
        new_state = not self._arducam_autofocus_var.get()
        self._arducam_autofocus_var.set(new_state)
        if new_state:
            self._af_btn.config(text="⟳  Autofocus: ON",
                                bg="#1e4466", fg="#89b4fa",
                                activebackground="#2a5070")
        else:
            self._af_btn.config(text="⟳  Autofocus: OFF",
                                bg="#313244", fg="#585b70",
                                activebackground="#45475a")
        if self._arducam is not None:
            val = 1 if new_state else 0
            supported = self._arducam.set(cv2.CAP_PROP_AUTOFOCUS, val)
            state_str = "ON" if new_state else "OFF"
            if supported:
                self._log(f"[OK   ] Autofocus set {state_str}.")
            else:
                self._log(f"[WARN ] Camera did not accept autofocus={state_str}.")
        else:
            self._log("[WARN ] Camera not connected — setting will apply on next connect.")

    def _trigger_autofocus(self) -> None:
        if self._arducam is None:
            self._log("[WARN ] Camera not connected.")
            return
        self._arducam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self._arducam_autofocus_var.set(True)
        self._af_btn.config(text="⟳  Autofocus: ON",
                            bg="#1e4466", fg="#89b4fa",
                            activebackground="#2a5070")
        self._log("[INFO ] Autofocus trigger sent.")

    def _on_focus_slider(self, value) -> None:
        if self._arducam is None:
            return
        try:
            self._arducam.set(cv2.CAP_PROP_FOCUS, int(float(value)))
        except (ValueError, TypeError):
            pass

    def _apply_manual_focus(self) -> None:
        if self._arducam is None:
            self._log("[WARN ] Camera not connected.")
            return
        try:
            focus_val = int(self._arducam_focus_var.get())
        except ValueError:
            self._log("[ERROR] Focus value must be an integer (0–1023).")
            return
        if not 0 <= focus_val <= 1023:
            self._log("[ERROR] Focus value must be between 0 and 1023.")
            return
        supported = self._arducam.set(cv2.CAP_PROP_FOCUS, focus_val)
        actual    = self._arducam.get(cv2.CAP_PROP_FOCUS)
        self._focus_readout_var.set(str(int(actual)))
        if supported:
            self._log(f"[OK   ] Focus set to {focus_val} (camera reports {int(actual)}).")
        else:
            self._log(f"[WARN ] Camera did not accept focus={focus_val}.")

    # ── Gantry connection ─────────────────────────────────────────────────────

    def _connect_gantry(self):
        if self._gantry_worker is not None:
            self._log("[WARN ] Gantry already connected.")
            return
        if not _GRBL_AVAILABLE:
            self._log("[ERROR] gantry.py not importable — check installation.")
            return
        if not _GantryWorker.available():
            self._log("[ERROR] pyserial not installed — pip install pyserial")
            return

        port = self._gantry_port_var.get().strip()
        if not port:
            self._log("[ERROR] No port specified.")
            return
        try:
            baud = int(self._gantry_baud_var.get())
        except ValueError:
            self._log("[ERROR] Invalid baud rate.")
            return

        self._log(f"[INFO ] Connecting to gantry on {port} @ {baud} …")
        try:
            worker = _GantryWorker(port, baud, self._gantry_resp_q)
            worker.open()
        except Exception as exc:
            self._log(f"[ERROR] Gantry: {exc}")
            return

        worker.start()
        self._gantry_worker = worker
        self._set_status("Gantry", True)
        self._log(f"[OK   ] Gantry connected — {port} @ {baud}")

        for key in ("Left", "Right", "Up", "Down", "Prior", "Next"):
            self.bind(f"<{key}>", self._on_key_jog)

    def _disconnect_gantry(self):
        if self._gantry_worker is not None:
            self._gantry_worker.close()
            self._gantry_worker = None
            self._log("[INFO ] Gantry disconnected.")
        for key in ("Left", "Right", "Up", "Down", "Prior", "Next"):
            self.unbind(f"<{key}>")
        self._gantry_state_var.set("—")
        self._gantry_x_var.set("—")
        self._gantry_y_var.set("—")
        self._gantry_z_var.set("—")
        self._set_status("Gantry", False)

    # ── Gantry helpers ────────────────────────────────────────────────────────

    def _gantry_cmd(self, fn) -> None:
        if self._gantry_worker is None:
            self._log("[WARN ] Gantry not connected.")
            return
        fn(self._gantry_worker)

    def _set_jog_step(self, step: float) -> None:
        self._jog_step_var.set(step)
        self._update_jog_step_btns()

    def _update_jog_step_btns(self) -> None:
        sel = self._jog_step_var.get()
        for step, btn in self._jog_step_btns.items():
            if abs(step - sel) < 1e-9:
                btn.config(bg="#89b4fa", fg="#1e1e2e", activebackground="#74c7ec")
            else:
                btn.config(bg="#313244", fg="#cdd6f4", activebackground="#45475a")

    def _toggle_goto_mode(self) -> None:
        rapid = not self._goto_rapid_var.get()
        self._goto_rapid_var.set(rapid)
        if rapid:
            self._goto_mode_btn.config(
                text="G0  Rapid", bg="#1e6640", fg="#a6e3a1",
                activebackground="#2a7a50")
            self._goto_feed_entry.config(state=tk.DISABLED)
        else:
            self._goto_mode_btn.config(
                text="G1  Feed ", bg="#6e4520", fg="#fab387",
                activebackground="#7e5030")
            self._goto_feed_entry.config(state=tk.NORMAL)

    def _jog(self, axis: str, sign: int) -> None:
        if self._gantry_worker is None:
            return
        try:
            step = self._jog_step_var.get()
            feed = float(self._jog_feed_var.get())
        except (ValueError, tk.TclError):
            self._log("[ERROR] Invalid jog parameters.")
            return
        self._gantry_worker.jog(axis, sign * step, feed)

    def _goto(self) -> None:
        if self._gantry_worker is None:
            self._log("[WARN ] Gantry not connected.")
            return
        try:
            x = float(self._goto_x_var.get()) if self._goto_x_var.get().strip() else None
            y = float(self._goto_y_var.get()) if self._goto_y_var.get().strip() else None
            z = float(self._goto_z_var.get()) if self._goto_z_var.get().strip() else None
            feed  = float(self._goto_feed_var.get())
            rapid = self._goto_rapid_var.get()
        except ValueError:
            self._log("[ERROR] GoTo: invalid coordinate.")
            return
        if x is None and y is None and z is None:
            self._log("[WARN ] GoTo: no coordinates specified.")
            return
        self._gantry_worker.goto(x=x, y=y, z=z, feed_mm_min=feed, rapid=rapid)
        mode = "G0" if rapid else "G1"
        self._log(f"[INFO ] GoTo {mode}: X={x} Y={y} Z={z}")

    def _on_key_jog(self, event) -> str:
        if self._gantry_worker is None:
            return ""
        focused = self.focus_get()
        if isinstance(focused, (tk.Entry, tk.Text)):
            return ""
        key_map = {
            "Left":  ("X", -1),
            "Right": ("X", +1),
            "Up":    ("Y", +1),
            "Down":  ("Y", -1),
            "Prior": ("Z", +1),
            "Next":  ("Z", -1),
        }
        if event.keysym in key_map:
            axis, sign = key_map[event.keysym]
            self._jog(axis, sign)
            return "break"
        return ""

    def _poll_gantry(self) -> None:
        try:
            while True:
                line = self._gantry_resp_q.get_nowait()
                if line.startswith("SERIAL_ERROR:"):
                    self._log(f"[ERROR] {line}")
                    self._disconnect_gantry()
                    return
                elif line == "ok":
                    pass
                elif line.lower().startswith("alarm"):
                    self._log(f"[ALARM] Gantry: {line}")
                elif line.lower().startswith("error"):
                    self._log(f"[ERROR] Gantry: {line}")
                else:
                    self._log(f"[GRBL ] {line}")
        except queue.Empty:
            pass

        if self._gantry_worker is None:
            return

        pos, state = self._gantry_worker.get_position()
        self._gantry_state_var.set(state)
        self._gantry_x_var.set(f"{pos['x']:9.3f}")
        self._gantry_y_var.set(f"{pos['y']:9.3f}")
        self._gantry_z_var.set(f"{pos['z']:9.3f}")

        colors = {
            "Idle":    "#a6e3a1",
            "Run":     "#89b4fa",
            "Hold":    "#fab387",
            "Alarm":   "#f38ba8",
            "Home":    "#cba6f7",
            "Unknown": "#585b70",
        }
        self._gantry_state_lbl.config(fg=colors.get(state, "#cdd6f4"))

    # ── Experiment control ────────────────────────────────────────────────────

    def _start_experiment(self):
        if not _GT_AVAILABLE:
            self._log("[ERROR] ground_track.py not available — check repo root.")
            return

        if self._ground_track is None or len(self._ground_track) == 0:
            self._log("[ERROR] No ground track loaded — click Generate / Load first.")
            return

        dry_run = self._gt_dry_run_var.get()

        if not dry_run:
            if self._gantry_worker is None:
                self._log("[ERROR] Gantry not connected — connect before running live.")
                return
            if self._map_corner_br is None or self._map_corner_tl is None:
                self._log("[ERROR] Map not calibrated — record both corners first.")
                return

        if self._map_corner_br is not None and self._map_corner_tl is not None:
            corner_br = self._map_corner_br
            corner_tl = self._map_corner_tl
        else:
            g  = self._cfg.gantry
            tx = g.travel_x_mm or 400.0
            ty = g.travel_y_mm or 800.0
            corner_br = (tx, ty)
            corner_tl = (0.0, 0.0)
            self._log("[WARN ] No calibration recorded — using config travel limits.")

        region = self._get_dem_region_bounds()
        if region is None:
            if self._ground_track and len(self._ground_track) > 0:
                lats = [wp.lat_deg for wp in self._ground_track.waypoints]
                lons = [wp.lon_deg for wp in self._ground_track.waypoints]
                region = DemRegionBounds(
                    lat_min_deg=min(lats),
                    lat_max_deg=max(lats),
                    lon_min_deg=min(lons),
                    lon_max_deg=max(lons),
                )
                self._log(
                    f"[INFO ] dem.region not configured — using track extent: "
                    f"lat [{region.lat_min_deg:.3f}, {region.lat_max_deg:.3f}]  "
                    f"lon [{region.lon_min_deg:.3f}, {region.lon_max_deg:.3f}]"
                )
            else:
                region = DemRegionBounds(0.0, 1.0, 0.0, 1.0)

        cal = MapCalibration(corner_br=corner_br, corner_tl=corner_tl, region=region)

        try:
            speed = float(self._gt_speed_var.get())
        except ValueError:
            self._log("[ERROR] Invalid speed value.")
            return

        tscale = self._cfg.simulation.time_scale_factor or 1.0

        runner = TrackRunner(
            calibration=cal,
            track=self._ground_track,
            gantry_worker=self._gantry_worker,
            speed_mm_min=speed,
            time_scale_factor=tscale,
            dry_run=dry_run,
            log_cb=self._log,
        )
        self._track_runner = runner
        runner.start()

        self._start_btn.configure(state=tk.DISABLED)
        self._stop_btn.configure(state=tk.NORMAL)
        self._log(f"[INFO ] Track runner started ({'dry run' if dry_run else 'LIVE'}).")
        self._poll_track_runner()

    def _stop_experiment(self):
        if self._track_runner is not None:
            self._track_runner.stop()
            self._log("[INFO ] Stop requested — waiting for runner to exit.")
        self._start_btn.configure(state=tk.NORMAL)
        self._stop_btn.configure(state=tk.DISABLED)

    def _poll_track_runner(self) -> None:
        if self._track_runner is None:
            return
        if self._track_runner.is_alive():
            self.after(200, self._poll_track_runner)
        else:
            self._track_runner = None
            self._start_btn.configure(state=tk.NORMAL)
            self._stop_btn.configure(state=tk.DISABLED)
            self._log("[INFO ] Track runner finished.")

    # ── Map calibration helpers ───────────────────────────────────────────────

    def _record_corner_br(self) -> None:
        if self._gantry_worker is None:
            self._log("[WARN ] Gantry not connected — cannot record corner.")
            return
        pos, _ = self._gantry_worker.get_position()
        self._map_corner_br = (pos["x"], pos["y"])
        self._map_br_var.set(f"X={pos['x']:.2f}  Y={pos['y']:.2f}")
        self._log(f"[INFO ] Map corner BR recorded: X={pos['x']:.2f} Y={pos['y']:.2f} mm")
        self._update_map_size_display()

    def _record_corner_tl(self) -> None:
        if self._gantry_worker is None:
            self._log("[WARN ] Gantry not connected — cannot record corner.")
            return
        pos, _ = self._gantry_worker.get_position()
        self._map_corner_tl = (pos["x"], pos["y"])
        self._map_tl_var.set(f"X={pos['x']:.2f}  Y={pos['y']:.2f}")
        self._log(f"[INFO ] Map corner TL recorded: X={pos['x']:.2f} Y={pos['y']:.2f} mm")
        self._update_map_size_display()

    def _update_map_size_display(self) -> None:
        if self._map_corner_br is None or self._map_corner_tl is None:
            self._map_size_var.set("—")
            return
        dx = abs(self._map_corner_br[0] - self._map_corner_tl[0])
        dy = abs(self._map_corner_br[1] - self._map_corner_tl[1])
        self._map_size_var.set(f"{dx:.1f} × {dy:.1f} mm")

    # ── Ground-track helpers ──────────────────────────────────────────────────

    def _on_gt_source_change(self) -> None:
        if self._gt_source_var.get() == "synthetic":
            self._gt_csv_frame.pack_forget()
            self._gt_synth_frame.pack(anchor="w")
        else:
            self._gt_synth_frame.pack_forget()
            self._gt_csv_frame.pack(anchor="w")

    def _browse_gt_csv(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select ground track CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self._gt_csv_path_var.set(path)

    _GT_COL_ALIASES: dict = {
        "lat_deg": ["lat", "latitude", "lat_degrees", "Lat", "Latitude",
                    "LAT", "LATITUDE", "lat_deg"],
        "lon_deg": ["lon", "longitude", "lon_degrees", "Lon", "Longitude",
                    "LON", "LONGITUDE", "lng", "Lng", "LNG", "lon_deg"],
        "time_s":  ["time", "t", "time_s", "timestamp", "Time", "T",
                    "elapsed_s", "elapsed"],
    }

    def _normalise_gt_csv(self, src_path: str) -> str:
        import tempfile

        with open(src_path, newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return src_path
            original_cols = list(reader.fieldnames)
            rows = list(reader)

        remap: dict[str, str] = {}
        for canonical, aliases in self._GT_COL_ALIASES.items():
            for col in original_cols:
                if col == canonical:
                    break
                if col in aliases:
                    remap[col] = canonical
                    break

        if not remap:
            self._log(f"[INFO ] CSV columns look correct: {original_cols}")
            return src_path

        self._log(f"[INFO ] CSV column remap: {remap}")

        new_cols = [remap.get(c, c) for c in original_cols]
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="")
        writer = csv.DictWriter(tmp, fieldnames=new_cols)
        writer.writeheader()
        for row in rows:
            new_row = {remap.get(k, k): v for k, v in row.items()}
            writer.writerow(new_row)
        tmp.close()
        return tmp.name

    def _load_ground_track(self) -> None:
        if not _GT_AVAILABLE:
            self._log("[ERROR] ground_track.py not found.")
            return
        try:
            if self._gt_source_var.get() == "synthetic":
                n       = int(self._gt_n_var.get())
                dur     = float(self._gt_duration_var.get())
                cycles  = float(self._gt_cycles_var.get())
                region  = self._get_dem_region_bounds()
                if region is not None:
                    track = generate_synthetic_pass(region, duration_s=dur,
                                                    n_points=n, n_cycles=cycles)
                else:
                    track = generate_synthetic_pass_normalized(n_points=n, duration_s=dur,
                                                               n_cycles=cycles)
                    self._log("[WARN ] dem.region not configured — using normalised [0,1] coords.")
            else:
                path = self._gt_csv_path_var.get().strip()
                if not path:
                    self._log("[ERROR] No CSV path specified.")
                    return
                track = _load_gt_csv(path)
                self._log(f"[INFO ] CSV loaded from: {path}")

            self._ground_track = track
            path_len = track.path_length_deg()
            self._gt_status_var.set(f"{len(track)} waypoints — path {path_len:.4f}°")
            self._log(f"[OK   ] Ground track loaded: {len(track)} waypoints, "
                      f"path length {path_len:.4f}°")
        except Exception as exc:
            self._log(f"[ERROR] Ground track: {exc}")

    def _get_dem_region_bounds(self):
        r = self._cfg.dem.region
        if None in (r.lat_min_deg, r.lat_max_deg, r.lon_min_deg, r.lon_max_deg):
            return None
        return DemRegionBounds(
            lat_min_deg=r.lat_min_deg,
            lat_max_deg=r.lat_max_deg,
            lon_min_deg=r.lon_min_deg,
            lon_max_deg=r.lon_max_deg,
        )

    # ── Camera frame polling ──────────────────────────────────────────────────

    def _poll_cameras(self):
        # RealSense D435
        try:
            item = self._rs_frame_q.get_nowait()
            if isinstance(item[0], str):
                self._log(f"[ERROR] RealSense worker: {item[1]}")
                self._disconnect_rs()
            else:
                depth_colorized, color_rgb, depth_m = item
                self._last_depth_m   = depth_m
                self._last_grayscale = color_rgb
                self._last_depth_rgb = depth_colorized
                self._draw(self._depth_canvas, depth_colorized, RS_W, RS_H)
                self._draw(self._gray_canvas,  color_rgb,       RS_W, RS_H)
        except queue.Empty:
            pass
        except Exception as e:
            self._log(f"[ERROR] Poll error: {e}")

        # ArduCam
        if self._arducam is not None:
            ret, frame_bgr = self._arducam.read()
            if ret:
                self._draw(self._arducam_canvas,
                           cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                           ARDUCAM_W, ARDUCAM_H)
                self._draw_crosshair(self._arducam_canvas, ARDUCAM_W, ARDUCAM_H)
                self._maybe_capture_frame(frame_bgr)
            focus_now = self._arducam.get(cv2.CAP_PROP_FOCUS)
            if focus_now >= 0:
                self._focus_readout_var.set(str(int(focus_now)))

        self._poll_gantry()
        self._poll_orb()
        self._after_id = self.after(POLL_MS, self._poll_cameras)

    def _draw(self, canvas: tk.Canvas, arr: np.ndarray, w: int, h: int):
        img   = Image.fromarray(arr).resize((w, h), Image.NEAREST)
        photo = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas._photo = photo

    def _draw_crosshair(self, canvas: tk.Canvas, w: int, h: int,
                        size: int = 20, gap: int = 6) -> None:
        cx, cy = w // 2, h // 2
        canvas.delete("crosshair")
        kw = dict(fill="#00ff88", width=1, tags="crosshair")
        canvas.create_line(cx - size, cy, cx - gap, cy, **kw)
        canvas.create_line(cx + gap,  cy, cx + size, cy, **kw)
        canvas.create_line(cx, cy - size, cx, cy - gap, **kw)
        canvas.create_line(cx, cy + gap,  cx, cy + size, **kw)
        canvas.create_oval(cx - 1, cy - 1, cx + 1, cy + 1,
                           fill="#00ff88", outline="", tags="crosshair")

    # ── Snap & Visualize ──────────────────────────────────────────────────────

    def _snap_and_plot(self):
        self._log("[WARN ] _snap_and_plot: keep your existing implementation here.")

    def _close_snap_window(self):
        if self._snap_window is not None:
            try:
                self._snap_window.destroy()
            except Exception:
                pass
            self._snap_window = None

    # ── ORB Localization ──────────────────────────────────────────────────────

    def _browse_orb_ref(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Mars reference map image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       ("All files", "*.*")],
        )
        if path:
            self._orb_ref_img_path.set(path)
            self._orb_load_basemap()

    def _orb_load_basemap(self) -> bool:
        if not _CV2_AVAILABLE:
            self._log("[ERROR] Basemap: OpenCV not available.")
            return False
        path = self._orb_ref_img_path.get().strip()
        self._log(f"[INFO ] Basemap: trying to load '{path}'")
        self._log(f"[INFO ] Basemap: cwd is '{os.getcwd()}'")
        if not path:
            self._log("[ERROR] Basemap: path is empty.")
            return False
        if not os.path.isfile(path):
            self._log(f"[ERROR] Basemap: file not found — '{path}'")
            self._log(f"[INFO ] Basemap: files in cwd: {os.listdir('.')}")
            return False
        try:
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                self._log(f"[ERROR] Basemap: cv2.imread returned None for '{path}'")
                return False
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb).resize(
                (ORB_MAP_W, ORB_MAP_H), Image.LANCZOS)
            self._orb_map_photo   = ImageTk.PhotoImage(pil_img)
            self._orb_basemap_pil = pil_img          # kept as PIL for video rendering
            self._orb_ref_img_bgr = True             # sentinel: basemap is loaded
            h, w = img_bgr.shape[:2]
            self._orb_ref_w = w
            self._orb_ref_h = h
            self._log(f"[OK   ] Basemap loaded: {w}\u00d7{h} px")
            return True
        except Exception as e:
            self._log(f"[ERROR] Basemap load exception: {e}")
            return False

    def _orb_draw_placeholder(self) -> None:
        cnv = self._orb_map_canvas
        cnv.delete("all")
        cnv.create_rectangle(0, 0, ORB_MAP_W, ORB_MAP_H, fill="#0d0d1a", outline="")
        grid_col = "#1e2040"
        for x in range(0, ORB_MAP_W, ORB_MAP_W // 12):
            cnv.create_line(x, 0, x, ORB_MAP_H, fill=grid_col, width=1)
        for y in range(0, ORB_MAP_H, ORB_MAP_H // 6):
            cnv.create_line(0, y, ORB_MAP_W, y, fill=grid_col, width=1)
        cnv.create_text(ORB_MAP_W // 2, ORB_MAP_H // 2,
                        text="Load reference map to enable ORB localization",
                        fill="#313244", font=("Consolas", 9), anchor="center")

    def _orb_redraw_map(self) -> None:
        """Redraw the ORB map canvas, then optionally capture a video frame."""
        cnv = self._orb_map_canvas
        cnv.delete("all")

        if self._orb_map_photo is not None:
            cnv.create_image(0, 0, anchor=tk.NW, image=self._orb_map_photo)
        else:
            self._orb_draw_placeholder()
            return

        n = len(self._orb_history_map)
        for i, (px, py) in enumerate(self._orb_history_map):
            alpha_frac = (i + 1) / max(n, 1)
            r = int(30  + alpha_frac * 80)
            g = int(40  + alpha_frac * 140)
            b = int(120 + alpha_frac * 135)
            color = f"#{r:02x}{g:02x}{b:02x}"
            r_dot = max(2, int(3 * alpha_frac))
            cnv.create_oval(px - r_dot, py - r_dot, px + r_dot, py + r_dot,
                            fill=color, outline="", tags="orb_trail")

        if len(self._orb_history_map) >= 2:
            flat = [coord for pt in self._orb_history_map for coord in pt]
            cnv.create_line(*flat, fill="#4488cc", width=1,
                            smooth=True, tags="orb_trail")

        for i, (gx_mm, gy_mm) in enumerate(self._orb_history_gantry):
            px, py = self._gantry_mm_to_canvas(gx_mm, gy_mm)
            if px is None:
                continue
            alpha_frac = (i + 1) / max(len(self._orb_history_gantry), 1)
            r_dot = max(2, int(3 * alpha_frac))
            shade = int(80 + alpha_frac * 175)
            color = f"#00{shade:02x}00"
            cnv.create_oval(px - r_dot, py - r_dot, px + r_dot, py + r_dot,
                            fill=color, outline="", tags="orb_gt_trail")

        if len(self._orb_history_gantry) >= 2:
            valid_pts = [self._gantry_mm_to_canvas(gx, gy)
                         for gx, gy in self._orb_history_gantry]
            valid_pts = [(px, py) for px, py in valid_pts if px is not None]
            if len(valid_pts) >= 2:
                flat_gt = [coord for pt in valid_pts for coord in pt]
                cnv.create_line(*flat_gt, fill="#44cc44", width=1,
                                smooth=True, tags="orb_gt_trail")

        if self._orb_history_map:
            cx, cy = self._orb_history_map[-1]
            R = 6
            cnv.create_oval(cx - R, cy - R, cx + R, cy + R,
                            fill="#00d4ff", outline="#ffffff", width=1,
                            tags="orb_est_dot")
            for dx, dy in ((-R*3, 0), (R*3, 0), (0, -R*3), (0, R*3)):
                ex, ey = cx + dx//3*2, cy + dy//3*2
                cnv.create_line(cx + dx//3, cy + dy//3, ex, ey,
                                fill="#00d4ff", width=1, tags="orb_est_dot")
            cnv.create_text(cx + R + 4, cy - R - 4,
                            text="EST", fill="#00d4ff",
                            font=("Consolas", 7, "bold"), anchor="w",
                            tags="orb_est_dot")

        if self._orb_history_gantry:
            gx_mm, gy_mm = self._orb_history_gantry[-1]
            px, py = self._gantry_mm_to_canvas(gx_mm, gy_mm)
            if px is not None:
                R = 6
                cnv.create_oval(px - R, py - R, px + R, py + R,
                                fill="#00ff88", outline="#ffffff", width=1,
                                tags="orb_gt_dot")
                cnv.create_text(px + R + 4, py - R - 4,
                                text="GT", fill="#00ff88",
                                font=("Consolas", 7, "bold"), anchor="w",
                                tags="orb_gt_dot")

        # ── Capture video frame if recording ─────────────────────────────
        # We call update_idletasks so the canvas has finished drawing before
        # we try to grab its pixels.
        self._orb_map_canvas.update_idletasks()
        self._maybe_capture_orb_video_frame()

    def _ref_px_to_canvas(self, ref_px_x: float, ref_px_y: float):
        if not hasattr(self, "_orb_ref_w") or self._orb_ref_w == 0:
            return None, None
        cx = ref_px_x / self._orb_ref_w  * ORB_MAP_W
        cy = ref_px_y / self._orb_ref_h  * ORB_MAP_H
        return cx, cy

    def _ref_px_to_lon_lat(self, ref_px_x: float, ref_px_y: float):
        if not hasattr(self, "_orb_ref_w") or self._orb_ref_w == 0:
            return None, None
        lon = (ref_px_x / self._orb_ref_w) * 360.0 - 180.0
        lat = 90.0 - (ref_px_y / self._orb_ref_h) * 180.0
        return lon, lat

    def _gantry_mm_to_canvas(self, gx_mm: float, gy_mm: float):
        if self._map_corner_br is None or self._map_corner_tl is None:
            return None, None
        tl_x, tl_y = self._map_corner_tl
        br_x, br_y = self._map_corner_br
        span_x = br_x - tl_x
        span_y = br_y - tl_y
        if abs(span_x) < 1e-6 or abs(span_y) < 1e-6:
            return None, None
        fx = (gy_mm - tl_y) / span_y
        fy = (gx_mm - tl_x) / span_x
        cx = fx * ORB_MAP_W
        cy = fy * ORB_MAP_H
        return cx, cy

    def _ref_px_to_gantry_mm(self, ref_px_x: float, ref_px_y: float):
        if self._map_corner_br is None or self._map_corner_tl is None:
            return None, None
        if not hasattr(self, "_orb_ref_w") or self._orb_ref_w == 0:
            return None, None
        fx = ref_px_x / self._orb_ref_w
        fy = ref_px_y / self._orb_ref_h
        tl_x, tl_y = self._map_corner_tl
        br_x, br_y = self._map_corner_br
        gx_mm = tl_x + fx * (br_x - tl_x)
        gy_mm = tl_y + fy * (br_y - tl_y)
        return gx_mm, gy_mm

    def _start_orb(self) -> None:
        self._log(f"[INFO ] Start ORB pressed — _ORB_AVAILABLE={_ORB_AVAILABLE}  "
                  f"_CV2_AVAILABLE={_CV2_AVAILABLE}  "
                  f"arducam={'connected' if self._arducam else 'None'}")
        if not _ORB_AVAILABLE:
            self._log("[ERROR] ORBStuff.py not importable.")
            return
        if not _CV2_AVAILABLE:
            self._log("[ERROR] OpenCV not available.")
            return
        if self._arducam is None:
            self._log("[WARN ] ArduCam not connected — connect camera first.")
            return

        if self._orb_map_photo is None:
            self._log("[INFO ] Basemap not yet loaded — attempting now...")
            ok = self._orb_load_basemap()
            if not ok:
                self._log("[WARN ] Reference map image not loaded — specify a valid path and try again.")
                return
        else:
            self._log("[INFO ] Basemap already loaded, skipping reload.")

        self._orb_running = True
        self._orb_start_btn.configure(state=tk.DISABLED)
        self._orb_stop_btn.configure(state=tk.NORMAL)
        self._orb_status_var.set("Running")
        self._orb_status_lbl.configure(fg="#a6e3a1")
        self._log("[INFO ] ORB localization started.")
        self._orb_dispatch_next_frame()

    def _stop_orb(self) -> None:
        self._orb_running = False
        self._orb_start_btn.configure(state=tk.NORMAL)
        self._orb_stop_btn.configure(state=tk.DISABLED)
        self._orb_status_var.set("Idle")
        self._orb_status_lbl.configure(fg="#a6adc8")
        self._log("[INFO ] ORB localization stopped.")

    def _orb_clear_history(self) -> None:
        self._orb_history_map.clear()
        self._orb_history_gantry.clear()
        self._orb_frame_count = 0
        self._orb_count_var.set("0")
        for var in (self._orb_est_lon_var, self._orb_est_lat_var,
                    self._orb_est_gx_var,  self._orb_est_gy_var,
                    self._orb_gt_gx_var,   self._orb_gt_gy_var):
            var.set("—")
        self._orb_redraw_map()
        self._log("[INFO ] ORB history cleared.")

    def _orb_dispatch_next_frame(self) -> None:
        if not self._orb_running or self._arducam is None:
            return
        if self._orb_worker is not None and self._orb_worker.is_alive():
            return
        ret, frame = self._arducam.read()
        if not ret:
            self._log("[WARN ] ORB: could not read ArduCam frame.")
            return
        self._orb_worker = ORBWorker(frame.copy(), self._orb_result_q)
        self._orb_worker.start()
        self._orb_status_var.set("Processing…")
        self._orb_status_lbl.configure(fg="#fab387")

    def _poll_orb(self) -> None:
        try:
            while True:
                item = self._orb_result_q.get_nowait()
                self._orb_handle_result(item)
        except queue.Empty:
            pass

        if self._orb_running:
            if self._orb_worker is None or not self._orb_worker.is_alive():
                self._orb_dispatch_next_frame()

    def _orb_handle_result(self, item: tuple) -> None:
        if item[0] == "err":
            self._log(f"[WARN ] ORB: {item[1]}")
            if self._orb_running:
                self._orb_status_var.set("No match")
                self._orb_status_lbl.configure(fg="#f38ba8")
            return

        _, center, ref_res = item
        cx_ref, cy_ref = center
        ref_w,  ref_h  = ref_res

        if np.isnan(cx_ref) or np.isnan(cy_ref):
            self._log("[WARN ] ORB: match returned NaN — not enough features.")
            if self._orb_running:
                self._orb_status_var.set("No match")
                self._orb_status_lbl.configure(fg="#f38ba8")
            return

        self._orb_frame_count += 1
        self._orb_count_var.set(str(self._orb_frame_count))

        self._orb_ref_w = ref_w
        self._orb_ref_h = ref_h

        canvas_x, canvas_y = self._ref_px_to_canvas(cx_ref, cy_ref)
        if canvas_x is not None:
            self._orb_history_map.append((canvas_x, canvas_y))

        lon, lat = self._ref_px_to_lon_lat(cx_ref, cy_ref)
        if lon is not None:
            self._orb_est_lon_var.set(f"{lon:+.3f}°")
            self._orb_est_lat_var.set(f"{lat:+.3f}°")

        est_gx, est_gy = self._ref_px_to_gantry_mm(cx_ref, cy_ref)
        if est_gx is not None:
            self._orb_est_gx_var.set(f"{est_gx:.1f}")
            self._orb_est_gy_var.set(f"{est_gy:.1f}")
        else:
            self._orb_est_gx_var.set("no cal")
            self._orb_est_gy_var.set("no cal")

        if self._gantry_worker is not None:
            pos, _ = self._gantry_worker.get_position()
            gt_gx, gt_gy = pos["x"], pos["y"]
            self._orb_gt_gx_var.set(f"{gt_gx:.1f}")
            self._orb_gt_gy_var.set(f"{gt_gy:.1f}")
            self._orb_history_gantry.append((gt_gx, gt_gy))
        else:
            self._orb_gt_gx_var.set("no gantry")
            self._orb_gt_gy_var.set("no gantry")

        self._orb_redraw_map()
        if self._orb_running:
            self._orb_status_var.set("Running")
            self._orb_status_lbl.configure(fg="#a6e3a1")

        self._log(
            f"[ORB  ] frame {self._orb_frame_count}  "
            f"ref=({cx_ref:.0f}, {cy_ref:.0f})  "
            f"lon={lon:+.2f}°  lat={lat:+.2f}°"
            + (f"  est_gantry=({est_gx:.1f}, {est_gy:.1f}) mm" if est_gx is not None else "")
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy(self):
        self.after_cancel(self._after_id)
        if self._track_runner is not None:
            self._track_runner.stop()
        if self._recording:
            self._stop_recording()
        if self._orb_video_recording:
            self._stop_orb_video()
        self._stop_orb()
        self._disconnect_rs()
        self._disconnect_arducam()
        self._disconnect_gantry()
        super().destroy()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    app = ExperimentApp()
    app.mainloop()


if __name__ == "__main__":
    main()