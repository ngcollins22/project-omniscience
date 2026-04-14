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

# ── Display constants ─────────────────────────────────────────────────────────
RS_W         = 640   # RealSense D435 native resolution
RS_H         = 480
ARDUCAM_W    = 640
ARDUCAM_H    = 480
POLL_MS      = 50    # GUI refresh interval



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
        self._rs_sensor    = None      # depth sensor handle for set_option()
        self._rs_colorizer = None      # rs.colorizer instance
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

        # Per-device status StringVars, populated in _build_status_bar
        self._status_color: dict[str, tk.StringVar] = {}
        self._status_text:  dict[str, tk.StringVar] = {}

        self._build_ui()
        self._after_id = self.after(POLL_MS, self._poll_cameras)

    # ── Top-level layout ──────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_status_bar()
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)
        self._build_notebook()
        self._build_log()

    # ── Status bar (always visible) ───────────────────────────────────────────

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
        tab = tk.Frame(self._nb, bg="#1e1e2e")

        # ── Map Calibration ───────────────────────────────────────────────
        cal_sec = self._section(tab, "Map Calibration")
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

        # ── Ground Track ──────────────────────────────────────────────────
        gt_sec = self._section(tab, "Ground Track")

        src_row = tk.Frame(gt_sec, bg="#181825")
        src_row.pack(anchor="w", padx=10, pady=(6, 4))
        for text, val in [("Synthetic pass", "synthetic"), ("Load CSV", "csv")]:
            tk.Radiobutton(src_row, text=text, variable=self._gt_source_var,
                           value=val, command=self._on_gt_source_change,
                           bg="#181825", fg="#cdd6f4", selectcolor="#313244",
                           activebackground="#181825", font=("Consolas", 9)
                           ).pack(side=tk.LEFT, padx=(0, 16))

        # Container that holds exactly one of the two sub-frames
        sub_container = tk.Frame(gt_sec, bg="#181825")
        sub_container.pack(anchor="w", padx=10, pady=(0, 2))

        self._gt_synth_frame = tk.Frame(sub_container, bg="#181825")
        self._gt_synth_frame.pack(anchor="w")   # visible by default
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
        # hidden by default — shown when "Load CSV" is selected
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

        # ── Run Settings ──────────────────────────────────────────────────
        run_sec = self._section(tab, "Run Settings")

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

    # ── LiDAR tab (RealSense D435) ────────────────────────────────────────────

    def _tab_lidar(self) -> tk.Frame:
        tab = tk.Frame(self._nb, bg="#1e1e2e")

        # Connection
        conn = self._section(tab, "Connection")

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

        # Settings
        settings = self._section(tab, "Settings")

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

        # Preview
        preview = self._section(tab, "Preview")
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

        # Snap & Visualize
        snap_sec = self._section(tab, "Snap & Visualize")

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

        conn = self._section(tab, "Connection")
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

        preview = self._section(tab, "Preview")
        tk.Label(preview, text="RGB Feed", bg="#181825", fg="#89b4fa",
                 font=("Consolas", 8)).pack(pady=(4, 2))
        self._arducam_canvas = tk.Canvas(preview, width=ARDUCAM_W, height=ARDUCAM_H,
                                         bg="#11111b", highlightthickness=1,
                                         highlightbackground="#313244")
        self._arducam_canvas.pack(padx=6, pady=(0, 6))

        return tab

    # ── Gantry tab ────────────────────────────────────────────────────────────

    def _tab_gantry(self) -> tk.Frame:
        tab = tk.Frame(self._nb, bg="#1e1e2e")

        # ── Connection ────────────────────────────────────────────────────
        conn = self._section(tab, "Connection")

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

        # ── Position ──────────────────────────────────────────────────────
        pos = self._section(tab, "Position")

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

        # ── Jog controls ──────────────────────────────────────────────────
        jog = self._section(tab, "Jog")

        # Step size + feed rate row
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

        # Direction pad + Z column
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

        # ── Go To ─────────────────────────────────────────────────────────
        goto_sec = self._section(tab, "Go To")

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

        # ── Machine ───────────────────────────────────────────────────────
        mach = self._section(tab, "Machine")
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

        # Work-coordinate origin row
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

    # ── Log (below notebook, always visible) ─────────────────────────────────

    def _build_log(self):
        frame = tk.LabelFrame(self, text="  Log  ",
                              bg="#181825", fg="#89b4fa",
                              font=("Consolas", 9, "bold"),
                              relief=tk.FLAT, highlightthickness=1,
                              highlightbackground="#313244")
        frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        self._log_widget = scrolledtext.ScrolledText(
            frame, height=6, bg="#11111b", fg="#a6adc8",
            font=("Consolas", 8), relief=tk.FLAT,
            state=tk.DISABLED, wrap=tk.WORD,
            insertbackground="#cdd6f4")
        self._log_widget.pack(fill=tk.X, padx=4, pady=4)

    # ── Shared widget helpers ─────────────────────────────────────────────────

    def _section(self, parent, title: str) -> tk.LabelFrame:
        """Titled card, packed into its parent."""
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

        # pipeline.start() can block for several seconds — run off the main thread
        # so Tkinter stays responsive.  GUI updates are scheduled back via after().
        def _do_start():
            try:
                pipeline = rs.pipeline()
                rs_cfg   = rs.config()
                rs_cfg.enable_stream(rs.stream.depth,
                                     cfg_rs.resolution_x, cfg_rs.resolution_y,
                                     rs.format.z16,  cfg_rs.fps)
                rs_cfg.enable_stream(rs.stream.color,
                                     cfg_rs.resolution_x, cfg_rs.resolution_y,
                                     rs.format.rgb8, cfg_rs.fps)
                profile    = pipeline.start(rs_cfg)
                dev        = profile.get_device()
                sensor     = dev.first_depth_sensor()
                colorizer  = rs.colorizer()
                serial     = dev.get_info(rs.camera_info.serial_number)
                name       = dev.get_info(rs.camera_info.name)
                fw         = dev.get_info(rs.camera_info.firmware_version)
                self.after(0, lambda: self._on_rs_connected(
                    pipeline, sensor, colorizer, serial, name, fw))
            except Exception as e:
                err = str(e)
                self.after(0, lambda: self._log(f"[ERROR] Connect failed: {err}"))

        threading.Thread(target=_do_start, daemon=True).start()

    def _on_rs_connected(self, pipeline, sensor, colorizer, serial, name, fw):
        """Called on the main thread once pipeline.start() succeeds."""
        self._log(f"[OK   ] {name}  serial={serial}  fw={fw}")
        self._rs_pipeline  = pipeline
        self._rs_sensor    = sensor
        self._rs_colorizer = colorizer
        self._rs_serial    = serial
        self._rs_serial_var.set(serial)
        self._set_status("LiDAR", True)

    def _disconnect_rs(self):
        # Clear handles immediately so polling stops and GUI is responsive.
        # pipeline.stop() can block for seconds, so run it in the background.
        pipeline = self._rs_pipeline
        self._rs_pipeline  = None
        self._rs_sensor    = None
        self._rs_colorizer = None
        self._rs_serial_var.set("—")
        self._set_status("LiDAR", False)

        if pipeline:
            self._log("[INFO ] RealSense disconnecting...")
            def _do_stop():
                try:
                    pipeline.stop()
                except Exception:
                    pass
                self.after(0, lambda: self._log("[INFO ] RealSense disconnected."))
            threading.Thread(target=_do_stop, daemon=True).start()

    def _apply_rs_settings(self):
        if self._rs_sensor is None:
            self._log("[WARN ] Not connected — connect first.")
            return
        try:
            power    = int(self._rs_settings["Emitter Power"].get())
            exposure = int(self._rs_settings["Exposure (µs)"].get())
        except ValueError:
            self._log("[ERROR] Settings must be integers.")
            return
        # Both calls happen on the main thread between poll cycles — no deadlock risk
        # because wait_for_frames() is also on the main thread (Tkinter is single-threaded).
        try:
            self._rs_sensor.set_option(rs.option.emitter_enabled, 1)
            self._rs_sensor.set_option(rs.option.laser_power, power)
            if exposure == 0:
                self._rs_sensor.set_option(rs.option.enable_auto_exposure, 1)
                self._log(f"[OK   ] Settings applied — emitter power: {power}, exposure: auto")
            else:
                self._rs_sensor.set_option(rs.option.enable_auto_exposure, 0)
                self._rs_sensor.set_option(rs.option.exposure, exposure)
                self._log(f"[OK   ] Settings applied — emitter power: {power}, exposure: {exposure} µs")
        except Exception as e:
            self._log(f"[ERROR] Apply settings failed: {e}")

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

    def _disconnect_arducam(self):
        if self._arducam:
            self._arducam.release()
            self._arducam = None
            self._log("[INFO ] ArduCam disconnected.")
        self._set_status("ArduCam", False)

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
            "Prior": ("Z", +1),   # Page Up
            "Next":  ("Z", -1),   # Page Down
        }
        if event.keysym in key_map:
            axis, sign = key_map[event.keysym]
            self._jog(axis, sign)
            return "break"
        return ""

    def _poll_gantry(self) -> None:
        # Drain GRBL response lines → log
        try:
            while True:
                line = self._gantry_resp_q.get_nowait()
                if line.startswith("SERIAL_ERROR:"):
                    self._log(f"[ERROR] {line}")
                    self._disconnect_gantry()
                    return
                elif line == "ok":
                    pass   # silent ack
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

        # Update position labels
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

        # Build calibration object
        if self._map_corner_br is not None and self._map_corner_tl is not None:
            corner_br = self._map_corner_br
            corner_tl = self._map_corner_tl
        else:
            # Dry-run fallback: span full gantry travel
            g  = self._cfg.gantry
            tx = g.travel_x_mm or 400.0
            ty = g.travel_y_mm or 800.0
            corner_br = (tx, ty)
            corner_tl = (0.0, 0.0)
            self._log("[WARN ] No calibration recorded — using config travel limits.")

        region = self._get_dem_region_bounds()
        if region is None:
            track = self._ground_track
            if track and len(track) > 0:
                wps = track.waypoints
                region = DemRegionBounds(
                    lat_min_deg=min(w.lat_deg for w in wps),
                    lat_max_deg=max(w.lat_deg for w in wps),
                    lon_min_deg=min(w.lon_deg for w in wps),
                    lon_max_deg=max(w.lon_deg for w in wps),
                )
                self._log("[INFO ] dem.region not configured — derived bounds from track extents.")
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

            self._ground_track = track
            path_len = track.path_length_deg()
            self._gt_status_var.set(f"{len(track)} waypoints — path {path_len:.4f}°")
            self._log(f"[OK   ] Ground track loaded: {len(track)} waypoints, "
                      f"path length {path_len:.4f}°")
        except Exception as exc:
            self._log(f"[ERROR] Ground track: {exc}")

    def _get_dem_region_bounds(self):
        """Return a DemRegionBounds from config, or None if not fully configured."""
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
        # RealSense D435 — poll directly on the main thread (no worker thread).
        # wait_for_frames(timeout_ms=1) returns immediately if no frame is ready,
        # avoiding all GIL / daemon-thread delivery issues on Windows.
        if self._rs_pipeline is not None:
            try:
                frameset    = self._rs_pipeline.wait_for_frames(timeout_ms=1)
                depth_frame = frameset.get_depth_frame()
                color_frame = frameset.get_color_frame()
                if depth_frame and color_frame:
                    depth_colorized = np.asanyarray(
                        self._rs_colorizer.colorize(depth_frame).get_data()
                    ).copy()
                    color_rgb = np.asanyarray(color_frame.get_data()).copy()
                    depth_m   = np.asanyarray(
                        depth_frame.get_data()
                    ).astype(np.float32) / 1000.0
                    depth_m[depth_m == 0] = np.nan
                    self._last_depth_m   = depth_m
                    self._last_grayscale = color_rgb
                    self._last_depth_rgb = depth_colorized
                    self._draw(self._depth_canvas, depth_colorized, RS_W, RS_H)
                    self._draw(self._gray_canvas,  color_rgb,       RS_W, RS_H)
            except RuntimeError:
                pass   # no frame ready within 1 ms — normal between frames
            except Exception as e:
                self._log(f"[ERROR] RealSense: {e}")
                self._disconnect_rs()

        # ArduCam
        if self._arducam is not None:
            ret, frame = self._arducam.read()
            if ret:
                self._draw(self._arducam_canvas,
                           cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                           ARDUCAM_W, ARDUCAM_H)
                self._draw_crosshair(self._arducam_canvas, ARDUCAM_W, ARDUCAM_H)

        self._poll_gantry()
        self._after_id = self.after(POLL_MS, self._poll_cameras)

    def _draw(self, canvas: tk.Canvas, arr: np.ndarray, w: int, h: int):
        img   = Image.fromarray(arr).resize((w, h), Image.NEAREST)
        photo = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas._photo = photo   # keep reference — Tkinter GCs it otherwise

    def _draw_crosshair(self, canvas: tk.Canvas, w: int, h: int,
                        size: int = 20, gap: int = 6) -> None:
        """Draw a centre crosshair over the canvas image (redrawn each frame)."""
        cx, cy = w // 2, h // 2
        canvas.delete("crosshair")
        kw = dict(fill="#00ff88", width=1, tags="crosshair")
        # horizontal arms
        canvas.create_line(cx - size, cy, cx - gap, cy, **kw)
        canvas.create_line(cx + gap,  cy, cx + size, cy, **kw)
        # vertical arms
        canvas.create_line(cx, cy - size, cx, cy - gap, **kw)
        canvas.create_line(cx, cy + gap,  cx, cy + size, **kw)
        # centre dot
        canvas.create_oval(cx - 1, cy - 1, cx + 1, cy + 1,
                           fill="#00ff88", outline="", tags="crosshair")

    # ── Snap & Visualize ──────────────────────────────────────────────────────

    def _snap_and_plot(self):
        if not _MPL_AVAILABLE:
            self._log("[ERROR] matplotlib not available — pip install matplotlib")
            return
        if not _PROJECTION_AVAILABLE:
            self._log("[ERROR] realsense_projection module not found — check Experiments/simulation/")
            return
        if self._last_depth_m is None:
            self._log("[WARN ] No frame buffered yet — connect and wait for the preview to start.")
            return

        try:
            altitude_mm = float(self._snap_alt_var.get())
        except ValueError:
            self._log("[ERROR] Altitude must be a number.")
            return

        # ── Snapshot the current frame arrays ────────────────────────────────
        _H = self._cfg.realsense.resolution_y
        _W = self._cfg.realsense.resolution_x
        depth_m   = self._last_depth_m.copy()
        color_rgb = (self._last_grayscale.copy()
                     if self._last_grayscale is not None
                     else np.zeros((_H, _W, 3), dtype=np.uint8))
        depth_rgb = (self._last_depth_rgb.copy()
                     if self._last_depth_rgb is not None
                     else np.zeros((_H, _W, 3), dtype=np.uint8))

        # ── Pixel-level statistics ────────────────────────────────────────────
        total_px  = depth_m.size                                    # 9600
        raw_valid = np.isfinite(depth_m)
        nan_px    = int((~raw_valid).sum())

        # Raw depth distribution BEFORE any filtering
        raw_vals_mm = depth_m[raw_valid] * 1000.0
        n_zero = int((depth_m == 0.0).sum())   # exact-zero returns (phantom artifact)
        if raw_vals_mm.size > 0:
            raw_p5  = float(np.percentile(raw_vals_mm,  5))
            raw_p25 = float(np.percentile(raw_vals_mm, 25))
            raw_p50 = float(np.percentile(raw_vals_mm, 50))
            raw_p75 = float(np.percentile(raw_vals_mm, 75))
            raw_p95 = float(np.percentile(raw_vals_mm, 95))
            raw_min = float(raw_vals_mm.min())
            raw_max = float(raw_vals_mm.max())
        else:
            raw_p5 = raw_p25 = raw_p50 = raw_p75 = raw_p95 = raw_min = raw_max = float("nan")

        min_range_cfg = self._cfg.realsense.min_range_mm or 0
        min_z_m   = min_range_cfg / 1000.0
        max_z_m   = 3.0 * altitude_mm / 1000.0
        too_close = raw_valid & (depth_m < min_z_m)
        too_far   = raw_valid & (depth_m > max_z_m)
        surviving = raw_valid & (depth_m >= min_z_m) & (depth_m < max_z_m)

        n_close = int(too_close.sum())
        n_far   = int(too_far.sum())
        n_surv  = int(surviving.sum())

        if surviving.any():
            z_surv_mm = depth_m[surviving] * 1000.0
            z_min_mm  = float(z_surv_mm.min())
            z_max_mm  = float(z_surv_mm.max())
            z_med_mm  = float(np.median(z_surv_mm))
        else:
            z_min_mm = z_max_mm = z_med_mm = float("nan")

        # ── Project ───────────────────────────────────────────────────────────
        self._log(f"[INFO ] Projecting at altitude {altitude_mm:.1f} mm...")
        try:
            model = RealSenseProjectionModel(self._cfg.realsense, altitude_mm)
            patch, x_grid, y_grid = model.project_depth(depth_m, altitude_mm)
        except Exception as e:
            self._log(f"[ERROR] Projection failed: {e}")
            return

        total_cells = patch.size
        valid_cells = int(np.isfinite(patch).sum())
        elev_min    = float(np.nanmin(patch)) if valid_cells > 0 else float("nan")
        elev_max    = float(np.nanmax(patch)) if valid_cells > 0 else float("nan")

        self._log(
            f"[OK   ] surviving {n_surv}/{total_px} px "
            f"({100*n_surv/total_px:.1f}%) | "
            f"NaN {nan_px} | <min {n_close} | >max {n_far} | "
            f"Z [{z_min_mm:.0f}, {z_max_mm:.0f}] mm | "
            f"elev [{elev_min:.1f}, {elev_max:.1f}] mm"
        )

        # ── Build filtered-Z RGB image via plasma colormap (no matplotlib axes) ──
        vmin_z = z_min_mm if surviving.any() else 0.0
        vmax_z = z_max_mm if surviving.any() else 1.0
        z_norm = np.where(surviving,
                          (depth_m * 1000.0 - vmin_z) / max(vmax_z - vmin_z, 1e-9),
                          0.0)
        z_norm    = np.clip(z_norm, 0.0, 1.0)
        z_rgba    = _mpl_cm.plasma(z_norm)
        z_rgb_img = (z_rgba[:, :, :3] * 255).astype(np.uint8)
        z_rgb_img[~surviving] = [17, 17, 27]   # dark for invalid pixels

        # ── Build window ──────────────────────────────────────────────────────
        self._close_snap_window()
        win = tk.Toplevel(self)
        win.title("RealSense D435 — Debug Snapshot")
        win.configure(bg="#1e1e2e")
        win.protocol("WM_DELETE_WINDOW", self._close_snap_window)
        self._snap_window = win

        # ── Top: three PIL/ImageTk canvases (same style as live preview) ─────
        SNAP_W, SNAP_H = 320, 240   # 640×480 at 0.5× scale
        img_outer = tk.Frame(win, bg="#1e1e2e")
        img_outer.pack(fill=tk.X, padx=4, pady=(4, 2))

        for label_text, arr in [
            ("Color (RGB)", color_rgb),
            ("Depth colormap (camera)", depth_rgb),
            (f"Filtered Z  [{vmin_z:.0f} – {vmax_z:.0f} mm]", z_rgb_img),
        ]:
            col_f = tk.Frame(img_outer, bg="#1e1e2e")
            col_f.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            tk.Label(col_f, text=label_text, bg="#1e1e2e", fg="#89b4fa",
                     font=("Consolas", 8)).pack(pady=(0, 2))
            cnv = tk.Canvas(col_f, width=SNAP_W, height=SNAP_H,
                            bg="#11111b", highlightthickness=1,
                            highlightbackground="#313244")
            cnv.pack()
            photo = ImageTk.PhotoImage(
                Image.fromarray(arr).resize((SNAP_W, SNAP_H), Image.NEAREST))
            cnv.create_image(0, 0, anchor=tk.NW, image=photo)
            cnv._photo = photo   # keep reference alive

        # ── Bottom: stats sidebar (left) + 3D plot (right) ───────────────────
        bot_frame = tk.Frame(win, bg="#1e1e2e")
        bot_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 4))

        # Stats sidebar ───────────────────────────────────────────────────────
        n_finite = total_px - nan_px
        stats_text = (
            "── Raw depth (no filter) ────\n"
            f"Finite:     {n_finite}  ({100*n_finite/total_px:.1f}%)\n"
            f"Zeros:      {n_zero}  ({100*n_zero/total_px:.1f}%)\n"
            f"Min:        {raw_min:.1f}\n"
            f"P5:         {raw_p5:.1f}\n"
            f"P25:        {raw_p25:.1f}\n"
            f"Median:     {raw_p50:.1f}\n"
            f"P75:        {raw_p75:.1f}\n"
            f"P95:        {raw_p95:.1f}\n"
            f"Max:        {raw_max:.1f}\n"
            "\n"
            "── Filtered pixels ─────────\n"
            f"Total:      {total_px}  ({_W}×{_H})\n"
            f"NaN:        {nan_px}  ({100*nan_px/total_px:.1f}%)\n"
            f"            invalid/out-of-range\n"
            f"<{min_range_cfg:.0f} mm:   {n_close}  ({100*n_close/total_px:.1f}%)\n"
            f"            near-field phantom\n"
            f">{max_z_m*1000:.0f} mm:  {n_far}  ({100*n_far/total_px:.1f}%)\n"
            f"            far-field clip\n"
            f"Surviving:  {n_surv}  ({100*n_surv/total_px:.1f}%)\n"
            "\n"
            "── Z stats (mm) ────────────\n"
            f"Z min:      {z_min_mm:.1f}\n"
            f"Z max:      {z_max_mm:.1f}\n"
            f"Z median:   {z_med_mm:.1f}\n"
            f"Altitude:   {altitude_mm:.0f}\n"
            f"Max clip:   {max_z_m*1000:.0f}\n"
            "\n"
            "── Projection ──────────────\n"
            f"Patch:      {patch.shape[1]}×{patch.shape[0]}\n"
            f"Valid:      {valid_cells:,}  ({100*valid_cells/total_cells:.1f}%)\n"
            f"Elev min:   {elev_min:.1f}\n"
            f"Elev max:   {elev_max:.1f}\n"
            "\n"
            "── Camera settings ─────────\n"
            f"Emitter:    {self._rs_settings['Emitter Power'].get()}\n"
            f"Exposure:   {self._rs_settings['Exposure (µs)'].get()} µs\n"
            f"Serial:     {self._rs_serial}"
        )
        n_lines = stats_text.count("\n") + 1

        sf = tk.Frame(bot_frame, bg="#11111b", padx=8, pady=6)
        sf.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        stats_w = tk.Text(sf, bg="#11111b", fg="#cdd6f4",
                          font=("Consolas", 8), relief=tk.FLAT,
                          width=30, height=n_lines, wrap=tk.NONE,
                          highlightthickness=0)
        stats_w.insert(tk.END, stats_text)
        stats_w.configure(state=tk.DISABLED)
        stats_w.pack(fill=tk.Y, expand=True)

        # 3D surface plot ─────────────────────────────────────────────────────
        fig = Figure(figsize=(8, 5), facecolor="#1e1e2e")
        fig.subplots_adjust(left=0.08, right=0.88, top=0.96, bottom=0.10)
        ax_3d = fig.add_subplot(111, projection="3d")
        ax_3d.set_facecolor("#181825")

        if valid_cells > 0:
            XX, YY = np.meshgrid(x_grid, y_grid)
            surf = ax_3d.plot_surface(XX, YY, patch,
                                      cmap="viridis", linewidth=0, antialiased=True,
                                      vmin=elev_min, vmax=elev_max)
            cbar = fig.colorbar(surf, ax=ax_3d, shrink=0.4, pad=0.08)
            cbar.set_label("mm", color="#a6adc8", fontsize=8)
            cbar.ax.yaxis.set_tick_params(color="#a6adc8", labelsize=7)
            for lbl in cbar.ax.yaxis.get_ticklabels():
                lbl.set_color("#a6adc8")
        else:
            ax_3d.text2D(0.5, 0.5, "No valid cells to plot",
                         transform=ax_3d.transAxes,
                         ha="center", va="center",
                         color="#f38ba8", fontsize=12)

        ax_3d.set_xlabel("X (mm)", color="#a6adc8", labelpad=6)
        ax_3d.set_ylabel("Y (mm)", color="#a6adc8", labelpad=6)
        ax_3d.set_zlabel("Elev (mm)", color="#a6adc8", labelpad=6)
        ax_3d.tick_params(colors="#a6adc8", labelsize=7)

        plot_frame = tk.Frame(bot_frame, bg="#1e1e2e")
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        canvas_3d = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas_3d.draw()
        canvas_3d.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas_3d, plot_frame)
        toolbar.update()

    def _close_snap_window(self):
        if self._snap_window is not None:
            try:
                self._snap_window.destroy()
            except Exception:
                pass
            self._snap_window = None

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy(self):
        self.after_cancel(self._after_id)
        if self._track_runner is not None:
            self._track_runner.stop()
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
