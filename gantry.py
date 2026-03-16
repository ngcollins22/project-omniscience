"""
GantryWorker — background serial driver for FoxAlien GRBL gantry.

The worker thread owns the serial port exclusively.  The main thread
communicates via:

    send() / jog() / goto() / home() / feed_hold() / soft_reset()
    get_position()  → ({"x", "y", "z"}, state_str)
    response_q      → str lines from GRBL (banner, ok, error, ALARM …)

GRBL status strings look like:
    <Idle|MPos:0.000,0.000,0.000|FS:0,0>
    <Run|MPos:12.500,0.000,-3.000|FS:1200,0>
"""

from __future__ import annotations

import queue
import re
import threading
import time
from typing import Optional

try:
    import serial
    _SERIAL_AVAILABLE = True
except ImportError:
    serial = None          # type: ignore[assignment]
    _SERIAL_AVAILABLE = False


class GantryWorker(threading.Thread):
    """Daemon thread that drives a GRBL controller over pyserial."""

    STATUS_INTERVAL = 0.2   # seconds between '?' status polls (5 Hz)
    READ_TIMEOUT    = 0.05  # serial readline timeout

    def __init__(self, port: str, baud: int, response_q: queue.Queue):
        super().__init__(daemon=True, name="GantryWorker")
        self._port       = port
        self._baud       = baud
        self._response_q = response_q

        self._cmd_q = queue.Queue()
        self._stop  = threading.Event()

        self._lock  = threading.Lock()
        self._pos   = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._state = "Unknown"

        self._serial: Optional["serial.Serial"] = None  # type: ignore[name-defined]

    # ── Lifecycle ──────────────────────────────────────────────────────────

    @staticmethod
    def available() -> bool:
        return _SERIAL_AVAILABLE

    def open(self) -> None:
        """Open the serial port.  Must be called before start()."""
        if not _SERIAL_AVAILABLE:
            raise RuntimeError("pyserial not installed — pip install pyserial")
        self._serial = serial.Serial(
            self._port, self._baud,
            timeout=self.READ_TIMEOUT,
            write_timeout=2.0,
        )
        # GRBL resets when DTR toggles on connect; wait for the boot banner
        time.sleep(2.0)
        self._serial.reset_input_buffer()

    def close(self) -> None:
        """Signal the worker to stop and close the port (any thread)."""
        self._stop.set()
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass

    # ── Public command API (all thread-safe) ───────────────────────────────

    def send(self, cmd: str) -> None:
        """Queue a newline-terminated GRBL command string."""
        self._cmd_q.put((cmd.rstrip("\n") + "\n").encode())

    def send_rt(self, byte: bytes) -> None:
        """Queue a single real-time command byte (!, ~, 0x18, 0x85 …)."""
        self._cmd_q.put(byte)

    # ── Motion helpers ─────────────────────────────────────────────────────

    def jog(self, axis: str, dist_mm: float, feed_mm_min: float) -> None:
        """Relative single-axis jog in machine coordinates (mm)."""
        self.send(f"$J=G91 G21 {axis}{dist_mm:+.3f} F{feed_mm_min:.0f}")

    def jog_cancel(self) -> None:
        """Cancel an in-progress jog without triggering a full reset."""
        self.send_rt(b"\x85")

    def goto(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        feed_mm_min: float = 1000.0,
        rapid: bool = True,
    ) -> None:
        """Absolute move.  rapid=True → G0 (rapid), rapid=False → G1 (feed)."""
        coords = "".join(
            f" {ax}{val:.3f}"
            for ax, val in (("X", x), ("Y", y), ("Z", z))
            if val is not None
        )
        if not coords:
            return
        if rapid:
            self.send(f"G90 G0{coords}")
        else:
            self.send(f"G90 G1{coords} F{feed_mm_min:.0f}")

    def home(self) -> None:
        self.send("$H")

    def feed_hold(self) -> None:
        self.send_rt(b"!")

    def cycle_start(self) -> None:
        self.send_rt(b"~")

    def soft_reset(self) -> None:
        self.send_rt(b"\x18")

    def get_position(self) -> tuple[dict, str]:
        """Return a copy of the current position dict and machine state string."""
        with self._lock:
            return dict(self._pos), self._state

    # ── Worker loop ────────────────────────────────────────────────────────

    # Matches both GRBL 0.9 (comma-separated) and 1.1 (pipe-separated) formats
    _STATUS_RE = re.compile(
        r"<(\w+)[|,]MPos:([-\d.]+),([-\d.]+),([-\d.]+)"
    )

    def run(self) -> None:
        last_poll = 0.0
        while not self._stop.is_set():
            # ── flush outbound queue ───────────────────────────────────────
            try:
                while True:
                    payload = self._cmd_q.get_nowait()
                    self._serial.write(payload)
            except queue.Empty:
                pass

            # ── periodic status request ────────────────────────────────────
            now = time.monotonic()
            if now - last_poll >= self.STATUS_INTERVAL:
                self._serial.write(b"?")
                last_poll = now

            # ── read available lines ───────────────────────────────────────
            try:
                while self._serial.in_waiting:
                    raw  = self._serial.readline()
                    line = raw.decode(errors="replace").strip()
                    if not line:
                        continue
                    if line.startswith("<"):
                        self._parse_status(line)
                    else:
                        self._response_q.put(line)
            except Exception as exc:
                self._response_q.put(f"SERIAL_ERROR: {exc}")
                self._stop.set()
                break

            time.sleep(0.01)

    def _parse_status(self, line: str) -> None:
        m = self._STATUS_RE.search(line)
        if not m:
            return
        with self._lock:
            self._state = m.group(1)
            self._pos   = {
                "x": float(m.group(2)),
                "y": float(m.group(3)),
                "z": float(m.group(4)),
            }
