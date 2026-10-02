"""
stream.py
---------
Video sources for the pipeline. Lets the same detector read from a
recorded file, a laptop webcam, or a phone used as a wireless camera
(via Camo, IP Webcam, or similar) -- anything OpenCV can open as a
camera index or a stream URL.
"""

import platform
import threading
import time

import cv2


def parse_source(source):
    """'1' -> 1 (camera index). Anything else (path / URL) is unchanged."""
    if isinstance(source, str) and source.strip().isdigit():
        return int(source.strip())
    return source


def is_live_source(source) -> bool:
    """True for webcams and network streams, False for video files."""
    if isinstance(source, int):
        return True
    return str(source).lower().startswith(("http://", "https://", "rtsp://", "rtmp://"))


def _open_capture(source):
    """
    Open a cv2.VideoCapture, using the DSHOW backend for webcam indices
    on Windows -- needed for virtual cameras like Camo, which the
    default backend sometimes fails to open even though available.
    """
    if isinstance(source, int) and platform.system() == "Windows":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(source)


class LiveFrameReader:
    """
    Reads a live stream in a background thread and always returns the
    latest frame, dropping stale ones -- keeps a slow detector from
    falling further and further behind a live camera.
    """

    def __init__(self, source):
        self.source = source
        self.cap = _open_capture(source)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._last_returned = 0
        self._running = self.cap.isOpened()
        self._thread = None

        if self._running:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def _loop(self):
        failures = 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                failures += 1
                if failures > 100:
                    self._running = False
                    break
                time.sleep(0.05)
                continue
            failures = 0
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def read(self, timeout: float = 5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None and self._frame_id != self._last_returned:
                    self._last_returned = self._frame_id
                    return True, self._frame.copy()
            if not self._running:
                return False, None
            time.sleep(0.005)
        return False, None

    def get(self, prop):
        return self.cap.get(prop)

    def release(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()


def open_source(source):
    """Open any supported source and return (reader, is_live)."""
    source = parse_source(source)
    if is_live_source(source):
        return LiveFrameReader(source), True
    return cv2.VideoCapture(source), False