"""Minimal MJPEG-over-HTTP streaming, shared by the live viewer and the demo recorder.

Streaming to a browser rather than opening a window is deliberate: it works
across the WSL boundary with no X server, and it lets you watch from Windows
while the process runs in Linux.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#141A22; color:#F7F6F2;
         font-family:'IBM Plex Sans',-apple-system,Segoe UI,sans-serif;
         display:flex; flex-direction:column; align-items:center; gap:14px; padding:20px; }}
  h1 {{ font-size:20px; font-weight:600; margin:0; letter-spacing:.3px; }}
  p  {{ margin:0; font-size:13px; color:#75808F; }}
  img {{ image-rendering:pixelated; max-width:100%; border-radius:10px;
         border:1px solid #2C3846; }}
</style></head>
<body>
  <h1>{title}</h1>
  <p>{subtitle}</p>
  <img src="/stream" alt="live view">
</body></html>
"""


class FrameSlot:
    """Latest-frame mailbox. Slow clients skip frames instead of backing up."""

    def __init__(self) -> None:
        self._jpeg: bytes | None = None
        self._cond = threading.Condition()
        self._seq = 0
        self.stop = False

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def publish_bgr(self, frame: np.ndarray, quality: int = 82) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            self.publish(buf.tobytes())

    def wait_for(self, last_seq: int, timeout: float = 5.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._jpeg, self._seq

    def shutdown(self) -> None:
        with self._cond:
            self.stop = True
            self._cond.notify_all()


def make_handler(slot: FrameSlot, title: str, subtitle: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # keep the console clean
            pass

        def do_GET(self):  # noqa: N802
            if self.path == "/":
                body = PAGE.format(title=title, subtitle=subtitle).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path != "/stream":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frameboundary"
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                while not slot.stop:
                    jpeg, seq = slot.wait_for(seq)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frameboundary\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def serve(
    slot: FrameSlot, host: str, port: int, title: str, subtitle: str
) -> ThreadingHTTPServer:
    """Start the server on a daemon thread and return it (call .shutdown() to stop)."""
    server = ThreadingHTTPServer((host, port), make_handler(slot, title, subtitle))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
