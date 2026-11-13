"""처리 화면을 웹 브라우저로 실시간 전송한다 (MJPEG over HTTP). SSH 환경에서 화면 확인용.

    person_mask.py --stream 8000  ->  같은 네트워크의 PC 브라우저에서 http://<라즈베리파이주소>:8000

- 메인 루프는 update()로 최신 화면만 넘기고, JPEG 압축은 접속한 브라우저마다 별도 스레드에서 한다.
  (브라우저가 없으면 압축도 하지 않아 측정에 주는 영향이 거의 없다)
- 분할 화면이면 원본도 전송된다. 개발용으로만 쓰고, 실제 장치에서는 --single 과 함께 쓸 것.
- 인증이 없으므로 같은 네트워크의 누구나 볼 수 있다. 신뢰할 수 있는 네트워크(개인 핫스팟 등)에서만 사용.
"""
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>DONT LOOK ME live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif;text-align:center}
img{max-width:100%;height:auto;margin-top:8px}p{margin:6px;font-size:14px}</style></head>
<body><p>DONT LOOK ME 실시간 화면 (새로고침하면 다시 연결)</p><img src="/stream.mjpg"></body></html>"""


class _DualStackServer(ThreadingHTTPServer):
    """IPv4와 IPv6 접속을 모두 받는다. (아이폰 핫스팟 등에서는 PC가 IPv6로만 파이에 닿는 경우가 있음)"""
    address_family = socket.AF_INET6
    daemon_threads = True

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


class MJPEGStreamer:
    def __init__(self, port: int, quality: int = 80):
        self.quality = quality
        self.frame = None
        self.seq = 0
        self.cond = threading.Condition()
        self.running = True
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # 접속 로그로 터미널을 어지럽히지 않는다
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path != "/stream.mjpg":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last = -1
                try:
                    while streamer.running:
                        with streamer.cond:
                            streamer.cond.wait_for(lambda: streamer.seq != last or not streamer.running,
                                                   timeout=2.0)
                            frame, last = streamer.frame, streamer.seq
                        if frame is None:
                            continue
                        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, streamer.quality])
                        if not ok:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg.tobytes())
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # 브라우저 탭을 닫음

        try:
            self.server = _DualStackServer(("::", port), Handler)
        except OSError:  # IPv6를 못 쓰는 환경
            self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
            self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def update(self, frame) -> None:
        """최신 화면을 넘긴다 (복사하지 않음: 호출하는 쪽이 매 프레임 새 배열을 만든다)."""
        with self.cond:
            self.frame = frame
            self.seq += 1
            self.cond.notify_all()

    def close(self) -> None:
        self.running = False
        with self.cond:
            self.cond.notify_all()
        self.server.shutdown()
        self.server.server_close()
