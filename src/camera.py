"""카메라 입력 공용 모듈 (라즈베리파이 CSI 카메라 / 웹캠 / 영상 파일).

모든 소스는 cv2.VideoCapture와 같은 read()/get()/release()/isOpened() 형태이고,
추가로 `last_meta`(직전 프레임의 카메라 메타데이터 dict)를 가진다.
Picamera2만 메타데이터를 채운다 (웹캠/파일은 빈 dict).

진단에 쓰는 Picamera2 메타데이터
    SensorTimestamp : 센서가 이 프레임을 찍은 시각 (ns). 연속 프레임 간격으로 건너뛴 프레임 수를 계산한다.
    FrameDuration   : 카메라의 프레임 간격 (us). 15FPS면 약 66666.
    ExposureTime    : 노출 시간 (us). 길면 움직이는 사람이 흐리게 찍혀 모델이 놓치기 쉽다.
"""
import time

import cv2


def now_ns() -> int:
    """카메라 SensorTimestamp와 비교할 수 있는 현재 시각 (ns).

    라즈베리파이(리눅스)는 절전(suspend)을 하지 않으므로 BOOTTIME과 MONOTONIC이 사실상 같다.
    """
    if hasattr(time, "CLOCK_BOOTTIME"):
        return time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    return time.monotonic_ns()


class PiCamera:
    """Picamera2를 cv2.VideoCapture와 비슷한 형태로 감싼다."""

    def __init__(self, width: int, height: int, fps: int, low_latency: bool = False):
        from picamera2 import Picamera2  # 라즈베리파이에서만 필요

        self.cam = Picamera2()
        # 주의: Picamera2(libcamera) 포맷 이름은 직관과 반대다.
        #   "RGB888" -> 배열이 [B, G, R] 순서 (OpenCV가 기대하는 순서)
        #   "BGR888" -> 배열이 [R, G, B] 순서 (빨강/파랑이 뒤바뀜)
        # 출처: Picamera2 Library Guide, "Image Formats"
        # 여기서 RGB888을 쓰므로 뒤에서 cvtColor를 추가로 하면 안 된다 (다시 뒤바뀜).
        controls = {"FrameRate": fps} if fps > 0 else {}
        extra = {}
        if low_latency:
            # 처리 속도가 카메라보다 느리면 기본 설정(버퍼 6개, 큐 사용)은 오래된 프레임을
            # 쌓아 두었다가 넘겨줄 수 있다 -> 지연 증가. 버퍼를 줄이고 큐를 끈다.
            extra = {"buffer_count": 2, "queue": False}
        cfg = self.cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls=controls,
            **extra,
        )
        self.cam.configure(cfg)
        self.cam.start()
        self.last_meta = {}

    def isOpened(self) -> bool:
        return True

    def read(self):
        """프레임과 함께 그 프레임의 메타데이터(SensorTimestamp 등)를 last_meta에 저장한다."""
        try:
            req = self.cam.capture_request()
            try:
                frame = req.make_array("main")  # 복사본이라 release 후에도 안전
                self.last_meta = req.get_metadata()
            finally:
                req.release()
            return True, frame
        except Exception:
            self.last_meta = {}
            return False, None

    def get(self, _prop) -> float:
        return 0.0

    def release(self) -> None:
        self.cam.stop()
        self.cam.close()


def open_source(source: str, width: int, height: int, fps: int, low_latency: bool = False):
    """'picamera', 웹캠 번호, 영상 파일/스트림 경로를 연다."""
    if source == "picamera":
        return PiCamera(width, height, fps, low_latency)
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
