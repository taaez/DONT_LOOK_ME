"""YOLOv8-seg로 사람 영역을 찾아 불투명 단색으로 가린다. (PC / 라즈베리파이 5 공용)

- 사람 클래스(COCO id 0)만 사용한다. 개/고양이 등은 가리지 않는다.
- 마스크는 반드시 같은 프레임의 추론 결과만 쓴다 (이전 프레임 마스크 재사용 금지).
- 마스크 경계를 dilate로 넓혀서 윤곽 밖으로 삐져나오는 몸을 줄인다.
- 추론/카메라 오류가 나면 원본을 내보내지 않고 검은 화면을 낸다 (fail-safe).

사용법
    python src/person_mask.py --source 0                     # USB 웹캠 (왼쪽 원본 | 오른쪽 마스크 분할 화면)
    python src/person_mask.py --source picamera              # 라즈베리파이 CSI 카메라 (Picamera2), 분할 화면
    python src/person_mask.py --source video.mp4
    python src/person_mask.py --source image.jpg --save out.jpg
    python src/person_mask.py --source picamera --single --fullscreen --fps 15 --imgsz 320   # 라즈베리파이 출력용 (마스크만)
    python src/person_mask.py --source 0 --no-show           # 화면 없이 FPS만 측정

라즈베리파이에서 빠르게 돌리려면 NCNN으로 변환해서 쓴다 (한 번만 실행).
    python src/person_mask.py --export ncnn --imgsz 320
    python src/person_mask.py --source picamera --weights weights/yolov8n-seg_ncnn_model --imgsz 320
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "yolov8n-seg.pt"
PERSON_CLASS = 0
WINDOW = "person_mask"


class PiCamera:
    """Picamera2를 cv2.VideoCapture와 비슷한 read()/release() 형태로 감싼다."""

    def __init__(self, width: int, height: int, fps: int):
        from picamera2 import Picamera2  # 라즈베리파이에서만 필요

        self.cam = Picamera2()
        # 주의: Picamera2(libcamera) 포맷 이름은 직관과 반대다.
        #   "RGB888" -> capture_array() 배열이 [B, G, R] 순서 (OpenCV가 기대하는 순서)
        #   "BGR888" -> capture_array() 배열이 [R, G, B] 순서 (빨강/파랑이 뒤바뀜)
        # 출처: Picamera2 Library Guide, "Image Formats"
        # 여기서 RGB888을 쓰므로 뒤에서 cvtColor를 추가로 하면 안 된다 (다시 뒤바뀜).
        controls = {"FrameRate": fps} if fps > 0 else {}
        cfg = self.cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls=controls,
        )
        self.cam.configure(cfg)
        self.cam.start()

    def isOpened(self) -> bool:
        return True

    def read(self):
        try:
            return True, self.cam.capture_array()
        except Exception:
            return False, None

    def get(self, _prop) -> float:
        return 0.0

    def release(self) -> None:
        self.cam.stop()
        self.cam.close()


def mask_people(frame: np.ndarray, result, color: tuple, dilate_px: int) -> np.ndarray:
    """frame에서 사람 마스크 영역을 color로 채운 새 프레임을 반환한다."""
    out = frame.copy()
    if result.masks is None or len(result.masks) == 0:
        return out

    h, w = frame.shape[:2]
    union = np.zeros((h, w), dtype=np.uint8)
    # masks.xy: 원본 해상도 좌표의 폴리곤 -> 후처리(리사이즈)가 필요 없어 가볍다
    for poly in result.masks.xy:
        if len(poly) >= 3:
            cv2.fillPoly(union, [poly.astype(np.int32)], 255)

    if dilate_px > 0:
        k = 2 * dilate_px + 1
        union = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    out[union > 0] = color
    return out


def side_by_side(original: np.ndarray, masked: np.ndarray, max_width: int) -> np.ndarray:
    """왼쪽 원본 | 오른쪽 마스크 처리본을 한 화면으로 붙인다. 개발/검증용 화면이다."""
    view = np.hstack([original, masked])
    for text, x in (("ORIGINAL", 10), ("MASKED", original.shape[1] + 10)):
        cv2.putText(view, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    if max_width > 0 and view.shape[1] > max_width:
        scale = max_width / view.shape[1]
        view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return view


def export_model(args: argparse.Namespace) -> None:
    """라즈베리파이용 가벼운 포맷(ncnn/onnx)으로 변환한다. 결과는 weights 폴더 안에 생긴다."""
    path = YOLO(args.weights).export(format=args.export, imgsz=args.imgsz)
    print(f"변환 완료: {path}")


def open_source(args: argparse.Namespace):
    if args.source == "picamera":
        return PiCamera(args.width, args.height, args.fps)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return cap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0",
                   help="웹캠 번호, 'picamera', 영상/이미지 경로, 스트림 주소(http://, rtsp://)")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                   help=".pt / ncnn 모델 폴더 / .onnx")
    p.add_argument("--imgsz", type=int, default=640, help="라즈베리파이는 320~416 권장")
    p.add_argument("--conf", type=float, default=0.25,
                   help="낮출수록 놓치는 사람이 줄고 오검출이 늘어난다 (가림 용도라 낮게 시작)")
    p.add_argument("--dilate", type=int, default=8, help="마스크 경계 확장 픽셀")
    p.add_argument("--color", default="40,40,40", help="가림 색 B,G,R")
    p.add_argument("--device", default=None, help="cpu, 0 등 (기본: 자동)")
    p.add_argument("--width", type=int, default=640, help="카메라 입력 가로")
    p.add_argument("--height", type=int, default=480, help="카메라 입력 세로")
    p.add_argument("--fps", type=int, default=15, help="출력 FPS 상한 (0이면 제한 없음)")
    p.add_argument("--save", default=None, help="이미지/영상 저장 경로")
    p.add_argument("--single", action="store_true",
                   help="분할 화면 대신 마스크 처리본만 출력 (실제 장치 출력용, 원본이 화면에 나오지 않음)")
    p.add_argument("--view-width", type=int, default=1280, help="분할 화면 최대 가로 (0이면 축소 안 함)")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--fullscreen", action="store_true", help="전체화면 출력 (소형 디스플레이용)")
    p.add_argument("--export", choices=["ncnn", "onnx"], help="모델 변환 후 종료")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.export:
        export_model(args)
        return

    color = tuple(int(c) for c in args.color.split(","))
    model = YOLO(args.weights, task="segment")

    is_image = Path(args.source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

    def process(frame: np.ndarray) -> np.ndarray:
        try:
            res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, classes=[PERSON_CLASS],
                                device=args.device, verbose=False)[0]
            return mask_people(frame, res, color, args.dilate)
        except Exception as e:  # 실패 시 원본 노출 금지
            print(f"[오류] 추론 실패, 검은 화면 출력: {e}")
            return np.zeros_like(frame)

    if is_image:
        frame = cv2.imread(args.source)
        if frame is None:
            raise SystemExit(f"이미지를 읽을 수 없음: {args.source}")
        out = process(frame)
        view = out if args.single else side_by_side(frame, out, args.view_width)
        if args.save:
            cv2.imwrite(args.save, view)
            print(f"저장: {args.save}")
        if not args.no_show:
            cv2.imshow(WINDOW, view)
            cv2.waitKey(0)
        return

    cap = open_source(args)
    if not cap.isOpened():
        raise SystemExit(f"소스를 열 수 없음: {args.source}")

    if not args.no_show and args.fullscreen:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    is_file = Path(args.source).is_file()
    frame_interval = 1.0 / args.fps if args.fps > 0 else 0.0
    last_shape = (args.height, args.width, 3)

    # ---- 녹화 설정 ----
    # 실시간 소스(카메라)는 AI 처리 속도가 파일 FPS보다 느릴 수 있다.
    # 루프마다 1프레임만 쓰면 "실제 30초 -> 영상 10초"처럼 빨리감기가 된다.
    # 그래서 실제 경과 시간(벽시계)에 맞춰, 새 프레임이 나오기 전까지의 빈 시간은
    # 직전 프레임을 반복 기록해 채운다. (영상 길이 = 실제 녹화 시간)
    # 영상 파일 입력은 실시간이 아니므로 원본 FPS로 1:1 기록한다.
    if is_file:
        record_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps or 15
    else:
        record_fps = args.fps if args.fps > 0 else 15
    writer = None
    record_start = None   # 첫 프레임 캡처 시각
    written = 0           # 파일에 실제로 쓴 프레임 수 (반복 포함)
    duplicated = 0        # 시간 맞추기용으로 반복 기록한 프레임 수
    prev_view = None

    def write_until(t: float) -> None:
        """record_start부터 시각 t까지 파일 프레임이 부족하면 직전 프레임으로 채운다."""
        nonlocal written, duplicated
        target = int((t - record_start) * record_fps)
        while prev_view is not None and written < target:
            writer.write(prev_view)
            written += 1
            duplicated += 1

    n, t0, next_t = 0, time.perf_counter(), time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            t_cap = time.perf_counter()  # 이 프레임이 실제로 들어온 시각
            if not ok:
                if is_file:  # 영상 파일 끝
                    break
                # 카메라 프레임을 못 받으면 원본이 없으므로 검은 화면 (fail-safe)
                print("[오류] 프레임 읽기 실패, 검은 화면 출력")
                frame = np.zeros(last_shape, dtype=np.uint8)
                out = frame
            else:
                last_shape = frame.shape
                out = process(frame)
            view = out if args.single else side_by_side(frame, out, args.view_width)

            if args.save and writer is None:
                h, w = view.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), record_fps, (w, h))
                if not writer.isOpened():
                    raise SystemExit(f"영상 저장 파일을 열 수 없음: {args.save}")
                record_start = t_cap
                print(f"녹화 시작: {args.save} ({record_fps:g} FPS, "
                      f"{'원본 1:1 기록' if is_file else '실제 시간 동기화'})")
            if writer:
                if not is_file:
                    write_until(t_cap)  # 이 프레임이 찍힌 시각까지 빈 구간 채우기
                writer.write(view)
                written += 1
                prev_view = view

            n += 1
            if n % 30 == 0:
                print(f"평균 {n / (time.perf_counter() - t0):.1f} FPS")
            if not args.no_show:
                cv2.imshow(WINDOW, view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # FPS 상한: 추론이 더 느리면 그대로 두고, 빠르면 남는 시간만큼 쉰다
            next_t += frame_interval
            wait = next_t - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                next_t = time.perf_counter()
    except KeyboardInterrupt:  # Ctrl+C로 정상 종료 (요약을 출력하고 파일을 닫는다)
        pass
    finally:
        t_end = time.perf_counter()
        cap.release()
        if writer:
            if not is_file:
                write_until(t_end)  # 마지막 프레임 이후 ~ 종료 시점까지 채우기
            writer.release()
        cv2.destroyAllWindows()

        elapsed = t_end - t0
        print("\n===== 실행 요약 =====")
        print(f"실제 경과 시간  : {elapsed:.1f} 초")
        print(f"AI 처리 프레임  : {n} 장 (평균 {n / elapsed if elapsed > 0 else 0:.1f} FPS)")
        if writer:
            print(f"파일 기록 프레임: {written} 장 @ {record_fps:g} FPS "
                  f"-> 재생 길이 {written / record_fps:.1f} 초")
            if not is_file:
                print(f"  (그중 시간 맞추기용 반복 프레임: {duplicated} 장)")


if __name__ == "__main__":
    main()
