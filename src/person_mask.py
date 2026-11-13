"""YOLOv8-seg로 사람 영역을 찾아 불투명 단색으로 가린다. (PC / 라즈베리파이 5 공용)

- 사람 클래스(COCO id 0)만 사용한다. 개/고양이 등은 가리지 않는다.
- 마스크는 반드시 같은 프레임의 추론 결과만 쓴다 (이전 프레임 마스크 재사용 금지).
- 마스크 경계를 dilate로 넓혀서 윤곽 밖으로 삐져나오는 몸을 줄인다.
- 추론/카메라 오류가 나면 원본을 내보내지 않고 검은 화면을 낸다 (fail-safe).

사용법 (저장소 최상위 폴더에서)
    python src/person_mask.py --source 0                     # USB 웹캠 (왼쪽 원본 | 오른쪽 마스크 분할 화면)
    python src/person_mask.py --source picamera              # 라즈베리파이 CSI 카메라 (Picamera2), 분할 화면
    python src/person_mask.py --source video.mp4
    python src/person_mask.py --source image.jpg --save out.jpg
    python src/person_mask.py --source picamera --single --fullscreen --fps 15 --imgsz 320   # 라즈베리파이 출력용 (마스크만)
    python src/person_mask.py --source 0 --no-show           # 화면 없이 FPS만 측정

프레임 진단 로그 (프레임마다 처리 시간/검출 결과/카메라 타임스탬프/CPU 상태를 CSV로 기록)
    python src/person_mask.py --source picamera --imgsz 320 --no-show --log runs/live.csv
    python tools/analyze_log.py runs/live.csv

라즈베리파이에서 빠르게 돌리려면 NCNN으로 변환해서 쓴다 (한 번만 실행).
    python src/person_mask.py --export ncnn --imgsz 320
    python src/person_mask.py --source picamera --weights weights/yolov8n-seg_ncnn_model --imgsz 320
"""
import argparse
import csv
import signal
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from camera import now_ns, open_source
from sysmon import SysMonitor

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "yolov8n-seg.pt"
PERSON_CLASS = 0
WINDOW = "person_mask"

# --log CSV 열 (단위: *_ms 밀리초, *_s 초)
LOG_FIELDS = [
    "frame",          # 처리한 프레임 번호
    "t_s",            # 프로그램 시작 후 프레임을 받은 시각
    "status",         # ok / read_fail / infer_error
    "sensor_ts_ns",   # 카메라 센서 촬영 시각 (Picamera2만)
    "frame_dur_us",   # 카메라 프레임 간격 (Picamera2만)
    "exposure_us",    # 노출 시간 (Picamera2만)
    "read_ms",        # 카메라에서 프레임을 기다린 시간
    "capture_age_ms", # 받았을 때 이미 지난 시간 = 받은 시각 - 촬영 시각 (크면 오래된 프레임이 큐에 쌓여 있었다는 뜻)
    "pre_ms", "inf_ms", "post_ms",  # ultralytics 내부 측정: 전처리 / 모델 추론 / 후처리
    "predict_ms",     # model.predict() 전체 (위 셋 + 부가 시간)
    "mask_ms",        # 마스크 합치기/확장/칠하기
    "out_ms",         # 분할 화면 만들기 + 파일 저장 + 화면 출력
    "work_ms",        # 프레임 하나 처리에 실제로 쓴 시간 (read 대기 제외)
    "sleep_ms",       # --fps 상한 때문에 쉰 시간
    "latency_ms",     # 촬영 시각 -> 출력 완료까지 (Picamera2만)
    "n_person", "max_conf", "min_conf",
    "mask_frac",      # 화면에서 마스크가 차지하는 비율 (0~1)
    "motion_frac",    # 직전 프레임 대비 움직인 픽셀 비율 (사람 없음 + 움직임 큼 = 미검출 의심)
    "cpu_pct", "cpu_max_core", "cpu_temp", "cpu_mhz", "throttled",
]


def person_union_mask(shape, result, dilate_px: int):
    """사람 폴리곤을 모두 합친 0/255 마스크를 만든다. 사람이 없으면 None."""
    if result.masks is None or len(result.masks) == 0:
        return None
    h, w = shape[:2]
    union = np.zeros((h, w), dtype=np.uint8)
    # masks.xy: 원본 해상도 좌표의 폴리곤 -> 후처리(리사이즈)가 필요 없어 가볍다
    for poly in result.masks.xy:
        if len(poly) >= 3:
            cv2.fillPoly(union, [poly.astype(np.int32)], 255)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        union = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return union


def mask_people(frame: np.ndarray, result, color: tuple, dilate_px: int):
    """frame에서 사람 영역을 color로 채운 새 프레임과, 사용한 마스크(없으면 None)를 반환한다."""
    out = frame.copy()
    union = person_union_mask(frame.shape, result, dilate_px)
    if union is not None:
        out[union > 0] = color
    return out, union


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
    p.add_argument("--log", default=None,
                   help="프레임별 진단 기록 CSV 경로 (예: runs/live.csv). tools/analyze_log.py로 분석")
    p.add_argument("--low-latency", action="store_true",
                   help="Picamera2 버퍼/큐를 줄여 오래된 프레임이 쌓이지 않게 한다 (지연 비교 실험용)")
    p.add_argument("--no-af", action="store_true", help="Camera Module 3 연속 자동초점 끄기")
    p.add_argument("--stream", type=int, default=0, metavar="PORT",
                   help="브라우저 실시간 화면 포트 (예: 8000 -> http://<파이주소>:8000). 0이면 끔")
    p.add_argument("--export", choices=["ncnn", "onnx"], help="모델 변환 후 종료")
    return p.parse_args()


def _stop(*_):
    raise KeyboardInterrupt


def main() -> None:
    # Ctrl+C(SIGINT)뿐 아니라 kill(SIGTERM)로 끝내도 요약 출력/파일 정리를 하도록 한다.
    # (백그라운드 실행은 SIGINT가 무시되므로 'pkill -f person_mask.py'로 끝낸다)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _stop)
    args = parse_args()
    if args.export:
        export_model(args)
        return

    color = tuple(int(c) for c in args.color.split(","))
    model = YOLO(args.weights, task="segment")
    # 첫 추론은 모델 준비 때문에 수백 ms 걸린다. 카메라 시작 전에 미리 한 번 돌려 둔다.
    try:
        model.predict(np.zeros((args.height, args.width, 3), np.uint8), imgsz=args.imgsz,
                      device=args.device, verbose=False)
    except Exception as e:
        print(f"[경고] 모델 예열 실패: {e}")

    is_image = Path(args.source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

    def process(frame: np.ndarray):
        """(마스킹된 프레임, 진단 정보 dict)를 반환한다."""
        info = {"status": "ok"}
        try:
            t0 = time.perf_counter()
            res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, classes=[PERSON_CLASS],
                                device=args.device, verbose=False)[0]
            t1 = time.perf_counter()
            out, union = mask_people(frame, res, color, args.dilate)
            t2 = time.perf_counter()
            speed = getattr(res, "speed", None) or {}
            confs = res.boxes.conf.tolist() if res.boxes is not None else []
            info.update(
                pre_ms=speed.get("preprocess"), inf_ms=speed.get("inference"),
                post_ms=speed.get("postprocess"),
                predict_ms=(t1 - t0) * 1e3, mask_ms=(t2 - t1) * 1e3,
                n_person=len(confs),
                max_conf=round(max(confs), 3) if confs else 0.0,
                min_conf=round(min(confs), 3) if confs else None,
                mask_frac=round(np.count_nonzero(union) / union.size, 4) if union is not None else 0.0,
            )
            return out, info
        except Exception as e:  # 실패 시 원본 노출 금지
            print(f"[오류] 추론 실패, 검은 화면 출력: {e}")
            info["status"] = "infer_error"
            return np.zeros_like(frame), info

    if is_image:
        frame = cv2.imread(args.source)
        if frame is None:
            raise SystemExit(f"이미지를 읽을 수 없음: {args.source}")
        out, _ = process(frame)
        view = out if args.single else side_by_side(frame, out, args.view_width)
        if args.save:
            cv2.imwrite(args.save, view)
            print(f"저장: {args.save}")
        if not args.no_show:
            cv2.imshow(WINDOW, view)
            cv2.waitKey(0)
        return

    cap = open_source(args.source, args.width, args.height, args.fps, args.low_latency,
                      autofocus=not args.no_af)
    if not cap.isOpened():
        raise SystemExit(f"소스를 열 수 없음: {args.source}")

    streamer = None
    if args.stream:
        from stream import MJPEGStreamer
        streamer = MJPEGStreamer(args.stream)
        print(f"실시간 화면: 브라우저에서 http://<이 기기 주소>:{args.stream} "
              f"(예: http://dontlookme.local:{args.stream})"
              + ("" if args.single else "  * 분할 화면이라 원본도 전송됨 (개발용)"))

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

    # ---- 진단 로그 설정 ----
    log_file = log_writer = sysmon = None
    prev_small = None
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.log, "w", newline="", encoding="utf-8")
        log_writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        log_writer.writeheader()
        sysmon = SysMonitor(interval=1.0)
        print(f"진단 로그 기록: {args.log}")

    n, t0, next_t = 0, time.perf_counter(), time.perf_counter()
    try:
        while True:
            t_read0 = time.perf_counter()
            ok, frame = cap.read()
            t_cap = time.perf_counter()  # 이 프레임이 실제로 들어온 시각
            t_cap_ns = now_ns()
            meta = getattr(cap, "last_meta", {}) if ok else {}
            if not ok:
                if is_file:  # 영상 파일 끝
                    break
                # 카메라 프레임을 못 받으면 원본이 없으므로 검은 화면 (fail-safe)
                print("[오류] 프레임 읽기 실패, 검은 화면 출력")
                frame = np.zeros(last_shape, dtype=np.uint8)
                out, info = frame, {"status": "read_fail"}
            else:
                last_shape = frame.shape
                out, info = process(frame)

            t_out0 = time.perf_counter()
            view = out if args.single else side_by_side(frame, out, args.view_width)
            if not args.single:
                # 개발용 분할 화면에만: 진단 로그(t_s, frame)와 녹화 영상을 맞춰 보기 위한 표시
                label = f"t={t_cap - t0:6.1f}s  #{n + 1}  persons={info.get('n_person', '-')}"
                cv2.putText(view, label, (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(view, label, (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 1, cv2.LINE_AA)

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

            if streamer:
                streamer.update(view)

            quit_requested = False
            if not args.no_show:
                cv2.imshow(WINDOW, view)
                quit_requested = (cv2.waitKey(1) & 0xFF) == ord("q")
            t_out1 = time.perf_counter()
            t_out_ns = now_ns()

            n += 1
            if n % 30 == 0:
                print(f"평균 {n / (time.perf_counter() - t0):.1f} FPS")

            # FPS 상한: 추론이 더 느리면 그대로 두고, 빠르면 남는 시간만큼 쉰다
            sleep_s = 0.0
            if not quit_requested:
                next_t += frame_interval
                wait = next_t - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                    sleep_s = wait
                else:
                    next_t = time.perf_counter()

            if log_writer:
                motion = None
                if ok:  # 직전 프레임 대비 움직임 (작게 줄여 계산해서 부담이 거의 없음)
                    small = cv2.cvtColor(cv2.resize(frame, (80, 60), interpolation=cv2.INTER_AREA),
                                         cv2.COLOR_BGR2GRAY)
                    if prev_small is not None:
                        motion = round(float((cv2.absdiff(small, prev_small) > 25).mean()), 4)
                    prev_small = small
                sensor_ts = meta.get("SensorTimestamp")
                row = {
                    "frame": n,
                    "t_s": round(t_cap - t0, 4),
                    "sensor_ts_ns": sensor_ts,
                    "frame_dur_us": meta.get("FrameDuration"),
                    "exposure_us": meta.get("ExposureTime"),
                    "read_ms": round((t_cap - t_read0) * 1e3, 2),
                    "capture_age_ms": round((t_cap_ns - sensor_ts) / 1e6, 2) if sensor_ts else None,
                    "out_ms": round((t_out1 - t_out0) * 1e3, 2),
                    "work_ms": round((t_out1 - t_cap) * 1e3, 2),
                    "sleep_ms": round(sleep_s * 1e3, 2),
                    "latency_ms": round((t_out_ns - sensor_ts) / 1e6, 2) if sensor_ts else None,
                    "motion_frac": motion,
                }
                for k, v in info.items():
                    row[k] = round(v, 2) if isinstance(v, float) and k.endswith("_ms") else v
                row.update(sysmon.sample())
                log_writer.writerow(row)

            if quit_requested:
                break
    except KeyboardInterrupt:  # Ctrl+C로 정상 종료 (요약을 출력하고 파일을 닫는다)
        pass
    finally:
        t_end = time.perf_counter()
        cap.release()
        if writer:
            if not is_file:
                write_until(t_end)  # 마지막 프레임 이후 ~ 종료 시점까지 채우기
            writer.release()
        if log_file:
            log_file.close()
        if streamer:
            streamer.close()
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
        if args.log:
            print(f"진단 로그: {args.log}\n  분석: python tools/analyze_log.py {args.log}")


if __name__ == "__main__":
    main()
