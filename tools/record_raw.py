"""AI 없이 카메라 원본만 녹화한다.

목적
  1) 기준선: AI 없이 카메라+저장만 해도 프레임을 건너뛰는지 확인한다.
     여기서도 건너뛰면 원인은 모델/CPU 추론이 아니라 카메라 설정이나 저장(인코딩) 속도다.
  2) tools/offline_eval.py 에 넣을 테스트 영상을 만든다. 같은 영상으로 여러 설정을 공정하게 비교할 수 있다.

주의: 사람이 찍힌 원본이 그대로 저장된다. 팀 내부 테스트용으로만 쓰고 외부에 공유/업로드하지 말 것.
      runs/ 폴더와 *.mp4, *.csv 는 git에 올라가지 않는다.

사용법 (저장소 최상위 폴더에서)
    python tools/record_raw.py --seconds 30 --fps 15 --out runs/raw.mp4
"""
import argparse
import csv
import signal
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from camera import open_source  # noqa: E402
from metrics import sensor_drops, stats  # noqa: E402


def _stop(*_):
    raise KeyboardInterrupt


def main() -> None:
    # 백그라운드 실행에서 kill(SIGTERM)로 끝내도 영상 파일을 정상적으로 닫는다
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _stop)
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="picamera")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15, help="카메라 FrameRate 및 영상 파일 FPS")
    p.add_argument("--seconds", type=float, default=30, help="녹화 시간 (Ctrl+C로 먼저 끝낼 수 있음)")
    p.add_argument("--out", default="runs/raw.mp4")
    p.add_argument("--low-latency", action="store_true")
    p.add_argument("--no-af", action="store_true", help="연속 자동초점 끄기")
    p.add_argument("--stream", type=int, default=0, metavar="PORT",
                   help="녹화 중인 원본을 브라우저로 보기 (예: 8000). 화면 구도 확인용")
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    log_path = str(Path(args.out).with_suffix(".csv"))
    cap = open_source(args.source, args.width, args.height, args.fps, args.low_latency,
                      autofocus=not args.no_af)
    if not cap.isOpened():
        raise SystemExit(f"소스를 열 수 없음: {args.source}")
    streamer = None
    if args.stream:
        from stream import MJPEGStreamer
        streamer = MJPEGStreamer(args.stream)
        print(f"녹화 화면 확인: http://dontlookme.local:{args.stream} (원본이 전송됨, 개발용)")

    writer = None
    ts, fds, write_ms = [], [], []
    n, t0 = 0, time.perf_counter()
    print(f"원본 녹화 중... {args.seconds:g}초 (Ctrl+C로 중단)")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "sensor_ts_ns", "frame_dur_us", "exposure_us", "write_ms"])
        try:
            while time.perf_counter() - t0 < args.seconds:
                ok, frame = cap.read()
                t = time.perf_counter()
                if not ok:
                    print("[오류] 프레임 읽기 실패")
                    continue
                if writer is None:
                    h, wd = frame.shape[:2]
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (wd, h))
                    if not writer.isOpened():
                        raise SystemExit(f"영상 파일을 열 수 없음: {args.out}")
                writer.write(frame)
                wm = (time.perf_counter() - t) * 1e3
                if streamer:
                    streamer.update(frame)
                meta = getattr(cap, "last_meta", {})
                n += 1
                ts.append(meta.get("SensorTimestamp"))
                fds.append(meta.get("FrameDuration"))
                write_ms.append(wm)
                w.writerow([n, round(t - t0, 4), meta.get("SensorTimestamp"), meta.get("FrameDuration"),
                            meta.get("ExposureTime"), round(wm, 2)])
        except KeyboardInterrupt:
            pass
        finally:
            elapsed = time.perf_counter() - t0
            cap.release()
            if writer:
                writer.release()
            if streamer:
                streamer.close()

    print("\n===== 원본 녹화 요약 =====")
    print(f"경과 {elapsed:.1f}초, 받은 프레임 {n}장 ({n / elapsed if elapsed else 0:.1f} FPS)")
    ws = stats(write_ms)
    if ws["n"]:
        print(f"프레임 저장 시간: 평균 {ws['mean']:.1f}ms, p95 {ws['p95']:.1f}ms")
    fd = stats([v / 1000 for v in fds if v])["p50"]
    if fd and sum(x is not None for x in ts) >= 2:
        d = sum(sensor_drops(ts, fd))
        print(f"카메라 간격 {fd:.1f}ms ({1000 / fd:.1f} FPS), 건너뛴 카메라 프레임 {d}장 ({d / (n + d):.1%})")
        if d:
            print("-> AI 없이도 건너뜀: 카메라 설정이나 저장(인코딩) 속도 문제. 이 영상은 실제보다 짧게 재생됨.")
        else:
            print("-> 카메라+저장만으로는 프레임 손실 없음 (기준선 OK).")
    else:
        print("카메라 센서 타임스탬프 없음 (picamera가 아닌 소스)")
    print(f"영상: {args.out}\n기록: {log_path}")
    print(f"다음: python tools/offline_eval.py {args.out}")


if __name__ == "__main__":
    main()
