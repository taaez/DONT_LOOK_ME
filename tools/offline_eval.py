"""녹화된 원본 영상의 '모든 프레임'을 시간 제약 없이 처리해서 모델 자체의 성능을 잰다.

실시간이 아니므로 프레임을 하나도 건너뛰지 않는다. 그래서 여기서 나오는 미검출은 CPU나 간격 문제가 아니라
순수하게 '모델/설정' 문제다. 실시간 결과(analyze_log.py)와 비교하면 두 원인을 분리할 수 있다.

기준 모델(--reference, 예: yolov8x-seg.pt)을 주면 그 결과를 '정답 대용'으로 삼아 아래를 센다.
  - 노출 프레임 : 기준 모델이 찾은 사람 픽셀 중, 우리 마스크(dilate 포함)가 못 가린 비율이 --leak 보다 큰 프레임
  - 완전 미검출 : 기준 모델은 사람을 찾았는데 우리 모델은 0명인 프레임
기준 모델도 완벽하지 않다. --dump 로 저장한 노출 프레임 이미지를 눈으로 꼭 확인할 것.
(빨간색 = 기준 모델 기준으로 가려지지 않은 사람 영역)

기준 모델은 PC(노트북)에서 돌리는 것을 권장한다 (라즈베리파이에서는 매우 느림).
처음 실행하면 ultralytics가 가중치를 자동으로 내려받는다 (수십~백수십 MB, 인터넷 필요).
추론 시간(ms)은 '이 스크립트를 돌린 컴퓨터' 기준이다. 라즈베리파이 속도는 라즈베리파이에서 잰 값만 믿을 것.

사용법 (저장소 최상위 폴더에서)
    python tools/offline_eval.py runs/raw.mp4 --imgsz 320,416,640
    python tools/offline_eval.py runs/raw.mp4 --imgsz 320,640 --conf 0.15,0.25 --reference yolov8x-seg.pt --dump 20
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from metrics import find_short_gaps, fmt, stats  # noqa: E402
from person_mask import DEFAULT_WEIGHTS, PERSON_CLASS, person_union_mask  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def parse_list(s: str, typ) -> list:
    return [typ(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="쉼표로 여러 개 가능 (예: .pt, ncnn 폴더)")
    p.add_argument("--imgsz", default="320", help="쉼표로 여러 개 (예: 320,416,640)")
    p.add_argument("--conf", default="0.25", help="쉼표로 여러 개 (예: 0.15,0.25)")
    p.add_argument("--dilate", type=int, default=8)
    p.add_argument("--reference", default=None, help="기준 모델 (예: yolov8x-seg.pt). 없으면 '짧은 끊김'으로만 추정")
    p.add_argument("--ref-imgsz", type=int, default=640)
    p.add_argument("--ref-conf", type=float, default=0.25)
    p.add_argument("--leak", type=float, default=0.05, help="못 가린 사람 픽셀 비율이 이보다 크면 '노출' (기본 5%%)")
    p.add_argument("--min-px", type=int, default=400, help="기준 모델의 사람 면적이 이보다 작으면 판정 제외 (픽셀)")
    p.add_argument("--max-gap", type=int, default=3)
    p.add_argument("--dump", type=int, default=0, help="설정별로 노출 프레임 이미지를 최대 몇 장 저장할지")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="결과 폴더 (기본: runs/offline_<영상이름>)")
    args = p.parse_args()

    weights, imgszs, confs = parse_list(args.weights, str), parse_list(args.imgsz, int), parse_list(args.conf, float)
    configs = [(w, s, c) for w in weights for s in imgszs for c in confs]
    label = {cfg: f"{Path(cfg[0]).name}|imgsz={cfg[1]}|conf={cfg[2]}" for cfg in configs}
    models = {w: YOLO(w, task="segment") for w in weights}
    ref = YOLO(args.reference, task="segment") if args.reference else None

    out_dir = Path(args.out or f"runs/offline_{Path(args.video).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없음: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"영상 {args.video}: {total}프레임, 설정 {len(configs)}개" + (f", 기준 모델 {args.reference}" if ref else ""))

    acc = {cfg: {"ms": [], "has": [], "ref_has": [], "leak": [], "exposed": 0, "full_miss": 0} for cfg in configs}
    dumped = {cfg: 0 for cfg in configs}
    fields = ["frame", "config", "n_person", "max_conf", "infer_ms", "mask_frac",
              "ref_n", "ref_px", "leak_px", "leak_ratio", "exposed", "full_miss"]
    f = open(out_dir / "frames.csv", "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()

    def predict(model, frame, imgsz, conf):
        t = time.perf_counter()
        res = model.predict(frame, imgsz=imgsz, conf=conf, classes=[PERSON_CLASS],
                            device=args.device, verbose=False)[0]
        return res, (time.perf_counter() - t) * 1e3

    idx, t_start = 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            ref_mask, ref_px, ref_n = None, 0, None
            if ref:
                rres, _ = predict(ref, frame, args.ref_imgsz, args.ref_conf)
                ref_mask = person_union_mask(frame.shape, rres, 0)  # 기준은 확장 없이 실제 사람 영역
                ref_px = int(np.count_nonzero(ref_mask)) if ref_mask is not None else 0
                ref_n = len(rres.boxes) if rres.boxes is not None else 0

            for cfg in configs:
                wt, sz, cf = cfg
                res, ms = predict(models[wt], frame, sz, cf)
                mask = person_union_mask(frame.shape, res, args.dilate)
                cs = res.boxes.conf.tolist() if res.boxes is not None else []
                a = acc[cfg]
                if idx > 1:  # 첫 프레임은 모델 준비(warm-up) 시간이 섞여서 제외
                    a["ms"].append(ms)
                a["has"].append(len(cs) > 0)
                row = {"frame": idx, "config": label[cfg], "n_person": len(cs),
                       "max_conf": round(max(cs), 3) if cs else 0, "infer_ms": round(ms, 1),
                       "mask_frac": round(np.count_nonzero(mask) / mask.size, 4) if mask is not None else 0}
                if ref:
                    a["ref_has"].append(ref_px >= args.min_px)
                    row.update(ref_n=ref_n, ref_px=ref_px)
                    if ref_px >= args.min_px:
                        covered = mask > 0 if mask is not None else np.zeros(frame.shape[:2], bool)
                        leak = (ref_mask > 0) & ~covered
                        leak_px = int(np.count_nonzero(leak))
                        ratio = leak_px / ref_px
                        exposed, full_miss = ratio > args.leak, len(cs) == 0
                        a["leak"].append(ratio)
                        a["exposed"] += exposed
                        a["full_miss"] += full_miss
                        row.update(leak_px=leak_px, leak_ratio=round(ratio, 4),
                                   exposed=int(exposed), full_miss=int(full_miss))
                        if exposed and dumped[cfg] < args.dump:
                            vis = frame.copy()
                            vis[covered] = (40, 40, 40)          # 실제 출력처럼 가린 영역
                            vis[leak] = (0.5 * vis[leak] + [0, 0, 127]).astype(np.uint8)  # 못 가린 영역 빨갛게
                            cv2.putText(vis, f"{label[cfg]} leak={ratio:.0%}", (8, 22),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                            name = f"exposed_f{idx:05d}_{sz}_{cf}_{Path(wt).stem}.jpg"
                            cv2.imwrite(str(out_dir / name), vis)
                            dumped[cfg] += 1
                w.writerow(row)

            if idx % 50 == 0:
                el = time.perf_counter() - t_start
                print(f"  {idx}/{total or '?'} 프레임 ({el:.0f}초 경과)")
    except KeyboardInterrupt:
        print("중단됨 - 지금까지 처리한 프레임으로 요약합니다.")
    finally:
        f.close()
        cap.release()

    # ---------------------------------------------------------------- 요약
    print(f"\n===== 오프라인 평가 요약 ({idx}프레임, 모든 프레임 처리 = 간격/CPU 영향 없음) =====")
    summary = []
    for cfg in configs:
        a = acc[cfg]
        ms = stats(a["ms"])
        gaps = find_short_gaps(a["has"], args.max_gap)
        gap_n = sum(b - s + 1 for s, b in gaps)
        s = {"config": label[cfg], "infer_ms_mean": fmt(ms["mean"]), "infer_ms_p95": fmt(ms["p95"]),
             "max_fps": fmt(1000 / ms["mean"] if ms["mean"] else None),
             "det_frames": sum(a["has"]), "short_gap_frames": gap_n}
        if ref:
            n_ref = sum(a["ref_has"])
            s.update(ref_person_frames=n_ref,
                     exposed_frames=a["exposed"], exposed_pct=fmt(100 * a["exposed"] / n_ref if n_ref else None, 2),
                     full_miss_frames=a["full_miss"],
                     full_miss_pct=fmt(100 * a["full_miss"] / n_ref if n_ref else None, 2),
                     leak_mean_pct=fmt(100 * stats(a["leak"])["mean"] if a["leak"] else None, 2),
                     leak_p95_pct=fmt(100 * stats(a["leak"])["p95"] if a["leak"] else None, 2))
        summary.append(s)

        print(f"\n[{label[cfg]}]")
        print(f"  추론 평균 {s['infer_ms_mean']}ms (p95 {s['infer_ms_p95']}ms) -> 이 컴퓨터에서 최대 약 {s['max_fps']} FPS")
        print(f"  사람 검출 프레임 {s['det_frames']}/{idx}, 짧게 끊긴 미검출 의심 {gap_n}장")
        if ref:
            print(f"  기준 모델 기준 사람 있는 프레임 {s['ref_person_frames']}장 중")
            print(f"    노출 프레임(못 가린 비율 > {args.leak:.0%}) : {s['exposed_frames']}장 ({s['exposed_pct']}%)  <- 목표 5% 미만, 최종 0")
            print(f"    완전 미검출                    : {s['full_miss_frames']}장 ({s['full_miss_pct']}%)")
            print(f"    못 가린 사람 픽셀 비율 평균 {s['leak_mean_pct']}%, p95 {s['leak_p95_pct']}%")

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as sf:
        sw = csv.DictWriter(sf, fieldnames=list(summary[0].keys()))
        sw.writeheader()
        sw.writerows(summary)
    print(f"\n결과: {out_dir}/summary.csv, {out_dir}/frames.csv"
          + (f", 노출 프레임 이미지 {sum(dumped.values())}장" if args.dump else ""))
    if not ref:
        print("참고: --reference 를 주면 '노출 프레임 비율'을 직접 잴 수 있습니다 (PC에서 권장).")


if __name__ == "__main__":
    main()
