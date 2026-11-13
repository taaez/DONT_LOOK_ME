"""같은 원본 영상을 서로 다른 방식으로 처리한 '출력 영상'들을 비교해 실제로 사람이 얼마나 노출됐는지 잰다.

방식 내부 로그가 아니라 실제로 나간 출력 화면을 기준으로 판단한다.
  - 정답 대용: 큰 기준 모델(--reference)이 원본에서 찾은 사람 영역
  - 가려짐  : 출력 픽셀이 가림 색(40,40,40) 또는 검은 보호 화면에 가까우면 가려진 것으로 본다
              (원본 자체가 그 색에 가까운 픽셀은 노출돼도 알아볼 수 없으므로 같이 '가려짐'으로 취급)
  - 노출 프레임: 기준 모델의 사람 픽셀 중 가려지지 않은 비율 > --leak

사용법 (출력 영상은 원본과 프레임 수·순서가 같아야 한다)
    python tools/compare_methods.py runs/compare_raw.mp4 \
        --method main=runs/cmp_main.mp4 --method seok=runs/cmp_seok.mp4:crop_top=150 \
        --reference yolov8m-seg.pt --dump 12 --out runs/compare
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from metrics import fmt, stats  # noqa: E402
from person_mask import PERSON_CLASS, person_union_mask  # noqa: E402
from ultralytics import YOLO  # noqa: E402

FILL = np.array([40, 40, 40], np.int16)


def parse_method(spec: str):
    """'이름=경로[:crop_top=N]'"""
    name, rest = spec.split("=", 1)
    path, crop = rest, 0
    if ":crop_top=" in rest:
        path, c = rest.split(":crop_top=")
        crop = int(c)
    return name, path, crop


def covered_mask(out: np.ndarray, tol: int) -> np.ndarray:
    """출력에서 가림 색(40,40,40) 또는 검정에 가까운 픽셀."""
    o = out.astype(np.int16)
    near_fill = (np.abs(o - FILL).max(axis=2) <= tol)
    near_black = (o.max(axis=2) <= tol)
    return near_fill | near_black


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw", help="모든 방식에 넣은 원본 영상")
    p.add_argument("--method", action="append", required=True, help="이름=출력영상[:crop_top=N] (여러 번)")
    p.add_argument("--reference", default="yolov8m-seg.pt", help="정답 대용 기준 모델")
    p.add_argument("--ref-imgsz", type=int, default=640)
    p.add_argument("--ref-conf", type=float, default=0.25)
    p.add_argument("--leak", type=float, default=0.05, help="가려지지 않은 사람 픽셀 비율이 이보다 크면 노출")
    p.add_argument("--min-px", type=int, default=400, help="기준 모델 사람 면적이 이보다 작으면 판정 제외")
    p.add_argument("--tol", type=int, default=22, help="가림 색 판정 허용 오차 (영상 압축 노이즈)")
    p.add_argument("--dump", type=int, default=0, help="방식별 노출 프레임 이미지를 최대 몇 장 저장")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="runs/compare")
    args = p.parse_args()

    methods = [parse_method(s) for s in args.method]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = YOLO(args.reference, task="segment")
    raw = cv2.VideoCapture(args.raw)
    caps = {name: cv2.VideoCapture(path) for name, path, _ in methods}
    crops = {name: crop for name, _, crop in methods}
    total = int(raw.get(cv2.CAP_PROP_FRAME_COUNT))
    for name, path, _ in methods:
        n = int(caps[name].get(cv2.CAP_PROP_FRAME_COUNT))
        if n != total:
            print(f"[경고] {name}: 출력 {n}프레임, 원본 {total}프레임 -> 앞에서부터 짝지어 비교")

    acc = {name: {"leak": [], "exposed": [], "black_person": 0, "black_empty": 0, "black": 0}
           for name, _, _ in methods}
    n_person = n_empty = 0
    dumped = {name: 0 for name, _, _ in methods}
    rows = []
    idx = 0
    while True:
        ok, frame = raw.read()
        outs = {}
        for name in caps:
            ok2, o = caps[name].read()
            if ok2:
                outs[name] = o[crops[name]:] if crops[name] else o
        if not ok or len(outs) != len(caps):
            break
        idx += 1
        r = ref.predict(frame, imgsz=args.ref_imgsz, conf=args.ref_conf, classes=[PERSON_CLASS],
                        device=args.device, verbose=False)[0]
        ref_mask = person_union_mask(frame.shape, r, 0)
        ref_px = int(np.count_nonzero(ref_mask)) if ref_mask is not None else 0
        has_person = ref_px >= args.min_px
        n_person += has_person
        n_empty += not has_person
        row = {"frame": idx, "ref_px": ref_px}
        # 원본 자체가 가림색/검정에 가까운 픽셀은 노출돼도 알아볼 수 없음
        raw_cov = covered_mask(frame, args.tol)
        for name, o in outs.items():
            if o.shape != frame.shape:
                o = cv2.resize(o, (frame.shape[1], frame.shape[0]))
            cov = covered_mask(o, args.tol) | raw_cov
            black = (o.max(axis=2) <= args.tol).mean() > 0.97  # 전체 보호 화면
            a = acc[name]
            a["black"] += black
            if has_person:
                a["black_person"] += black
                leak = (ref_mask > 0) & ~cov
                ratio = np.count_nonzero(leak) / ref_px
                exposed = ratio > args.leak
                a["leak"].append(ratio)
                a["exposed"].append(idx if exposed else None)
                row[f"{name}_leak"] = round(ratio, 4)
                row[f"{name}_exposed"] = int(exposed)
                if exposed and dumped[name] < args.dump:
                    vis = np.hstack([frame, o])
                    red = vis[:, :frame.shape[1]]
                    red[leak] = (0.4 * red[leak] + [0, 0, 150]).astype(np.uint8)
                    cv2.putText(vis, f"{name} #{idx} t={(idx - 1) / 15:.1f}s leak={ratio:.0%}", (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imwrite(str(out_dir / f"{name}_exposed_f{idx:05d}.jpg"), vis)
                    dumped[name] += 1
            else:
                a["black_empty"] += black
            row[f"{name}_black"] = int(black)
        rows.append(row)
        if idx % 100 == 0:
            print(f"  {idx}/{total}")

    with open(out_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "frame", k))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"\n===== 출력 영상 비교 ({idx}프레임, 기준 모델 {args.reference}) =====")
    print(f"사람 있는 프레임 {n_person}장 / 사람 없는 프레임 {n_empty}장\n")
    hdr = f"{'방식':<8}{'노출 프레임':>14}{'노출률 평균':>12}{'노출률 p95':>12}{'검은화면(사람有)':>16}{'검은화면(사람無)':>16}"
    print(hdr)
    summary = []
    for name, _, _ in methods:
        a = acc[name]
        ex = [i for i in a["exposed"] if i is not None]
        ls = stats(a["leak"])
        pct = 100 * len(ex) / n_person if n_person else 0
        print(f"{name:<8}{len(ex):>7}장 ({pct:4.1f}%){fmt(100 * ls['mean'] if ls['mean'] is not None else None, 2):>11}%"
              f"{fmt(100 * ls['p95'] if ls['p95'] is not None else None, 2):>11}%"
              f"{a['black_person']:>14}장{a['black_empty']:>15}장")
        summary.append({"method": name, "exposed_frames": len(ex), "exposed_pct": round(pct, 2),
                        "leak_mean_pct": round(100 * ls["mean"], 2) if ls["mean"] is not None else None,
                        "leak_p95_pct": round(100 * ls["p95"], 2) if ls["p95"] is not None else None,
                        "black_with_person": a["black_person"], "black_empty_room": a["black_empty"],
                        "exposed_frame_ids": " ".join(map(str, ex))})
        if ex:
            # 연속 구간으로 묶어 시각 표시 (원본 15FPS 기준)
            segs, s = [], ex[0]
            for x, y in zip(ex, ex[1:] + [None]):
                if y != x + 1:
                    segs.append((s, x))
                    s = y
            print("        노출 구간: " + ", ".join(f"{(a_ - 1) / 15:.1f}~{(b_ - 1) / 15:.1f}s({b_ - a_ + 1}장)"
                                               for a_, b_ in segs[:12]) + (" ..." if len(segs) > 12 else ""))
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\n결과: {out_dir}/summary.csv, frames.csv"
          + (f", 노출 프레임 이미지 {sum(dumped.values())}장 (왼쪽 원본의 빨간 부분 = 노출)" if args.dump else ""))
    print("참고: '검은화면(사람無)'은 빈 방인데 검은 화면이 나간 프레임 = 노출은 아니지만 홈캠 사용성 손실")


if __name__ == "__main__":
    main()
