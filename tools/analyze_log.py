"""person_mask.py --log 로 만든 CSV를 분석해 문제의 원인을 3갈래로 나눈다.

  [A] 간격  : 카메라가 찍었는데 처리하지 못하고 건너뛴 프레임. 왜 건너뛰었는지(처리가 느려서 / 타이밍이 어긋나서)도 나눈다.
  [B] CPU   : 프레임 하나 처리에 어느 단계가 얼마나 걸리는지, 발열·전원 때문에 느려졌는지.
  [C] 모델  : 처리는 했는데 사람을 놓친 것으로 '의심'되는 프레임. (정답 라벨이 없으므로 의심 수준,
              확정은 tools/offline_eval.py 의 기준 모델 비교로 한다)

프라이버시 관점: [A]에서 건너뛴 프레임은 화면에 나가지 않으므로 '노출'이 아니라 끊김/지연 문제다.
실제로 사람이 노출되는 건 [C]의 프레임이다.

사용법
    python tools/analyze_log.py runs/live.csv
    python tools/analyze_log.py runs/live.csv --plot runs/live.png    # 시간축 그래프 저장
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from metrics import find_short_gaps, fmt, sensor_drops, stats  # noqa: E402
from sysmon import decode_throttled  # noqa: E402


def num(v):
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k != "status":
                r[k] = num(v)
    return rows


def median(values):
    return stats(values)["p50"]


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--conf", type=float, default=0.25, help="실행 때 쓴 --conf (경계선 신뢰도 판정용)")
    p.add_argument("--target-fps", type=float, default=15,
                   help="카메라 타임스탬프가 없을 때(웹캠/파일) 쓸 목표 FPS")
    p.add_argument("--max-gap", type=int, default=3, help="이 프레임 수 이하로 끊긴 검출은 미검출 의심")
    p.add_argument("--motion", type=float, default=0.03, help="움직임 비율이 이보다 큰데 사람 0명이면 의심")
    p.add_argument("--plot", default=None, help="그래프 PNG 저장 경로")
    args = p.parse_args()

    rows = load(args.csv)
    if len(rows) < 2:
        raise SystemExit("기록된 프레임이 너무 적습니다.")
    col = lambda k, rs=rows: [r.get(k) for r in rs]  # noqa: E731
    findings = []  # (분류, 내용)

    # ------------------------------------------------------------ 개요
    section("개요")
    dur = rows[-1]["t_s"] - rows[0]["t_s"]
    proc_fps = (len(rows) - 1) / dur if dur > 0 else 0
    status = Counter(r["status"] for r in rows)
    print(f"기록 시간 {dur:.1f}초, 처리 프레임 {len(rows)}장, 처리 FPS {proc_fps:.1f}")
    print("상태: " + ", ".join(f"{k} {v}" for k, v in status.items()))
    if status.get("infer_error") or status.get("read_fail"):
        findings.append(("오류", f"추론 오류 {status.get('infer_error', 0)}장 / 카메라 읽기 실패 "
                                 f"{status.get('read_fail', 0)}장 (검은 화면으로 대체됨 - 노출은 아니지만 원인 확인 필요)"))

    # ------------------------------------------------------------ [A] 간격
    section("[A] 간격: 카메라 프레임을 건너뛰었나")
    ts = col("sensor_ts_ns")
    fd_ms = median([v / 1000 for v in col("frame_dur_us") if v])
    drops = None
    if fd_ms and sum(t is not None for t in ts) >= 2:
        drops = sensor_drops(ts, fd_ms)
        total_drop = sum(drops)
        cam_fps = 1000 / fd_ms
        rate = total_drop / (len(rows) + total_drop)
        # 건너뛴 원인 나누기: 직전 프레임 처리(work)가 카메라 간격보다 길었나?
        slow, timing = 0, 0
        for i, d in enumerate(drops):
            if d and i > 0:
                if (rows[i - 1]["work_ms"] or 0) > fd_ms:
                    slow += d
                else:
                    timing += d
        print(f"카메라 {cam_fps:.1f} FPS (간격 {fd_ms:.1f}ms) -> 처리 {proc_fps:.1f} FPS")
        print(f"건너뛴 카메라 프레임: {total_drop}장 / 카메라가 찍은 {len(rows) + total_drop}장 = {rate:.1%}")
        print(f"  - 처리가 느려서 (직전 처리시간 > 카메라 간격): {slow}장")
        print(f"  - 처리는 빨랐는데 타이밍이 어긋나서          : {timing}장")
        if rate > 0.05:
            if slow >= timing:
                findings.append(("A 간격", f"카메라 프레임의 {rate:.0%}를 건너뜀. 대부분 처리 시간이 카메라 간격"
                                         f"({fd_ms:.0f}ms)보다 길어서임 -> 원인은 [B] CPU/처리시간"))
            else:
                findings.append(("A 간격", f"카메라 프레임의 {rate:.0%}를 건너뜀. 처리는 빨랐는데 타이밍이 어긋남 -> "
                                         "--fps 상한과 카메라 FrameRate를 맞추거나 --fps 0으로 비교"))
        age = stats(col("capture_age_ms"))
        lat = stats(col("latency_ms"))
        print(f"받은 시점의 프레임 나이: 중앙 {fmt(age['p50'])}ms, p95 {fmt(age['p95'])}ms "
              f"(카메라 간격 {fd_ms:.0f}ms보다 많이 크면 큐에 오래된 프레임이 쌓인 것)")
        print(f"촬영 -> 출력 지연: 중앙 {fmt(lat['p50'])}ms, p95 {fmt(lat['p95'])}ms (목표 100ms 이하)")
        if age["p50"] is not None and age["p50"] > 1.5 * fd_ms:
            findings.append(("A 간격", f"카메라 큐에 오래된 프레임이 쌓임 (프레임 나이 중앙 {age['p50']:.0f}ms) "
                                     "-> --low-latency 로 다시 측정해 비교"))
        if lat["p50"] is not None and lat["p50"] > 100:
            findings.append(("A 지연", f"촬영->출력 지연 중앙 {lat['p50']:.0f}ms로 목표(100ms) 초과"))
        exp = median(col("exposure_us"))
        if exp:
            print(f"노출 시간 중앙 {exp / 1000:.1f}ms" + (" (길어서 움직이면 흐려짐 -> 미검출 원인 가능)"
                                                     if exp > 20000 else ""))
    else:
        print("카메라 센서 타임스탬프 없음 (웹캠/영상 파일) -> 건너뛴 프레임 계산 생략")
        fd_ms = 1000 / args.target_fps

    # ------------------------------------------------------------ [B] CPU
    section("[B] CPU / 처리 시간 (ms)")
    ok = [r for r in rows if r["status"] == "ok"]
    ok_t = ok[1:] if len(ok) > 2 else ok  # 첫 프레임은 시작 직후라 시간 통계에서 제외
    names =[("read_ms", "카메라 대기"), ("pre_ms", "전처리"), ("inf_ms", "모델 추론"), ("post_ms", "후처리"),
             ("predict_ms", "predict 전체"), ("mask_ms", "마스크 칠하기"), ("out_ms", "저장/출력"),
             ("work_ms", "처리 합계"), ("sleep_ms", "FPS 상한 대기")]
    print(f"{'단계':<14}{'평균':>8}{'중앙':>8}{'p95':>8}{'최대':>8}")
    st = {}
    for k, label in names:
        st[k] = stats(col(k, ok_t))
        s = st[k]
        print(f"{label:<14}{fmt(s['mean']):>8}{fmt(s['p50']):>8}{fmt(s['p95']):>8}{fmt(s['max']):>8}")

    work = st["work_ms"]["mean"] or 0
    if work > 0:
        parts = {
            "모델 추론(inference)": st["inf_ms"]["mean"] or 0,
            "전/후처리": (st["pre_ms"]["mean"] or 0) + (st["post_ms"]["mean"] or 0),
            "마스크 칠하기": st["mask_ms"]["mean"] or 0,
            "저장/출력": st["out_ms"]["mean"] or 0,
        }
        parts["기타"] = max(work - sum(parts.values()), 0)
        print("\n처리 시간 비중: " + ", ".join(f"{k} {v / work:.0%}" for k, v in parts.items()))
        over = sum((r["work_ms"] or 0) > fd_ms for r in ok_t) / max(len(ok_t), 1)
        print(f"처리 시간이 카메라 간격({fd_ms:.0f}ms)을 넘은 프레임: {over:.0%}")
        top = max(parts, key=parts.get)
        if over > 0.1:
            tip = {"모델 추론(inference)": "--imgsz 낮추기 / NCNN 변환 / AI HAT+(Hailo) 가속",
                   "전/후처리": "입력 해상도(--width/--height) 낮추기, NCNN 변환",
                   "마스크 칠하기": "--dilate 줄이기",
                   "저장/출력": "--save, 화면 출력을 끄고 다시 측정해 비교",
                   "기타": "다른 프로세스가 CPU를 쓰는지 확인 (htop)"}[top]
            findings.append(("B CPU", f"처리 평균 {work:.0f}ms로 카메라 간격 {fd_ms:.0f}ms보다 김 ({over:.0%} 프레임). "
                                     f"가장 큰 비중: {top} {parts[top] / work:.0%} -> {tip}"))
        inf = st["inf_ms"]
        if inf["p50"] and inf["p95"] and inf["p95"] > 1.5 * inf["p50"]:
            findings.append(("B CPU", f"추론 시간이 들쭉날쭉함 (중앙 {inf['p50']:.0f}ms, p95 {inf['p95']:.0f}ms) "
                                     "-> 발열/다른 프로세스/메모리 확인"))
        # 시간이 지나며 느려졌나 (발열)
        k = max(len(ok_t) // 5, 1)
        first, last = stats(col("inf_ms", ok_t[:k]))["mean"], stats(col("inf_ms", ok_t[-k:]))["mean"]
        if first and last:
            print(f"추론 시간 변화: 처음 20% 평균 {first:.0f}ms -> 마지막 20% 평균 {last:.0f}ms")
            if last > first * 1.2:
                findings.append(("B CPU", f"시간이 지날수록 추론이 느려짐 ({first:.0f} -> {last:.0f}ms) "
                                         "-> 발열 가능성, 쿨러/온도 확인"))

    temp, mhz = stats(col("cpu_temp")), stats(col("cpu_mhz"))
    cpu, core = stats(col("cpu_pct")), stats(col("cpu_max_core"))
    mhz_min = min((v for v in col("cpu_mhz") if v is not None), default=None)
    print(f"\nCPU 사용률 평균 {fmt(cpu['mean'])}% (가장 바쁜 코어 평균 {fmt(core['mean'])}%)")
    print(f"CPU 온도 최대 {fmt(temp['max'])}°C, 클럭 중앙 {fmt(mhz['p50'], 0)}MHz / 최저 {fmt(mhz_min, 0)}MHz"
          if mhz["n"] else "CPU 온도/클럭 정보 없음 (라즈베리파이가 아님)")
    thr = 0
    for v in col("throttled"):
        thr |= int(v) if v is not None else 0
    flags = decode_throttled(thr)
    print("전원/발열 제한: " + (", ".join(flags) if flags else "없음"))
    if flags:
        findings.append(("B CPU", "전원/발열 제한 감지: " + ", ".join(flags) + " -> 공식 27W 어댑터, 쿨러 확인"))
    if temp["max"] and temp["max"] >= 80:
        findings.append(("B CPU", f"CPU 온도 {temp['max']:.0f}°C (80°C 이상이면 성능 제한 시작)"))
    if cpu["mean"] is not None and core["mean"] is not None and cpu["mean"] < 50 and core["mean"] > 90:
        findings.append(("B CPU", "한 코어만 바쁘고 나머지는 놀고 있음 -> 단일 스레드 병목"))

    # ------------------------------------------------------------ [C] 모델
    section("[C] 모델: 사람을 놓친 것으로 의심되는 프레임")
    has = [(r["n_person"] or 0) > 0 for r in ok]
    n_det = sum(has)
    gaps = find_short_gaps(has, args.max_gap)
    gap_idx = {i for a, b in gaps for i in range(a, b + 1)}
    print(f"사람 검출 프레임 {n_det}장 / 미검출 {len(ok) - n_det}장")
    if n_det == 0:
        print("사람이 한 번도 검출되지 않음 -> 사람이 화면에 있었다면 모델/설정 문제가 매우 큼")
    miss_rate = len(gap_idx) / max(n_det + len(gap_idx), 1)
    print(f"[의심1] 앞뒤는 검출됐는데 {args.max_gap}프레임 이하로 끊긴 프레임: {len(gap_idx)}장 "
          f"({len(gaps)}구간), 사람 등장 프레임 대비 {miss_rate:.1%}")
    for a, b in gaps[:15]:
        print(f"    프레임 {int(ok[a]['frame'])}~{int(ok[b]['frame'])} (t={ok[a]['t_s']:.2f}s)")
    if len(gaps) > 15:
        print(f"    ... 외 {len(gaps) - 15}구간")
    motion_sus = [i for i, r in enumerate(ok) if not has[i] and i not in gap_idx
                  and (r.get("motion_frac") or 0) > args.motion]
    print(f"[의심2] 사람 0명인데 화면 움직임이 큰 프레임 (조명 변화일 수도 있음): {len(motion_sus)}장")
    low = [r for r in ok if 0 < (r["max_conf"] or 0) < args.conf + 0.15]
    print(f"[주의]  검출은 됐지만 신뢰도가 기준({args.conf})에 가까운 프레임 (곧 놓칠 수 있음): {len(low)}장")
    partial = 0
    mf = [r["mask_frac"] or 0 for r in ok]
    for i in range(len(ok)):
        if has[i]:
            nb = [mf[j] for j in range(max(0, i - 3), min(len(ok), i + 4)) if j != i and has[j]]
            if nb and mf[i] < 0.5 * median(nb):
                partial += 1
    print(f"[의심3] 검출은 됐는데 마스크 면적이 주변 프레임의 절반 미만 (몸 일부만 가림): {partial}장")

    if gap_idx:
        findings.append(("C 모델", f"짧게 끊긴 미검출 {len(gap_idx)}장 ({miss_rate:.1%}) -> 이 프레임은 원본이 노출됨. "
                                 "tools/offline_eval.py 로 기준 모델과 비교해 확정, --conf 낮추기/--imgsz 올리기 비교"))
    if motion_sus:
        findings.append(("C 모델", f"사람 0명 + 큰 움직임 {len(motion_sus)}장 -> 영상으로 직접 확인 필요"))
    if partial:
        findings.append(("C 모델", f"몸 일부만 가린 의심 프레임 {partial}장 -> --dilate 키우기 / 박스 기반 가림 검토"))
    if low:
        findings.append(("C 모델", f"신뢰도 경계선 프레임 {len(low)}장 -> --conf 를 낮추면 미검출이 줄어드는지 비교"))

    # ------------------------------------------------------------ 종합
    section("종합 판정")
    if not findings:
        print("뚜렷한 문제를 찾지 못함. 사람이 빠르게 움직이기/앉기/눕기/화면 가장자리 등 어려운 상황으로 다시 측정 권장.")
    for cat, msg in findings:
        print(f"- [{cat}] {msg}")
    print("\n참고: [A] 건너뛴 프레임은 화면에 나가지 않아 '노출'이 아님 (끊김/지연 문제).")
    print("      실제 노출 위험은 [C] 프레임. [C]는 정답 없이 추정한 '의심'이므로 offline_eval.py로 확정할 것.")

    if args.plot:
        plot(rows, ok, drops, gap_idx, motion_sus, fd_ms, args.plot)


def plot(rows, ok, drops, gap_idx, motion_sus, fd_ms, path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib이 없어 그래프를 생략합니다.")
        return
    t = [r["t_s"] for r in ok]
    z = lambda k: [r.get(k) or 0 for r in ok]  # noqa: E731
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    ax[0].stackplot(t, z("pre_ms"), z("inf_ms"), z("post_ms"), z("mask_ms"), z("out_ms"),
                    labels=["preprocess", "inference", "postprocess", "mask", "save/show"])
    ax[0].axhline(fd_ms, color="k", ls="--", lw=1, label=f"camera interval {fd_ms:.0f}ms")
    ax[0].set_ylabel("ms / frame")
    ax[0].legend(loc="upper right", fontsize=8, ncol=3)
    ax[0].set_title("[B] processing time per frame")

    ax[1].step(t, z("n_person"), where="mid", label="persons")
    ax[1].plot(t, z("max_conf"), lw=1, label="max conf")
    for i in gap_idx:
        ax[1].axvspan(t[i] - 0.03, t[i] + 0.03, color="red", alpha=0.4)
    for i in motion_sus:
        ax[1].axvspan(t[i] - 0.03, t[i] + 0.03, color="orange", alpha=0.3)
    ax[1].set_ylabel("count / conf")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].set_title("[C] detections (red = short miss gap, orange = motion but no person)")

    if drops:
        ax[2].bar([r["t_s"] for r in rows], drops, width=0.05, color="tab:red")
    ax[2].set_ylabel("skipped frames")
    ax[2].set_title("[A] camera frames skipped before each processed frame")

    ax[3].plot(t, z("cpu_temp"), color="tab:red", label="temp °C")
    ax[3].set_ylabel("°C")
    a2 = ax[3].twinx()
    a2.plot(t, z("cpu_pct"), color="tab:blue", lw=1, label="CPU %")
    a2.set_ylabel("CPU %")
    ax[3].set_xlabel("time (s)")
    ax[3].set_title("[B] temperature / CPU usage")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    print(f"\n그래프 저장: {path}")


if __name__ == "__main__":
    main()
