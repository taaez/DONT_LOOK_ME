"""진단 스크립트들이 같이 쓰는 계산 함수."""
import numpy as np


def find_short_gaps(has_person, max_gap: int = 3) -> list:
    """앞뒤로 사람이 검출됐는데 중간 1~max_gap 프레임만 검출이 끊긴 구간을 찾는다.

    사람이 1~3프레임(15FPS 기준 0.07~0.2초) 사이에 화면에서 사라졌다 다시 나타날 가능성은 낮으므로,
    이런 구간은 '모델이 놓친 프레임(미검출)'일 가능성이 높다. 반환: [(시작 index, 끝 index), ...]
    """
    gaps = []
    i, n = 0, len(has_person)
    while i < n:
        if has_person[i]:
            i += 1
            continue
        j = i
        while j < n and not has_person[j]:
            j += 1
        # i..j-1 이 검출 없는 구간. 앞뒤가 모두 검출이어야 한다.
        if i > 0 and j < n and (j - i) <= max_gap:
            gaps.append((i, j - 1))
        i = j
    return gaps


def sensor_drops(sensor_ts_ns, frame_dur_ms: float) -> list:
    """연속으로 받은 프레임의 센서 촬영 시각 간격으로, 그 사이에 건너뛴 카메라 프레임 수를 센다.

    반환: 프레임마다 '이 프레임 직전에 건너뛴 수' 리스트 (첫 프레임/타임스탬프 없음은 0).
    예) 카메라 간격 66.7ms인데 두 프레임 간격이 200ms -> 그 사이 2장을 처리하지 못하고 건너뜀.
    """
    drops = [0] * len(sensor_ts_ns)
    prev = None
    for i, ts in enumerate(sensor_ts_ns):
        if ts is None:
            continue
        if prev is not None and frame_dur_ms:
            gap_ms = (ts - prev) / 1e6
            drops[i] = max(int(round(gap_ms / frame_dur_ms)) - 1, 0)
        prev = ts
    return drops


def stats(values) -> dict:
    """None/NaN을 뺀 평균, 중앙값, p95, 최대값."""
    a = np.array([v for v in values if v is not None and v == v], dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def fmt(v, digits: int = 1) -> str:
    return "-" if v is None else f"{v:.{digits}f}"
