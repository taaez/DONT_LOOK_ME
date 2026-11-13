"""시스템 상태(CPU 사용률/온도/클럭/스로틀링) 측정. 라즈베리파이가 아니면 빈 값(None)을 낸다."""
import subprocess
import time
from pathlib import Path

# vcgencmd get_throttled 비트 의미 (Raspberry Pi 공식 문서)
THROTTLE_BITS = {
    0: "저전압(지금)",
    1: "CPU 클럭 제한(지금)",
    2: "스로틀링(지금)",
    3: "온도 소프트 제한(지금)",
    16: "저전압 발생한 적 있음",
    17: "CPU 클럭 제한 발생한 적 있음",
    18: "스로틀링 발생한 적 있음",
    19: "온도 소프트 제한 발생한 적 있음",
}


def decode_throttled(value) -> list:
    if value is None:
        return []
    return [msg for bit, msg in THROTTLE_BITS.items() if value & (1 << bit)]


def _read_number(path: str):
    try:
        return float(Path(path).read_text().strip())
    except Exception:
        return None


class SysMonitor:
    """interval초에 한 번만 실제로 읽고, 그 사이에는 마지막 값을 돌려준다 (측정 자체가 부담되지 않게)."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.last = -1e9
        self.values = {"cpu_pct": None, "cpu_max_core": None, "cpu_temp": None,
                       "cpu_mhz": None, "throttled": None}
        try:
            import psutil
            self.psutil = psutil
            psutil.cpu_percent(None, percpu=True)  # 첫 호출은 기준점
        except Exception:
            self.psutil = None
        self.has_vcgencmd = True

    def sample(self) -> dict:
        t = time.monotonic()
        if t - self.last < self.interval:
            return self.values
        self.last = t
        v = self.values
        if self.psutil:
            cores = self.psutil.cpu_percent(None, percpu=True)
            if cores:
                v["cpu_pct"] = round(sum(cores) / len(cores), 1)
                v["cpu_max_core"] = round(max(cores), 1)
        temp = _read_number("/sys/class/thermal/thermal_zone0/temp")
        v["cpu_temp"] = round(temp / 1000, 1) if temp is not None else None
        khz = _read_number("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        v["cpu_mhz"] = round(khz / 1000) if khz is not None else None
        if self.has_vcgencmd:
            try:
                out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                                     text=True, timeout=1).stdout  # "throttled=0x0"
                v["throttled"] = int(out.strip().split("=")[1], 16)
            except Exception:
                self.has_vcgencmd = False  # 라즈베리파이가 아니면 다시 시도하지 않는다
        return v
