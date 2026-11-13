# DONT LOOK ME

홈카메라 렌즈 앞에서 촬영 영상 속 **사람만 불투명한 마스크로 가려** 다시 출력하는 프로젝트입니다.
사람의 위치와 움직임은 확인할 수 있게 두고, 외형은 노출하지 않는 것이 목표입니다.

- 하드웨어: Raspberry Pi 5 + Camera Module 3 Wide (`imx708_wide`)
- 모델: YOLOv8n-seg (COCO 사전학습, 사람 클래스만 사용)
- 추론: 현재 CPU (AI HAT+ / Hailo 가속은 아직 미사용)

## 폴더 구조

```
DONT_LOOK_ME/
├── src/
│   ├── person_mask.py     # 메인 프로그램 (카메라 → YOLO 분할 → 사람 마스킹 → 출력/저장, --log 진단 기록)
│   ├── camera.py          # 카메라 입력 (Picamera2 / 웹캠 / 영상 파일), 연속 자동초점
│   ├── stream.py          # 브라우저 실시간 화면 (MJPEG)
│   ├── sysmon.py          # CPU 사용률·온도·클럭·스로틀 측정
│   └── metrics.py         # 진단 계산 공용 함수
├── tools/
│   ├── record_raw.py      # AI 없이 원본 녹화 (기준선 + 테스트 영상)
│   ├── analyze_log.py     # --log CSV 분석: [A]간격 / [B]CPU / [C]모델 원인 판정
│   ├── offline_eval.py    # 모든 프레임 오프라인 평가 + 기준 모델 대비 노출 프레임 비율
│   └── compare_methods.py # 같은 원본을 처리한 여러 방식의 출력 영상을 비교 (실제 노출 프레임)
├── weights/yolov8n-seg.pt # 모델 가중치
├── docs/
│   ├── 실행방법.md         # 설치·실행·옵션·문제 해결
│   ├── 프레임진단.md       # 마스킹 누락 원인 진단 방법
│   ├── 학습자료/           # 팀원용 작업 정리·개념 설명
│   ├── 실험결과/           # 날짜별 실험 결과 (예: 261113 기준선 및 main vs seok 비교)
│   └── 계획/               # 앞으로 할 일 (예: 261113 Hailo 도입 계획)
└── requirements.txt
```

## 빠른 실행 (Raspberry Pi)

```bash
git clone https://github.com/taaez/DONT_LOOK_ME.git
cd DONT_LOOK_ME
sudo apt install -y python3-picamera2 python3-opencv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install ultralytics

# SSH 터미널: 화면 없이 원본|마스크 비교 영상 저장 (Ctrl+C로 종료하면 실행 요약 출력)
python src/person_mask.py --source picamera --imgsz 320 --no-show --save compare.mp4

# 브라우저로 실시간 화면 보기 (같은 네트워크의 PC에서 http://dontlookme.local:8000)
python src/person_mask.py --source picamera --imgsz 320 --no-show --stream 8000

# 실제 장치 출력: 마스크 처리본만 전체화면
python src/person_mask.py --source picamera --imgsz 320 --single --fullscreen
```

자세한 내용은 [docs/실행방법.md](docs/실행방법.md)를 참고하세요.

## 프레임 진단

```bash
python src/person_mask.py --source picamera --imgsz 320 --no-show --log runs/live.csv
python tools/analyze_log.py runs/live.csv --plot runs/live.png
```

마스킹 누락의 원인을 간격 / CPU / 모델로 나눠서 봅니다. 자세한 절차는 [docs/프레임진단.md](docs/프레임진단.md)를 참고하세요.

## 최근 실험

- [261113 작업 정리 및 학습 자료 (팀원용, 처음 읽을 문서)](docs/학습자료/261113_작업정리_및_학습자료.md)
- [261113 CPU 기준선 및 main vs seok 비교](docs/실험결과/261113_기준선_및_main_vs_seok_비교.md)
- [261113 Hailo 도입 계획 및 FPS·지연 목표](docs/계획/261113_Hailo_도입_계획.md)

## 알려진 한계

- 오류 없이 사람을 **검출하지 못한 프레임**은 원본이 그대로 출력됩니다 (fail-safe는 오류에만 동작).
- 분할 화면(기본)과 `compare.mp4`에는 원본이 포함됩니다. 검증용으로만 쓰고, 실제 장치에서는 `--single`을 사용하세요.
