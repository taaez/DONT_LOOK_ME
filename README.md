# DONT LOOK ME

홈카메라 렌즈 앞에서 촬영 영상 속 **사람만 불투명한 마스크로 가려** 다시 출력하는 프로젝트입니다.
사람의 위치와 움직임은 확인할 수 있게 두고, 외형은 노출하지 않는 것이 목표입니다.

- 하드웨어: Raspberry Pi 5 + Camera Module 3 Wide (`imx708_wide`)
- 모델: YOLOv8n-seg (COCO 사전학습, 사람 클래스만 사용)
- 추론: 현재 CPU (AI HAT+ / Hailo 가속은 아직 미사용)

## 폴더 구조

```
DONT_LOOK_ME/
├── src/person_mask.py     # 메인 프로그램 (카메라 → YOLO 분할 → 사람 마스킹 → 출력/저장)
├── weights/yolov8n-seg.pt # 모델 가중치
├── docs/실행방법.md        # 설치·실행·옵션·문제 해결 상세 문서
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

# 실제 장치 출력: 마스크 처리본만 전체화면
python src/person_mask.py --source picamera --imgsz 320 --single --fullscreen
```

자세한 내용은 [docs/실행방법.md](docs/실행방법.md)를 참고하세요.

## 알려진 한계

- 오류 없이 사람을 **검출하지 못한 프레임**은 원본이 그대로 출력됩니다 (fail-safe는 오류에만 동작).
- 분할 화면(기본)과 `compare.mp4`에는 원본이 포함됩니다. 검증용으로만 쓰고, 실제 장치에서는 `--single`을 사용하세요.
