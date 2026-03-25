# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`pose_blur.py` — GPU 가속 영상 처리 파이프라인:

**PyNvVideoCodec(NVDEC) → YOLO 감지(SAHI) + Pose 추론 → BBox Gaussian 블러(GPU) + 스켈레톤 오버레이 → FFmpeg NVENC**

Windows 11 / Linux 호환. NVIDIA 하드웨어 미사용 시 소프트웨어 폴백.

---

## 의존성

```bash
pip install ultralytics opencv-python torch torchvision numpy
pip install sahi pynvvideocodec cupy-cuda12x
```

- FFmpeg full 빌드 필요 (NVENC/NVDEC 포함)
  - Windows: https://www.gyan.dev/ffmpeg/builds/ (`ffmpeg-release-full.7z`)
  - Linux: `sudo apt install ffmpeg`

---

## 실행

```bash
# 기본
venv312/Scripts/python pose_blur.py -i input.mp4 -o output.mp4

# SAHI + pose 스켈레톤 (권장)
venv312/Scripts/python pose_blur.py -i input.mp4 -o output.mp4 --sahi --sahi-interval 3

# RTSP 스트림 (FFmpeg 디코드 폴백)
venv312/Scripts/python pose_blur.py -i rtsp://192.168.0.100:554/stream -o output.mp4 --sahi

# 환경 점검
venv312/Scripts/python pose_blur.py --check -i dummy

# FFmpeg 명령 출력만
venv312/Scripts/python pose_blur.py -i input.mp4 -o output.mp4 --print-cmd
```

---

## 주요 CLI 옵션

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `-m/--model` | `yolo11n-pose.pt` | pose 추론 모델 (스켈레톤용) |
| `--det-model` | `yolo11n.pt` | SAHI 감지 모델 (블러용, --sahi 시) |
| `--fps` | 25 | 출력 프레임레이트 |
| `--aspect` | `16:9` | 종횡비 (기준 폭 640px) |
| `--blur-strength` | 51 | Gaussian 블러 커널 크기 (홀수) |
| `--blur-pad` | 10 | BBox 블러 여유 px |
| `--conf` | 0.15 | 감지 신뢰도 임계값 |
| `--max-det` | 20 | 프레임당 최대 감지 인원 |
| `--sahi` | off | SAHI 슬라이스 추론 활성화 |
| `--sahi-slice` | 480 | SAHI 슬라이스 크기 |
| `--sahi-overlap` | 0.1 | SAHI 슬라이스 겹침 비율 |
| `--sahi-interval` | 1 | SAHI를 N프레임마다 실행 |
| `--no-nvdec` | off | PyNvVideoCodec 대신 SW 디코드 강제 |
| `--no-nvenc` | off | libx264 인코딩 강제 |
| `--gpu` | 0 | GPU 인덱스 |

---

## 환경

- OS: Windows 11 x64
- GPU: NVIDIA GeForce RTX 5050 Laptop GPU (Blackwell sm_120)
- Python: 3.12 (venv312) — Python 3.14는 PyTorch CUDA 미지원
- 실행 prefix: `venv312/Scripts/python pose_blur.py ...`
- CUDA: 12.8, torch 2.11.0+cu128

### 설치된 주요 패키지

| 패키지 | 버전 |
|--------|------|
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| opencv-python | 4.11.0.86 |
| numpy | 2.4.3 |
| ultralytics | 8.4.26 |
| sahi | 0.11.36 |
| pynvvideocodec | 2.1.0 |
| cupy-cuda12x | 14.0.1 |
| lapx | 0.9.4 (ByteTrack 의존) |

---

## 알려진 특성 / 주의사항

- **yolo26 계열 감지 실패**: 이 영상에서 end2end 모드 recall 낮음 (conf=0.01에서도 0.01~0.02). yolo11n이 동일 장면에서 0.5~0.6으로 우수
- **원거리 소형 피사체** (화면 높이 15% 수준) → SAHI 필수, imgsz=1280 필수
- **PyNvVideoCodec Windows DLL**: torch/lib 경로를 `os.add_dll_directory`로 먼저 등록해야 로드 가능. CUDA_PATH 환경변수 불필요
- **PyNvVideoCodec GPU 인코드 미지원**: v2.1.0에서 `usecpuinputbuffer=False` 경로가 동작 안 함 (Error code 8). 인코딩은 FFmpeg NVENC pipe 사용
- **SAHI 모델 분리**: 블러용(yolo11n.pt)과 pose용(yolo11n-pose.pt)을 반드시 분리. pose 모델로 SAHI를 돌리면 원거리 감지율 저하
- **픽셀 포맷**: NVENC 출력 시 `-vf format=yuv420p` 필수 (없으면 gbrp 인코딩 → 재생 불가)
- **Python 3.14 불가**: PyTorch CUDA 빌드가 3.9~3.12까지만 존재 → venv312 사용
- **cupy CUDA path 경고**: cupy가 CUDA_PATH를 못 찾아 경고 출력하나 동작에 무관 (torch DLL로 로드됨)

---

## 테스트 영상

- `1.mp4` / `2.mp4`: 640×360, 25fps, 228프레임, 해안 원거리 촬영 (2명)
