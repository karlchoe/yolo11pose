# architecture.md — pose_blur.py 아키텍처 및 교훈

## 파이프라인 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                         GPU 메모리                               │
│                                                                 │
│  PyNvVideoCodec    YOLO 감지       YOLO Pose       블러          │
│  ──────────────   (yolo11n+SAHI)  (yolo11n-pose)  (torchvision) │
│  NVDEC → CUDA  →  BBox 목록    →  Keypoints     →  ROI GaussBlur│
│  tensor(RGB)                                                    │
└──────────────────────────────────────┬──────────────────────────┘
                                       │ GPU→CPU (1회/frame)
                                       ▼
                              BGR numpy (cv2 HUD)
                                       │
                                       ▼
                              FFmpeg NVENC pipe → output.mp4
```

### 프레임당 처리 순서

1. **PyNvVideoCodec 디코드** → RGB CUDA tensor (CPU 전송 없음)
2. **GPU→CPU 1회 복사** → BGR numpy (YOLO/cv2/FFmpeg 파이프 공용)
3. **SAHI 감지** (interval 프레임마다): yolo11n.pt → BBox 목록 (블러용)
4. **Pose 추론** (SAHI와 동일 주기): yolo11n-pose.pt → Keypoints (스켈레톤용)
5. **트래커 갱신** (IoU 매칭 또는 ByteTrack ID): BBox + Keypoints 저장
6. **블러** (GPU): 트래커 bbox ROI만 GPU 업로드 → torchvision gaussian_blur → 복사
7. **스켈레톤 그리기** (CPU, cv2): 라인만, 원 없음
8. **HUD** (CPU, cv2): fps / 인원수 / 프레임번호
9. **FFmpeg NVENC pipe** → H.264 인코딩

---

## GPU 처리 범위

| 단계 | 처리 위치 | 비고 |
|------|----------|------|
| 디코딩 | **GPU (NVDEC)** | PyNvVideoCodec → CUDA tensor |
| YOLO 추론 | **GPU (CUDA)** | yolo11n / yolo11n-pose |
| SAHI 슬라이스 분할·NMS | CPU | 라이브러리 내부 |
| 블러 (ROI) | **GPU (CUDA)** | torchvision.TF.gaussian_blur |
| 스켈레톤·HUD | CPU | cv2 |
| 인코딩 | **GPU (NVENC)** | FFmpeg h264_nvenc pipe |
| 인코딩 입력 전송 | CPU→FFmpeg→NVENC | PyNvVideoCodec GPU 인코드 미지원 |

---

## 속도 프로파일 (yolo11n + SAHI 480/0.1, interval=3)

| 단계 | 평균 소요 | 비율 |
|------|----------|------|
| NVC 디코드 + GPU→CPU | 1.0ms | 3% |
| SAHI 추론 | 34.2ms | 89% |
| 블러 (GPU ROI) | 3.1ms | 8% |
| encode pipe write | 0.1ms | 0% |
| **합계 (이론)** | **38.3ms** | **26fps** |
| **실측** | — | **44~47fps** (interval=3) |

### 병목: SAHI

- 640×360 영상, slice=480, overlap=0.1 → 프레임당 ~4회 YOLO 추론
- slice=320, overlap=0.2 → ~7회 → 10fps (느림)
- `--sahi-interval 3` → 3프레임 중 1회만 SAHI → 44fps

---

## 주요 설계 결정 및 교훈

### 1. 디코더: FFmpeg pipe → PyNvVideoCodec

**문제**: FFmpeg pipe 디코드는 NVDEC→CPU→파이프→numpy 경로. CPU 전송이 병목.

**해결**: `PyNvVideoCodec.SimpleDecoder(output_color_type=OutputColorType.RGB)`
- `torch.as_tensor(frame, device='cuda:0')` → CPU 전송 없이 CUDA tensor 획득
- 속도 향상: 7.9fps → 10.6fps (디코드 병목 제거)

**주의사항**:
- Windows에서 torch/lib 경로를 `os.add_dll_directory`로 먼저 등록해야 DLL 로드 가능
- RTSP 스트림은 PyNvVideoCodec 미지원 → FFmpeg pipe 폴백
- `SimpleDecoder`는 랜덤 접근(`dec[i]`) API → 순차 접근으로도 사용 가능

### 2. 인코더: PyNvVideoCodec GPU → FFmpeg NVENC pipe 유지

**시도**: `nvc.CreateEncoder(W, H, 'NV12', usecpuinputbuffer=False)`에 cupy GPU 배열 전달

**결과**: Error code 8 "incorrect usage of CPU input buffer" — v2.1.0에서 GPU 입력 경로 미동작

**결론**: 인코딩 입력은 CPU numpy → FFmpeg NVENC pipe 유지. NVENC 인코딩 자체는 여전히 GPU.

### 3. 블러: 전체 프레임 GPU 업로드 → bbox ROI만 업로드

**v1 (전체 프레임)**: 640×360 전체를 GPU에 올려 블러 후 내림
- 소규모 bbox 2~3개에는 오버헤드가 더 큼

**v2 (ROI만)**: 각 bbox 영역만 GPU 업로드 → 블러 → 복사
- 전송 데이터 대폭 감소, 속도 동등 또는 향상

### 4. 원거리 소형 인물 감지: SAHI 필수

**문제**: 화면 높이 15% 수준의 원거리 인물 → 단일 YOLO 추론에서 bbox가 상반신만 커버하거나 미감지

**해결**: SAHI (Sliced Inference) — 프레임을 겹치는 타일로 분할 후 각각 추론, NMS 병합
- slice=320, overlap=0.2: 7회 추론 → 완전한 전신 bbox 획득
- slice=480, overlap=0.1: 4회 추론 → 품질 유지하며 2.3배 빠름

### 5. SAHI 모델 분리: 감지용 vs pose용

**문제**: `yolo11n-pose.pt`로 SAHI를 돌리면 원거리 감지율이 `yolo11n.pt`보다 낮음

**해결**: 역할별 모델 분리
- `--det-model yolo11n.pt` → SAHI 감지 (블러 bbox 결정)
- `--model yolo11n-pose.pt` → Pose 추론 (스켈레톤 keypoints)

두 추론은 SAHI interval 프레임에서 동시 실행, 중간 프레임은 트래커 유지.

### 6. SAHI interval: 매 프레임 → N프레임마다

**문제**: SAHI가 전체 처리 시간의 89% 차지

**해결**: `--sahi-interval N` — N프레임마다 1회 SAHI 실행, 중간 프레임은 트래커의 마지막 bbox 유지

- interval=1 (매 프레임): 23.9fps
- interval=3 : 44.6fps (1.9배)
- 적정값: 2~5 (빠른 움직임 시 bbox 밀림 가능성 있음)

### 7. yolo26 계열 미사용

**이유**: 이 영상에서 yolo26 end2end 모델은 conf=0.01에서도 신뢰도 0.01~0.02로 사실상 감지 불가.
yolo11n이 동일 장면에서 0.5~0.6으로 훨씬 우수.

**결론**: yolo26n.pt / yolo26n-pose.pt는 기본값에서 제거, yolo11n 계열 사용.

### 8. Ghost Tracking

감지가 일시적으로 끊겨도 `GHOST_FRAMES=5` 동안 마지막 bbox/keypoints 유지.
SAHI interval과 조합 시 중간 프레임의 블러·스켈레톤 연속성 보장.

---

## 트래커 구조

```python
tracker = {
    tid: {
        "bbox":    np.array([x1, y1, x2, y2]),  # 블러 영역
        "kpts":    np.array shape (17, 2),        # COCO keypoints
        "confs":   np.array shape (17,),          # keypoint 신뢰도
        "missing": int,                           # ghost 프레임 카운터
    }
}
```

- ByteTrack ID 우선 사용 (model.track persist=True)
- track_id 없으면 IoU greedy 매칭 폴백
- pose_dets(keypoints)는 SAHI bbox와 별도 IoU 매칭으로 tracker에 갱신

---

## 프레임 소스 추상화

```
NvcFrameSource   — PyNvVideoCodec, 파일 입력, GPU decode
FfmpegFrameSource — FFmpeg pipe, RTSP/기타, SW or NVDEC decode
```

두 소스 모두 `read() -> (bool, BGR numpy)` 인터페이스로 통일.

---

## 파일 구조

```
yolo26/
├── pose_blur.py       # 메인 스크립트
├── CLAUDE.md          # Claude Code 가이드
├── architecture.md    # 아키텍처 및 교훈 (이 파일)
├── context.md         # 초기 작업 컨텍스트 (구버전 참고용)
├── yolo11n.pt         # 감지 모델 (SAHI 블러용)
├── yolo11n-pose.pt    # Pose 모델 (스켈레톤용)
├── 1.mp4 / 2.mp4      # 테스트 영상 (640×360, 25fps, 228프레임)
└── venv312/           # Python 3.12 가상환경
```
