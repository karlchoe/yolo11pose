#!/usr/bin/env python3
"""
NVDEC(PyNvVideoCodec) → YOLO26 → BBox 블러(GPU) → NVENC(FFmpeg)
================================================================
Windows 11 / Linux 호환

디코딩: PyNvVideoCodec → RGB CUDA tensor (CPU pipe 없음)
추론:   YOLO on GPU
블러:   torchvision GPU Gaussian blur
인코딩: FFmpeg NVENC pipe

사용법:
  python pose_blur.py -i input.mp4 -o output.mp4
  python pose_blur.py -i input.mp4 -o output.mp4 --sahi --show
  python pose_blur.py -i rtsp://... -o output.mp4   # RTSP: FFmpeg 디코드 폴백
"""

from __future__ import annotations

import subprocess
import sys
import os
import signal
import platform
import argparse
import time
import numpy as np

# PyNvVideoCodec: torch DLL 경로를 먼저 등록해야 로드 가능 (Windows)
def _add_torch_dll_dir():
    try:
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(os.path.abspath(lib))
    except Exception:
        pass

_add_torch_dll_dir()

try:
    import cv2
except ImportError:
    sys.exit("[ERROR] pip install opencv-python")

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] pip install ultralytics")

import torch
import torchvision.transforms.functional as TF

try:
    import PyNvVideoCodec as nvc
    HAS_NVC = True
except Exception:
    HAS_NVC = False

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    HAS_SAHI = True
except ImportError:
    HAS_SAHI = False


IS_WINDOWS = platform.system() == "Windows"
GHOST_FRAMES = 5

# ── COCO 17 Keypoints / Skeleton ──
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
CONF_THR      = 0.3
LIMB_THICKNESS = 2
PERSON_PALETTES = [
    (255, 220,   0), (180,  50, 255), ( 50, 255,  50), ( 50,  50, 255),
    (  0, 165, 255), (255,   0, 200), (255, 180,   0), (  0, 255, 170),
]


# ═══════════════════════════════════════════════════════════════════
#  사전 점검
# ═══════════════════════════════════════════════════════════════════

def check_ffmpeg():
    try:
        r = subprocess.run(["ffmpeg", "-version"],
                           capture_output=True, text=True, timeout=10)
        print(f"[CHECK] FFmpeg: {r.stdout.split(chr(10))[0]}")
    except FileNotFoundError:
        print("[ERROR] ffmpeg를 찾을 수 없습니다!")
        if IS_WINDOWS:
            print("  winget install ffmpeg  또는  https://www.gyan.dev/ffmpeg/builds/")
        else:
            print("  sudo apt install ffmpeg")
        sys.exit(1)

    r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                       capture_output=True, text=True, timeout=10)
    has_nvenc = "h264_nvenc" in r.stdout

    r = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"],
                       capture_output=True, text=True, timeout=10)
    has_cuvid = "h264_cuvid" in r.stdout

    print(f"[CHECK] h264_nvenc : {'OK' if has_nvenc else 'NOT FOUND'}")
    print(f"[CHECK] h264_cuvid : {'OK' if has_cuvid else 'NOT FOUND'}")
    return has_nvenc, has_cuvid


def check_nvidia_gpu():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            print(f"[CHECK] GPU: {r.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    print("[WARN] nvidia-smi를 찾을 수 없습니다")
    return False


def check_cuda_pytorch():
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            print(f"[CHECK] PyTorch CUDA: OK ({torch.cuda.get_device_name(0)})")
        else:
            print("[WARN] PyTorch CUDA 불가")
        return cuda_ok
    except ImportError:
        print("[ERROR] PyTorch 미설치")
        return False


# ═══════════════════════════════════════════════════════════════════
#  BBox 블러 (GPU)
# ═══════════════════════════════════════════════════════════════════

def draw_skeleton(frame: np.ndarray, kpts: np.ndarray,
                  confs: np.ndarray, color: tuple) -> None:
    """스켈레톤 선만 그리기 (원 없음)."""
    for i, j in SKELETON:
        if confs[i] > CONF_THR and confs[j] > CONF_THR:
            cv2.line(frame,
                     (int(kpts[i][0]), int(kpts[i][1])),
                     (int(kpts[j][0]), int(kpts[j][1])),
                     color, LIMB_THICKNESS, cv2.LINE_AA)


def blur_bboxes(frame: np.ndarray, boxes: list,
                ksize: int = 51, pad: int = 10,
                device: str = "cpu") -> None:
    """bbox ROI만 GPU로 전송해 Gaussian blur."""
    if not boxes:
        return
    h, w = frame.shape[:2]
    ksize = ksize if ksize % 2 == 1 else ksize + 1

    if device != "cpu":
        for box in boxes:
            x1 = max(0, int(box[0]) - pad)
            y1 = max(0, int(box[1]) - pad)
            x2 = min(w, int(box[2]) + pad)
            y2 = min(h, int(box[3]) + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            roi_t = (torch.from_numpy(frame[y1:y2, x1:x2])
                     .to(device).permute(2, 0, 1).unsqueeze(0).float())
            blurred = TF.gaussian_blur(roi_t, kernel_size=ksize)
            frame[y1:y2, x1:x2] = (blurred.squeeze(0).permute(1, 2, 0)
                                    .byte().cpu().numpy())
    else:
        for box in boxes:
            x1 = max(0, int(box[0]) - pad)
            y1 = max(0, int(box[1]) - pad)
            x2 = min(w, int(box[2]) + pad)
            y2 = min(h, int(box[3]) + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (ksize, ksize), 0)


# ═══════════════════════════════════════════════════════════════════
#  트래커
# ═══════════════════════════════════════════════════════════════════

def _iou_box(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (union + 1e-6)


def _match_tracks(tracker: dict, dets: list):
    tids = list(tracker.keys())
    if not tids or not dets:
        return {}, tids, list(range(len(dets)))
    iou_mat = np.array([[_iou_box(tracker[t]["bbox"], d) for d in dets]
                        for t in tids])
    matched, used_t, used_d = {}, set(), set()
    while iou_mat.max() >= 0.15:
        i, j = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
        matched[tids[i]] = j
        used_t.add(i); used_d.add(j)
        iou_mat[i, :] = -1; iou_mat[:, j] = -1
    unmatched_tids = [tids[i] for i in range(len(tids)) if i not in used_t]
    unmatched_dets = [j for j in range(len(dets)) if j not in used_d]
    return matched, unmatched_tids, unmatched_dets


def _dets_from_yolo(results) -> list:
    dets = []
    if not results or results[0].boxes is None:
        return dets
    boxes_obj = results[0].boxes
    kpt_obj   = results[0].keypoints
    for i in range(len(boxes_obj)):
        box      = boxes_obj.xyxy[i].cpu().numpy()
        track_id = int(boxes_obj.id[i]) if boxes_obj.id is not None else None
        kpts  = kpt_obj.xy[i].cpu().numpy()   if kpt_obj is not None else None
        confs = kpt_obj.conf[i].cpu().numpy() if kpt_obj is not None else None
        dets.append({"bbox": box, "track_id": track_id, "kpts": kpts, "confs": confs})
    return dets


def _dets_from_sahi(sahi_result) -> list:
    dets = []
    for pred in sahi_result.object_prediction_list:
        if pred.category.name != "person":
            continue
        b = pred.bbox
        dets.append({
            "bbox": np.array([b.minx, b.miny, b.maxx, b.maxy], dtype=np.float32),
            "track_id": None, "kpts": None, "confs": None,
        })
    return dets


def process_frame(frame: np.ndarray, dets: list, blur_k: int, blur_pad: int,
                  tracker: dict, next_tid: list, device: str = "cpu",
                  pose_dets: list | None = None) -> int:
    use_ext_ids = bool(dets) and dets[0].get("track_id") is not None

    if use_ext_ids:
        active_tids = set()
        for det in dets:
            tid = det["track_id"]
            active_tids.add(tid)
            if tid in tracker:
                tracker[tid].update({"bbox": det["bbox"], "missing": 0,
                                     "kpts": det["kpts"], "confs": det["confs"]})
            else:
                tracker[tid] = {"bbox": det["bbox"], "missing": 0,
                                "kpts": det["kpts"], "confs": det["confs"]}
        for tid in list(tracker.keys()):
            if tid not in active_tids:
                tracker[tid]["missing"] += 1
        for tid in [t for t in list(tracker) if tracker[t]["missing"] > GHOST_FRAMES]:
            del tracker[tid]
    else:
        boxes = [d["bbox"] for d in dets]
        matched, unmatched_tids, unmatched_dets = _match_tracks(tracker, boxes)
        for tid, di in matched.items():
            tracker[tid].update({"bbox": boxes[di], "missing": 0,
                                 "kpts": dets[di]["kpts"], "confs": dets[di]["confs"]})
        expired = [t for t in unmatched_tids
                   if tracker[t]["missing"] + 1 > GHOST_FRAMES]
        for tid in unmatched_tids:
            if tid not in expired:
                tracker[tid]["missing"] += 1
        for tid in expired:
            del tracker[tid]
        for di in unmatched_dets:
            tid = next_tid[0]; next_tid[0] += 1
            tracker[tid] = {"bbox": boxes[di], "missing": 0,
                            "kpts": dets[di]["kpts"], "confs": dets[di]["confs"]}

    # SAHI 모드: pose_dets의 keypoints를 IoU로 매칭해 tracker에 갱신
    if pose_dets:
        for pd in pose_dets:
            if pd.get("kpts") is None:
                continue
            best_tid, best_iou = None, 0.3
            for tid, t in tracker.items():
                iou = _iou_box(t["bbox"], pd["bbox"])
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid is not None:
                tracker[best_tid]["kpts"]  = pd["kpts"]
                tracker[best_tid]["confs"] = pd["confs"]

    # Phase 1: 블러
    blur_bboxes(frame, [t["bbox"] for t in tracker.values()],
                ksize=blur_k, pad=blur_pad, device=device)

    # Phase 2: 스켈레톤 (keypoints 있는 경우만)
    for idx, (tid, t) in enumerate(tracker.items()):
        if t.get("kpts") is not None and t.get("confs") is not None:
            color = PERSON_PALETTES[tid % len(PERSON_PALETTES)]
            draw_skeleton(frame, t["kpts"], t["confs"], color)

    return len(tracker)


# ═══════════════════════════════════════════════════════════════════
#  HUD
# ═══════════════════════════════════════════════════════════════════

def draw_hud(frame: np.ndarray, fps: float, n_p: int, n_f: int,
             dec_tag: str, enc_tag: str) -> None:
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame,
                f"{dec_tag}>{enc_tag} {fps:.1f}fps {n_p}person(s) #{n_f}",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 0), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════
#  FFmpeg 인코더 CLI
# ═══════════════════════════════════════════════════════════════════

def build_enc(dst: str, w: int, h: int, fps: int, gpu: int,
              hw: bool, br: str, preset: str, tune: str,
              gop: int | None) -> list:
    c = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-video_size", f"{w}x{h}", "-framerate", str(fps), "-i", "pipe:0",
         "-vf", "format=yuv420p"]
    if hw:
        g = gop or fps * 2
        c += ["-c:v", "h264_nvenc", "-gpu", str(gpu),
              "-preset", preset, "-tune", tune, "-profile:v", "high",
              "-rc", "cbr", "-b:v", br, "-maxrate", br, "-bufsize", br,
              "-g", str(g), "-bf", "0", "-surfaces", "32", "-zerolatency", "1"]
    else:
        c += ["-c:v", "libx264", "-preset", "fast", "-tune", "zerolatency",
              "-b:v", br, "-g", str(gop or fps * 2)]
    if dst.startswith("rtmp"):
        c += ["-f", "flv", dst]
    elif dst.startswith("rtsp"):
        c += ["-f", "rtsp", "-rtsp_transport", "tcp", dst]
    elif dst.startswith(("udp", "srt")):
        c += ["-f", "mpegts", dst]
    else:
        c += ["-movflags", "+faststart", dst]
    return c


# FFmpeg 디코더 (RTSP 등 PyNvVideoCodec 미지원 소스용 폴백)
def build_dec(src: str, w: int, h: int, gpu: int,
              hw: bool, fps: int | None) -> list:
    c = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
    if hw:
        c += ["-hwaccel", "cuda", "-hwaccel_device", str(gpu),
              "-c:v", "h264_cuvid", "-resize", f"{w}x{h}"]
    if src.startswith("rtsp"):
        c += ["-rtsp_transport", "tcp", "-buffer_size", "4194304",
              "-max_delay", "500000",
              "-fflags", "+genpts+discardcorrupt", "-flags", "low_delay"]
    c += ["-i", src]
    if fps:
        c += ["-r", str(fps)]
    if not hw:
        c += ["-vf", f"scale={w}:{h}"]
    c += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-sn", "pipe:1"]
    return c


# ═══════════════════════════════════════════════════════════════════
#  프레임 소스 추상화
# ═══════════════════════════════════════════════════════════════════

class NvcFrameSource:
    """PyNvVideoCodec 기반 GPU 디코더."""

    def __init__(self, path: str, gpu: int, target_w: int, target_h: int):
        self.dec = nvc.SimpleDecoder(path, gpu_id=gpu,
                                     output_color_type=nvc.OutputColorType.RGB)
        meta = self.dec.get_stream_metadata()
        self.src_w = meta.width
        self.src_h = meta.height
        self.fps   = meta.average_fps
        self.total = meta.num_frames
        self.target_w = target_w
        self.target_h = target_h
        self.idx   = 0
        self.device = f"cuda:{gpu}"

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.idx >= self.total:
            return False, None
        frame = self.dec[self.idx]
        self.idx += 1

        # GPU 텐서로 변환 (RGB, uint8, CUDA) — CPU 전송 없음
        t = torch.as_tensor(frame, device=self.device)  # (H, W, 3)

        # 필요 시 GPU에서 리사이즈
        if t.shape[1] != self.target_w or t.shape[0] != self.target_h:
            t = (t.permute(2, 0, 1).unsqueeze(0).float()
                 / 255.0)
            t = torch.nn.functional.interpolate(
                t, size=(self.target_h, self.target_w),
                mode="bilinear", align_corners=False)
            t = (t.squeeze(0).permute(1, 2, 0) * 255).byte()

        # RGB → BGR numpy (YOLO/cv2/FFmpeg 파이프용 — 1회 전송)
        bgr = t[:, :, [2, 1, 0]].contiguous().cpu().numpy()
        return True, bgr

    def release(self):
        pass


class FfmpegFrameSource:
    """FFmpeg pipe 기반 디코더 (RTSP 등 폴백)."""

    def __init__(self, cmd: list, w: int, h: int, popen_extra: dict):
        self.fb = w * h * 3
        self.W  = w
        self.H  = h
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     bufsize=self.fb * 4, **popen_extra)

    def read(self) -> tuple[bool, np.ndarray | None]:
        raw = self.proc.stdout.read(self.fb)
        if len(raw) < self.fb:
            return False, None
        return True, np.frombuffer(raw, np.uint8).reshape(self.H, self.W, 3).copy()

    def release(self):
        try:
            self.proc.stdout.close()
        except Exception:
            pass
        try:
            self.proc.terminate(); self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        try:
            err = self.proc.stderr.read().decode(errors="replace").strip()
            if err:
                print(f"\n[DEC LOG]\n{err}")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="PyNvVideoCodec 디코드 → YOLO26 → GPU 블러 → NVENC\n"
                    "Windows 11 / Linux compatible")

    p.add_argument("-i", "--input",  required=True)
    p.add_argument("-o", "--output", default="output_blur.mp4")
    p.add_argument("--fps",         type=int, default=25)
    p.add_argument("--aspect",      default="16:9")

    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--no-nvdec",    action="store_true",
                   help="PyNvVideoCodec 대신 FFmpeg SW 디코딩 강제")
    p.add_argument("--no-nvenc",    action="store_true")
    p.add_argument("--bitrate",     default="2M")
    p.add_argument("--preset",      default="p4")
    p.add_argument("--tune",        default="ll")
    p.add_argument("--gop",         type=int, default=None)

    p.add_argument("-m", "--model",     default="yolo11n-pose.pt",
                   help="pose 추론 모델 (스켈레톤용)")
    p.add_argument("--det-model",       default="yolo11n.pt",
                   help="SAHI 감지 모델 (블러용, --sahi 사용 시)")
    p.add_argument("--imgsz",       type=int, default=1280)
    p.add_argument("--conf",        type=float, default=0.15)
    p.add_argument("--max-det",     type=int, default=20)

    p.add_argument("--blur-strength", type=int, default=51)
    p.add_argument("--blur-pad",      type=int, default=10)

    p.add_argument("--sahi",           action="store_true")
    p.add_argument("--sahi-slice",    type=int, default=480)
    p.add_argument("--sahi-overlap",  type=float, default=0.1)
    p.add_argument("--sahi-interval", type=int, default=1,
                   help="SAHI를 N프레임마다 실행 (중간 프레임은 트래커 유지, 기본 1=매프레임)")

    p.add_argument("--show",          action="store_true")
    p.add_argument("--print-cmd",     action="store_true")
    p.add_argument("--check",         action="store_true")

    args = p.parse_args()

    # ── 환경 점검 ──
    print(f"[SYS] {platform.system()} {platform.release()} "
          f"{'x64' if platform.machine().endswith('64') else platform.machine()}")
    print()
    has_gpu = check_nvidia_gpu()
    has_nvenc, has_cuvid = check_ffmpeg()
    has_cuda = check_cuda_pytorch()

    is_rtsp = args.input.startswith("rtsp")
    use_nvc  = HAS_NVC and has_cuda and not args.no_nvdec and not is_rtsp
    print(f"[CHECK] PyNvVideoCodec: {'OK (GPU decode)' if use_nvc else 'NOT USED'}")
    if is_rtsp:
        print("[INFO]  RTSP → FFmpeg 디코드 폴백")
    print()

    if args.check:
        print("[CHECK] 점검 완료"); return

    if not has_nvenc and not args.no_nvenc:
        print("[AUTO] NVENC 불가 → libx264")
        args.no_nvenc = True
    if not has_cuda:
        print("[AUTO] CUDA 불가 → CPU 추론")

    # ── 해상도 ──
    aw, ah = map(int, args.aspect.split(":"))
    W, H = 640, int(640 * ah / aw)
    H = H if H % 2 == 0 else H + 1
    fb = W * H * 3

    # ── 인코더 CLI ──
    enc_cmd = build_enc(args.output, W, H, args.fps, args.gpu,
                        not args.no_nvenc, args.bitrate,
                        args.preset, args.tune, args.gop)
    et = "NVENC" if not args.no_nvenc else "x264"

    print(f"[RES] {W}x{H}  aspect={args.aspect}")
    if not use_nvc:
        dec_cmd = build_dec(args.input, W, H, args.gpu,
                            has_cuvid and not args.no_nvdec, args.fps)
        print(f"[DEC] {' '.join(dec_cmd)}")
    else:
        print(f"[DEC] PyNvVideoCodec GPU decode  ({args.input})")
    print(f"[ENC] {' '.join(enc_cmd)}\n")
    if args.print_cmd:
        return

    # ── YOLO / SAHI 로드 ──
    dev = f"cuda:{args.gpu}" if has_cuda else "cpu"
    if args.sahi and not HAS_SAHI:
        print("[WARN] sahi 미설치 → --sahi 무시")
        args.sahi = False

    # pose 모델 (스켈레톤)
    print(f"[MODEL] pose  : {args.model} -> {dev}")
    model = YOLO(args.model)
    model.to(dev)
    model.predict(np.zeros((H, W, 3), np.uint8), imgsz=args.imgsz, verbose=False)

    # SAHI 감지 모델 (블러용, 별도 det 모델 사용)
    sahi_model = None
    if args.sahi:
        print(f"[MODEL] detect: {args.det_model} -> {dev} [SAHI]")
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=args.det_model,
            confidence_threshold=args.conf, device=dev)
        print(f"[SAHI] slice={args.sahi_slice} overlap={args.sahi_overlap} interval={args.sahi_interval}")

    print("[MODEL] ready\n")

    # ── 프로세스 시작 ──
    popen_extra = {}
    if IS_WINDOWS:
        popen_extra["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

    # 디코더
    if use_nvc:
        src = NvcFrameSource(args.input, args.gpu, W, H)
        dt = "NVC+GPU"
    else:
        dec_cmd = build_dec(args.input, W, H, args.gpu,
                            has_cuvid and not args.no_nvdec, args.fps)
        src = FfmpegFrameSource(dec_cmd, W, H, popen_extra)
        dt = "NVDEC" if (has_cuvid and not args.no_nvdec) else "SW"

    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE,
                           stderr=subprocess.PIPE, bufsize=fb * 4,
                           **popen_extra)

    # ── Ctrl+C 핸들링 ──
    running = True

    def stop_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    if not IS_WINDOWS:
        signal.signal(signal.SIGTERM, stop_handler)

    tracker: dict = {}
    next_tid: list = [0]
    n, t0, fv = 0, time.time(), 0.0

    print(f"[RUN] {dt} → Blur(GPU) → {et}  "
          f"blur_k={args.blur_strength} max_det={args.max_det}")
    print("[RUN] Ctrl+C to stop\n")

    try:
        while running:
            ok, frame = src.read()
            if not ok:
                print("[DEC] stream ended")
                break

            # ── YOLO / SAHI 추론 ──
            pose_dets = None
            if args.sahi and (n % args.sahi_interval == 0):
                # 블러용: SAHI (원거리 감지 향상)
                sahi_res = get_sliced_prediction(
                    frame, sahi_model,
                    slice_height=args.sahi_slice,
                    slice_width=args.sahi_slice,
                    overlap_height_ratio=args.sahi_overlap,
                    overlap_width_ratio=args.sahi_overlap,
                    verbose=0)
                dets = _dets_from_sahi(sahi_res)
                # 스켈레톤용: pose 모델 단일 추론
                pose_res = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                         device=dev, max_det=args.max_det,
                                         classes=[0], verbose=False)
                pose_dets = _dets_from_yolo(pose_res)
            elif args.sahi:
                dets = []   # 트래커가 이전 bbox 유지
            else:
                res = model.track(frame, imgsz=args.imgsz, conf=args.conf,
                                  device=dev, max_det=args.max_det,
                                  classes=[0], persist=True, verbose=False)
                dets = _dets_from_yolo(res)

            # ── 블러 (GPU) ──
            n_persons = process_frame(frame, dets,
                                      args.blur_strength, args.blur_pad,
                                      tracker, next_tid, device=dev,
                                      pose_dets=pose_dets)

            n += 1
            if n % 10 == 0:
                d = time.time() - t0
                fv = n / d if d > 0 else 0
            draw_hud(frame, fv, n_persons, n, dt, et)

            try:
                enc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                print("[ENC] pipe closed")
                break

            if args.show:
                cv2.imshow("Blur", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")

    finally:
        el = time.time() - t0
        print(f"\n[DONE] {n} frames / {el:.1f}s = {n/el:.1f} fps" if el > 0
              else f"\n[DONE] {n} frames")

        src.release()

        try:
            if enc.stdin and not enc.stdin.closed:
                enc.stdin.close()
        except Exception:
            pass
        try:
            enc.wait(timeout=10)
        except Exception:
            enc.kill()

        if args.show:
            cv2.destroyAllWindows()

        try:
            err = enc.stderr.read().decode(errors="replace").strip()
            if err:
                print(f"\n[ENC LOG]\n{err}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
