#!/usr/bin/env python3
"""
Upscale video bằng AI (Real-ESRGAN): 720p -> 2K (1440p) / 4K (2160p).

Khác bản cũ (FFmpeg Lanczos chỉ phóng to điểm ảnh, không tạo thêm chi tiết), bản này
dùng mạng nơ-ron siêu phân giải để "vẽ lại" chi tiết, khử nhiễu và khử vỡ nén.

Quy trình (streaming, không bung frame ra ổ cứng với backend torch):
    FFmpeg decode -> AI x4 (720p -> 2880p) -> thu nhỏ chất lượng cao về 1440p / 2160p
    -> FFmpeg encode, giữ nguyên âm thanh.
Một lần chạy AI xuất được nhiều độ phân giải cùng lúc (mặc định 2K và 4K).

Hai backend:
  * torch (mặc định): PyTorch + spandrel. Nhanh nhất trên NVIDIA (CUDA, fp16), chạy được
    trên Apple Silicon (MPS) và CPU. Nạp được mọi model trên https://openmodeldb.info
  * ncnn: file chạy realesrgan-ncnn-vulkan, dùng Vulkan nên chạy trên mọi GPU
    (AMD / Intel / NVIDIA) mà không cần cài PyTorch.

Ví dụ:
    python upscale_video.py input.mp4                          # ra input_2k.mp4 và input_4k.mp4
    python upscale_video.py input.mp4 --targets 4k --model x4plus --codec hevc_nvenc
    python upscale_video.py input.mp4 --model general --denoise 0.8
    python upscale_video.py input.mp4 --model path/to/4x-UltraSharp.pth
    python upscale_video.py input.mp4 --backend ncnn --ncnn-bin ./realesrgan-ncnn-vulkan
"""

from __future__ import annotations

import argparse
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGETS = {"720p": 720, "1k": 1080, "1080p": 1080, "2k": 1440, "1440p": 1440, "4k": 2160, "2160p": 2160}

RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"
# Model Real-ESRGAN chính thức. "ncnn" là tên model tương ứng trong bản realesrgan-ncnn-vulkan.
MODELS = {
    # Nhanh, dành cho video/ảnh đời thực, hỗ trợ --denoise. Mặc định.
    "general": {"url": f"{RELEASES}/v0.2.5.0/realesr-general-x4v3.pth",
                "wdn_url": f"{RELEASES}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                "ncnn": "realesrgan-x4plus"},
    # Chất lượng cao nhất cho video đời thực, chậm hơn ~10 lần.
    "x4plus": {"url": f"{RELEASES}/v0.1.0/RealESRGAN_x4plus.pth", "ncnn": "realesrgan-x4plus"},
    # Hoạt hình / anime, rất nhanh.
    "anime": {"url": f"{RELEASES}/v0.2.5.0/realesr-animevideov3.pth", "ncnn": "realesr-animevideov3"},
}
MODEL_DIR = Path.home() / ".cache" / "upscale_video" / "models"

MP4_AUDIO_OK = {"aac", "mp3", "ac3", "eac3", "opus", "alac", "flac"}


# ----------------------------------------------------------------------------- FFmpeg

def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("Không tìm thấy FFmpeg. Cài FFmpeg hoặc chạy: pip install imageio-ffmpeg")


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: str
    frames: int | None
    audio_codec: str | None
    color_matrix: str


def probe(ffmpeg: str, path: Path) -> VideoInfo:
    """Đọc thông tin video từ log của `ffmpeg -i` (không cần ffprobe)."""
    err = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    v = re.search(r"Stream #\S+.*?: Video: (.*)", err)
    if not v:
        sys.exit(f"Không đọc được luồng video trong {path}")
    vline = v.group(1)
    w, h = map(int, re.search(r"\b(\d{2,5})x(\d{2,5})\b", vline).groups())
    fps_m = re.search(r"([\d.]+) fps", vline) or re.search(r"([\d.]+) tbr", vline)
    fps = fps_m.group(1) if fps_m else "30"
    # 29.97 / 59.94 / 23.98 thực chất là 30000/1001...
    for num in (24, 30, 60):
        if abs(float(fps) - num * 1000 / 1001) < 0.01:
            fps = f"{num * 1000}/1001"
            break

    frames = None
    d = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if d:
        secs = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3))
        num, _, den = fps.partition("/")
        frames = round(secs * float(num) / float(den or 1))

    a = re.search(r"Stream #\S+.*?: Audio: (\w+)", err)
    # Ma trận màu của nguồn; nếu không gắn tag thì HD mặc định là bt709.
    cm = re.search(r"\((?:tv|pc), (bt709|bt470bg|smpte170m|bt2020nc)", vline)
    matrix = cm.group(1) if cm else ("bt709" if h >= 720 else "bt601")
    matrix = {"bt470bg": "bt601", "smpte170m": "bt601", "bt2020nc": "bt2020"}.get(matrix, matrix)
    return VideoInfo(w, h, fps, frames, a.group(1) if a else None, matrix)


class Decoder:
    def __init__(self, ffmpeg: str, path: Path, info: VideoInfo):
        self.w, self.h = info.width, info.height
        cmd = [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0",
               "-vf", f"scale=in_color_matrix={info.color_matrix}:flags=bicubic+accurate_rnd+full_chroma_int",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        self.size = self.w * self.h * 3

    def read(self) -> np.ndarray | None:
        buf = self.proc.stdout.read(self.size)
        if len(buf) < self.size:
            return None
        return np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)

    def close(self):
        self.proc.stdout.close()
        self.proc.wait()


class Encoder:
    def __init__(self, ffmpeg: str, src: Path, dst: Path, w: int, h: int, info: VideoInfo,
                 codec: str, crf: int, preset: str):
        if codec in ("libx264", "libx265"):
            vcodec = ["-c:v", codec, "-crf", str(crf), "-preset", preset]
        elif codec.endswith("_nvenc"):
            vcodec = ["-c:v", codec, "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
        else:
            vcodec = ["-c:v", codec]
        if codec in ("libx265", "hevc_nvenc"):
            vcodec += ["-tag:v", "hvc1"]  # để QuickTime / iPhone mở được

        mp4 = dst.suffix.lower() in (".mp4", ".mov", ".m4v")
        acodec = ["-c:a", "copy"]
        if info.audio_codec and mp4 and info.audio_codec not in MP4_AUDIO_OK:
            acodec = ["-c:a", "aac", "-b:a", "192k"]

        cmd = [ffmpeg, "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-framerate", info.fps, "-i", "-",
               "-i", str(src), "-map", "0:v:0", "-map", "1:a?",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=bicubic+accurate_rnd+full_chroma_int",
               *vcodec, "-pix_fmt", "yuv420p",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
               *acodec]
        if mp4:
            cmd += ["-movflags", "+faststart"]
        self.proc = subprocess.Popen(cmd + [str(dst)], stdin=subprocess.PIPE)
        self.dst = dst

    def write(self, frame: np.ndarray):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> int:
        self.proc.stdin.close()
        return self.proc.wait()


def target_size(src_w: int, src_h: int, height: int) -> tuple[int, int]:
    w = round(src_w * height / src_h / 2) * 2  # giữ tỉ lệ, chiều rộng chẵn
    return w, height


# ----------------------------------------------------------------------------- Models

def download(url: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODEL_DIR / url.rsplit("/", 1)[-1]
    if not dst.exists():
        print(f"Đang tải model {dst.name} ...")
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)
    return dst


def load_torch_model(name: str, denoise: float):
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader

    if name in MODELS:
        spec = MODELS[name]
        path = download(spec["url"])
        if "wdn_url" in spec and denoise < 1.0:
            # Nội suy trọng số giữa model khử nhiễu mạnh và yếu (cách của Real-ESRGAN gốc).
            wdn = download(spec["wdn_url"])

            def params(p):
                sd = torch.load(p, map_location="cpu", weights_only=True)
                return sd.get("params_ema", sd.get("params", sd))

            a, b = params(path), params(wdn)
            sd = {k: denoise * a[k] + (1 - denoise) * b[k] for k in a}
            model = ModelLoader().load_from_state_dict(sd)
        else:
            model = ModelLoader().load_from_file(path)
    else:
        model = ModelLoader().load_from_file(name)  # file .pth / .safetensors bất kỳ

    if not isinstance(model, ImageModelDescriptor):
        sys.exit("Model này không phải model upscale ảnh.")
    return model


# ----------------------------------------------------------------------------- Backends

class TorchUpscaler:
    def __init__(self, model_name: str, denoise: float, device: str, fp16: bool, tile: int):
        import torch

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        desc = load_torch_model(model_name, denoise)
        self.scale = desc.scale
        self.half = fp16 and device == "cuda" and desc.supports_half
        desc.to(self.device).eval()
        if self.half:
            desc.half()
        self.model = desc
        self.tile = tile
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
        print(f"Backend torch | thiết bị: {self.device} | fp16: {self.half} | "
              f"model: {desc.architecture.name} x{self.scale}")

    def _run(self, x):
        if not self.tile:
            return self.model(x)
        # Chia ô (tile) có chồng mép để tiết kiệm VRAM mà không lộ đường nối.
        s, t, pad = self.scale, self.tile, 16
        _, c, h, w = x.shape
        out = x.new_zeros(x.shape[0], c, h * s, w * s)
        for y0 in range(0, h, t):
            for x0 in range(0, w, t):
                y1, x1 = min(y0 + t, h), min(x0 + t, w)
                py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
                py1, px1 = min(y1 + pad, h), min(x1 + pad, w)
                o = self.model(x[:, :, py0:py1, px0:px1])
                oy, ox = (y0 - py0) * s, (x0 - px0) * s
                out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = o[:, :, oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s]
        return out

    def __call__(self, frames: list[np.ndarray], sizes: list[tuple[int, int]]) -> list[list[np.ndarray]]:
        torch = self.torch
        F = torch.nn.functional
        with torch.inference_mode():
            x = torch.from_numpy(np.stack(frames)).to(self.device)
            x = x.permute(0, 3, 1, 2).to(torch.float16 if self.half else torch.float32) / 255.0
            while True:
                try:
                    y = self._run(x)
                    break
                except torch.cuda.OutOfMemoryError:
                    if self.tile and self.tile <= 128:
                        raise
                    self.tile = self.tile // 2 if self.tile else 512
                    torch.cuda.empty_cache()
                    print(f"\nHết VRAM, chuyển sang chia ô --tile {self.tile}")
            y = y.float()
            results = []
            for w, h in sizes:
                if (y.shape[3], y.shape[2]) == (w, h):
                    r = y
                else:
                    # Thu nhỏ có khử răng cưa (antialias) từ ảnh x4 -> nét hơn phóng thẳng.
                    r = F.interpolate(y, size=(h, w), mode="bicubic", antialias=True, align_corners=False)
                r = (r.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                results.append(list(r))
            return results


class NcnnUpscaler:
    """Gọi realesrgan-ncnn-vulkan theo từng cụm frame trong thư mục tạm."""

    def __init__(self, model_name: str, binary: str | None, gpu_id: str | None, tile: int):
        from PIL import Image

        self.Image = Image
        exe = binary or shutil.which("realesrgan-ncnn-vulkan") or shutil.which("realesrgan-ncnn-vulkan.exe")
        if not exe or not Path(exe).exists():
            sys.exit("Không tìm thấy realesrgan-ncnn-vulkan. Tải tại "
                     "https://github.com/xinntao/Real-ESRGAN/releases rồi truyền --ncnn-bin")
        self.exe = str(Path(exe).resolve())
        self.model = MODELS[model_name]["ncnn"] if model_name in MODELS else model_name
        self.scale = 4
        self.gpu_id, self.tile = gpu_id, tile
        self.tmp = Path(tempfile.mkdtemp(prefix="upscale_"))
        print(f"Backend ncnn-vulkan | model: {self.model} x{self.scale}")

    def __call__(self, frames: list[np.ndarray], sizes: list[tuple[int, int]]) -> list[list[np.ndarray]]:
        inp, out = self.tmp / "in", self.tmp / "out"
        for d in (inp, out):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir()
        for i, f in enumerate(frames):
            self.Image.fromarray(f).save(inp / f"{i:06d}.png", compress_level=1)
        cmd = [self.exe, "-i", str(inp), "-o", str(out), "-n", self.model, "-s", str(self.scale), "-f", "png",
               "-m", str(Path(self.exe).parent / "models")]
        if self.gpu_id is not None:
            cmd += ["-g", self.gpu_id]
        if self.tile:
            cmd += ["-t", str(self.tile)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        results = [[] for _ in sizes]
        for i in range(len(frames)):
            img = self.Image.open(out / f"{i:06d}.png").convert("RGB")
            for k, (w, h) in enumerate(sizes):
                r = img if img.size == (w, h) else img.resize((w, h), self.Image.LANCZOS, reducing_gap=3.0)
                results[k].append(np.asarray(r))
        return results

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- Main

def main() -> None:
    p = argparse.ArgumentParser(description="Upscale video bằng AI (Real-ESRGAN): 720p -> 2K / 4K",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", type=Path, help="Video đầu vào")
    p.add_argument("-o", "--output-dir", type=Path, help="Thư mục lưu kết quả (mặc định: cạnh file gốc)")
    p.add_argument("-t", "--targets", default="2k,4k",
                   help="Độ phân giải đích, cách nhau dấu phẩy: 1k, 2k, 4k hoặc số (vd 1800)")
    p.add_argument("-m", "--model", default="general",
                   help="general | x4plus | anime | đường dẫn file model .pth/.safetensors")
    p.add_argument("--denoise", type=float, default=0.5,
                   help="Mức khử nhiễu 0..1 (chỉ model general). Video nén nhiều thì tăng lên")
    p.add_argument("--backend", choices=["torch", "ncnn"], default="torch")
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | mps | cpu (backend torch)")
    p.add_argument("--no-fp16", action="store_true", help="Tắt fp16 trên CUDA")
    p.add_argument("--tile", type=int, default=0, help="Chia ô khi thiếu VRAM (vd 512). 0 = không chia")
    p.add_argument("--batch", type=int, default=0, help="Số frame xử lý 1 lần (0 = tự chọn)")
    p.add_argument("--ncnn-bin", help="Đường dẫn realesrgan-ncnn-vulkan (backend ncnn)")
    p.add_argument("--gpu-id", help="GPU id cho ncnn (vd 0 hoặc 0,1)")
    p.add_argument("--codec", default="libx264",
                   help="libx264 | libx265 | h264_nvenc | hevc_nvenc | ... (4K nên dùng libx265/hevc_nvenc)")
    p.add_argument("--crf", type=int, default=16, help="Chất lượng encode (thấp = đẹp hơn, file nặng hơn)")
    p.add_argument("--preset", default="slow", help="Preset x264/x265")
    args = p.parse_args()

    src: Path = args.input
    if not src.is_file():
        sys.exit(f"Không tìm thấy file: {src}")

    ffmpeg = find_ffmpeg()
    info = probe(ffmpeg, src)
    print(f"Nguồn: {info.width}x{info.height} @ {info.fps} fps, ~{info.frames or '?'} frame, "
          f"màu {info.color_matrix}, audio: {info.audio_codec or 'không có'}")

    heights = []
    for t in args.targets.split(","):
        t = t.strip().lower()
        heights.append(TARGETS[t] if t in TARGETS else int(t.rstrip("p")))
    sizes = [target_size(info.width, info.height, h) for h in heights]

    if args.backend == "torch":
        try:
            up = TorchUpscaler(args.model, args.denoise, args.device, not args.no_fp16, args.tile)
        except ImportError:
            sys.exit("Thiếu thư viện. Cài: pip install torch torchvision spandrel "
                     "(hoặc dùng --backend ncnn)")
        batch = args.batch or (4 if up.device.type == "cuda" else 1)
    else:
        up = NcnnUpscaler(args.model, args.ncnn_bin, args.gpu_id, args.tile)
        batch = args.batch or 32

    model_h = info.height * up.scale
    for w, h in sizes:
        if h > model_h:
            print(f"Lưu ý: {h}p lớn hơn đầu ra của model ({model_h}p), phần còn lại sẽ phóng bằng bicubic.")

    out_dir = args.output_dir or src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {1080: "1k", 1440: "2k", 2160: "4k"}
    encoders = [Encoder(ffmpeg, src, out_dir / f"{src.stem}_{names.get(h, f'{h}p')}.mp4", w, h, info,
                        args.codec, args.crf, args.preset) for w, h in sizes]

    # Luồng đọc và luồng ghi chạy song song với AI để GPU không phải chờ I/O.
    dec = Decoder(ffmpeg, src, info)
    in_q: queue.Queue = queue.Queue(maxsize=4)
    out_q: queue.Queue = queue.Queue(maxsize=4)

    def reader():
        while True:
            chunk = []
            while len(chunk) < batch:
                f = dec.read()
                if f is None:
                    break
                chunk.append(f)
            if chunk:
                in_q.put(chunk)
            if len(chunk) < batch:
                in_q.put(None)
                return

    write_errors: list[Exception] = []

    def writer():
        while (item := out_q.get()) is not None:
            if write_errors:
                continue  # encoder đã hỏng: vẫn rút hàng đợi để luồng chính không bị treo
            try:
                for enc, frames in zip(encoders, item):
                    for f in frames:
                        enc.write(f)
            except (BrokenPipeError, OSError) as e:
                write_errors.append(e)

    threading.Thread(target=reader, daemon=True).start()
    wt = threading.Thread(target=writer, daemon=True)
    wt.start()

    done, t0 = 0, time.time()
    try:
        while (chunk := in_q.get()) is not None:
            if write_errors:
                print("\nFFmpeg encode bị lỗi, dừng lại.")
                break
            out_q.put(up(chunk, sizes))
            done += len(chunk)
            speed = done / (time.time() - t0)
            eta = f", còn ~{(info.frames - done) / speed / 60:.1f} phút" if info.frames and info.frames > done else ""
            total = f"/{info.frames}" if info.frames else ""
            print(f"\rFrame {done}{total} | {speed:.2f} fps{eta}   ", end="", flush=True)
    finally:
        out_q.put(None)
        wt.join()
        dec.close()
        codes = [e.close() for e in encoders]
        if isinstance(up, NcnnUpscaler):
            up.cleanup()

    print(f"\nXong {done} frame trong {(time.time() - t0) / 60:.1f} phút:")
    for e, c in zip(encoders, codes):
        print(f"  {e.dst}" + ("" if c == 0 else f"  (FFmpeg lỗi, mã {c})"))


if __name__ == "__main__":
    main()
