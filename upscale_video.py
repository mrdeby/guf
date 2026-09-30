#!/usr/bin/env python3
"""
Upscale video theo từng bước: 720p -> 1K (1080p) -> 2K (1440p).

Dùng FFmpeg (scale Lanczos + làm nét nhẹ), giữ nguyên âm thanh và tỉ lệ khung hình.

Cách dùng:
    python upscale_video.py input.mp4
    python upscale_video.py input.mp4 -o output_dir --crf 16 --preset slow
    python upscale_video.py input.mp4 --gpu            # encode bằng NVIDIA NVENC
    python upscale_video.py input.mp4 --keep-1k        # giữ lại file 1080p trung gian

Yêu cầu: FFmpeg trong PATH, hoặc `pip install imageio-ffmpeg`.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Các bước upscale: (tên, chiều cao đích)
STAGES = [
    ("1k", 1080),  # 1920x1080
    ("2k", 1440),  # 2560x1440
]


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("Không tìm thấy FFmpeg. Cài FFmpeg hoặc chạy: pip install imageio-ffmpeg")


def probe_height(ffmpeg: str, path: Path) -> int | None:
    """Lấy chiều cao video (dùng ffprobe nếu có, nếu không thì đọc log của ffmpeg)."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", str(path)],
            capture_output=True, text=True,
        )
        try:
            return int(json.loads(out.stdout)["streams"][0]["height"])
        except (KeyError, IndexError, ValueError):
            return None

    import re

    out = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", out.stderr)
    return int(m.group(2)) if m else None


def upscale(ffmpeg: str, src: Path, dst: Path, height: int, crf: int, preset: str,
            gpu: bool, sharpen: bool) -> None:
    # scale=-2:H giữ tỉ lệ khung hình, chiều rộng luôn chẵn
    vf = f"scale=-2:{height}:flags=lanczos"
    if sharpen:
        vf += ",unsharp=5:5:0.6:5:5:0.0"  # làm nét nhẹ sau khi phóng to

    if gpu:
        vcodec = ["-c:v", "h264_nvenc", "-preset", "p6", "-cq", str(crf), "-b:v", "0"]
    else:
        vcodec = ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]

    cmd = [
        ffmpeg, "-hide_banner", "-y", "-i", str(src),
        "-vf", vf,
        *vcodec,
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",          # giữ nguyên audio
        "-map", "0:v:0", "-map", "0:a?",
        "-movflags", "+faststart",
        str(dst),
    ]
    print(f"\n==> {src.name} -> {dst.name} ({height}p)")
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Upscale video 720p -> 1K (1080p) -> 2K (1440p)")
    p.add_argument("input", type=Path, help="Video đầu vào (720p)")
    p.add_argument("-o", "--output-dir", type=Path, default=None, help="Thư mục lưu kết quả")
    p.add_argument("--crf", type=int, default=18, help="Chất lượng (thấp = đẹp hơn, mặc định 18)")
    p.add_argument("--preset", default="slow", help="Preset x264 (ultrafast..veryslow)")
    p.add_argument("--gpu", action="store_true", help="Dùng NVIDIA NVENC để encode")
    p.add_argument("--no-sharpen", action="store_true", help="Tắt làm nét")
    p.add_argument("--keep-1k", action="store_true", help="Giữ lại file 1K trung gian")
    args = p.parse_args()

    src: Path = args.input
    if not src.is_file():
        sys.exit(f"Không tìm thấy file: {src}")

    ffmpeg = find_ffmpeg()
    h = probe_height(ffmpeg, src)
    if h is not None:
        print(f"Độ phân giải gốc: {h}p")
        if h != 720:
            print("Lưu ý: video gốc không phải 720p, vẫn tiếp tục upscale.")

    out_dir = args.output_dir or src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    current = src
    outputs = []
    for name, height in STAGES:
        dst = out_dir / f"{src.stem}_{name}.mp4"
        upscale(ffmpeg, current, dst, height, args.crf, args.preset, args.gpu, not args.no_sharpen)
        outputs.append(dst)
        current = dst

    if not args.keep_1k:
        outputs[0].unlink(missing_ok=True)
        outputs = outputs[1:]

    print("\nHoàn tất:")
    for f in outputs:
        print(f"  {f}")


if __name__ == "__main__":
    main()
