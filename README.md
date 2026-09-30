# guf

## upscale_video.py — Upscale video bằng AI (720p → 2K / 4K)

Phóng to bằng FFmpeg (Lanczos, bicubic…) chỉ nội suy điểm ảnh nên lên 2K/4K gần như không
nét hơn. Script này dùng **Real-ESRGAN** — mạng siêu phân giải "vẽ lại" chi tiết, khử nhiễu và
khử vỡ nén — rồi thu nhỏ chất lượng cao từ ảnh x4 về đúng độ phân giải đích.

```
720p ──AI x4──> 2880p ──thu nhỏ antialias──> 1440p (2K)
                                         └──> 2160p (4K)   (một lần chạy AI, xuất nhiều file)
```

### Cài đặt

```bash
# 1. PyTorch đúng với GPU (NVIDIA ví dụ CUDA 12.8; xem pytorch.org cho máy khác)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 2. Các thư viện còn lại
pip install -r requirements.txt
```

Model tự tải về `~/.cache/upscale_video/models` ở lần chạy đầu.

### Dùng

```bash
python upscale_video.py video.mp4                            # ra video_2k.mp4 + video_4k.mp4
python upscale_video.py video.mp4 -t 4k --codec hevc_nvenc   # chỉ 4K, encode H.265 bằng GPU
python upscale_video.py video.mp4 -m x4plus                  # chất lượng cao nhất (chậm)
python upscale_video.py video.mp4 --denoise 0.8              # video nén nhiều, nhiễu nhiều
python upscale_video.py anime.mp4 -m anime                   # hoạt hình / anime
python upscale_video.py video.mp4 --tile 512                 # GPU ít VRAM
```

| Model (`-m`) | Dùng cho | Tốc độ |
|---|---|---|
| `general` (mặc định) | Video đời thực, có `--denoise 0..1` | Nhanh |
| `x4plus` | Video đời thực, chi tiết tốt nhất | Chậm ~10 lần |
| `anime` | Hoạt hình, anime | Rất nhanh |
| `đường/dẫn/model.pth` | Bất kỳ model nào trên [OpenModelDB](https://openmodeldb.info) (4x-UltraSharp, Nomos, HAT, SwinIR, DAT…) | Tuỳ model |

Tuỳ chọn khác: `--codec libx265|h264_nvenc|hevc_nvenc`, `--crf` (mặc định 16), `--device cuda|mps|cpu`,
`--batch`, `--no-fp16`. Xem đủ bằng `python upscale_video.py -h`.

### GPU AMD / Intel (không cần PyTorch)

Tải `realesrgan-ncnn-vulkan` tại [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases)
(giải nén, giữ thư mục `models` cạnh file chạy), rồi:

```bash
pip install numpy pillow imageio-ffmpeg
python upscale_video.py video.mp4 --backend ncnn --ncnn-bin ./realesrgan-ncnn-vulkan
```

### Tốc độ tham khảo

Nên chạy trên GPU. Trên CPU, model `general` mất khoảng 6 giây mỗi frame 720p, chỉ hợp để chạy thử.
`x4plus` chậm hơn `general` nhiều lần; `anime` nhanh nhất.

### Tham khảo

- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — model gốc
- [chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel) — nạp model đủ kiến trúc, thay cho `basicsr` (đã hỏng với torchvision mới)
- [k4yt3x/video2x](https://github.com/k4yt3x/video2x) — app có sẵn giao diện, dùng cùng các model
- Muốn ổn định giữa các frame hơn (ít nhấp nháy) có thể xem các model video như
  [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR), đổi lại rất nặng.
