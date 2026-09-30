#!/usr/bin/env python3
"""Xóa watermark tĩnh (vd. "Dola AI") khỏi video bằng OpenCV inpainting.

Cách hoạt động:
  1. Dò watermark: watermark đứng yên còn nội dung video thay đổi, nên các cạnh
     (edge) xuất hiện ở cùng vị trí trong gần như mọi frame chính là watermark.
     Chỉ tìm trong vùng góc (mặc định góc dưới-phải) để tránh nhầm với nền tĩnh.
  2. Tạo mask từ các cạnh tĩnh đó, nới rộng để phủ cả chữ lẫn bóng đổ.
  3. Inpaint từng frame bằng cv2.inpaint, ghi video mới rồi ghép lại audio gốc.

Cài đặt:
  pip install opencv-contrib-python-headless numpy imageio-ffmpeg
  (bản contrib có thuật toán FSR giữ đường nét tốt hơn; nếu chỉ có opencv-python
   thì script tự dùng Telea)

Sử dụng:
  python remove_watermark.py input.mp4 output.mp4
  python remove_watermark.py input.mp4 output.mp4 --box 605 1232 100 34   # chỉ định vùng x y w h
  python remove_watermark.py input.mp4 output.mp4 --corner bottom-left --save-mask mask.png
"""
import argparse
import shutil
import subprocess
import sys

import cv2
import numpy as np

CORNERS = ("bottom-right", "bottom-left", "top-right", "top-left")


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def corner_roi(w, h, corner, frac_w=0.4, frac_h=0.12):
    rw, rh = int(w * frac_w), int(h * frac_h)
    x = w - rw if "right" in corner else 0
    y = h - rh if "bottom" in corner else 0
    return x, y, rw, rh


def detect_mask(path, corner, samples=40, thresh=0.7):
    """Trả về mask (uint8, 0/255) kích thước bằng frame, đánh dấu vùng watermark."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rx, ry, rw, rh = corner_roi(w, h, corner)

    idx = np.linspace(0, max(n - 1, 0), min(samples, max(n, 1))).astype(int)
    edge_sum = np.zeros((rh, rw), np.float32)
    used = 0
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame[ry:ry + rh, rx:rx + rw], cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 90)
        # nới nhẹ để chịu được nhiễu nén giữa các frame
        edge_sum += cv2.dilate(edges, np.ones((3, 3), np.uint8)) > 0
        used += 1
    cap.release()
    if used == 0:
        sys.exit("Không đọc được frame nào từ video.")

    static = ((edge_sum / used) >= thresh).astype(np.uint8) * 255

    # Giữ các cụm cạnh tĩnh lớn nhất nằm gần nhau (chữ watermark), bỏ nhiễu lẻ
    joined = cv2.morphologyEx(static, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9)))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    if num <= 1:
        sys.exit("Không tự dò được watermark. Hãy dùng --box x y w h.")
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == best).astype(np.uint8) * 255

    # Phủ kín chữ + bóng đổ
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # Nới rộng thêm để ăn hết viền mờ của bóng đổ
    region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))

    mask = np.zeros((h, w), np.uint8)
    mask[ry:ry + rh, rx:rx + rw] = region
    return mask


def box_mask(path, x, y, bw, bh):
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    mask = np.zeros((h, w), np.uint8)
    mask[y:y + bh, x:x + bw] = 255
    return mask


def process(src, dst, mask, radius, method):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if method == "fsr" and not hasattr(cv2, "xphoto"):
        print("Không có cv2.xphoto (cần opencv-contrib-python) -> dùng telea.")
        method = "telea"

    # Chỉ inpaint trên vùng cắt quanh mask cho nhanh
    ys, xs = np.where(mask > 0)
    pad = radius * 4 + 10
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
    sub_mask = mask[y0:y1, x0:x1]
    if method == "fsr":
        fsr_mask = 255 - sub_mask  # xphoto.inpaint: 0 = pixel cần vá
        fsr_out = np.empty((y1 - y0, x1 - x0, 3), np.uint8)

        def inpaint(img):
            cv2.xphoto.inpaint(img, fsr_mask, fsr_out, cv2.xphoto.INPAINT_FSR_FAST)
            return fsr_out
    else:
        flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

        def inpaint(img):
            return cv2.inpaint(img, sub_mask, radius, flag)

    ff = ffmpeg_exe()

    if ff:
        # Ghi thẳng raw frame vào ffmpeg -> H.264 chất lượng cao, giữ audio gốc
        cmd = [ff, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-i", src, "-map", "0:v", "-map", "1:a?",
               "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
               "-c:a", "copy", "-shortest", dst]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        write = proc.stdin.write
    else:
        print("Không tìm thấy ffmpeg: xuất video không có audio (mp4v).")
        writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        write = None

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame[y0:y1, x0:x1] = inpaint(frame[y0:y1, x0:x1])
        if write:
            write(frame.tobytes())
        else:
            writer.write(frame)
        i += 1
        if i % 24 == 0 or i == n:
            print(f"\r  {i}/{n} frame", end="", flush=True)
    print()
    cap.release()

    if ff:
        proc.stdin.close()
        if proc.wait() != 0:
            sys.exit("ffmpeg lỗi khi ghi video.")
    else:
        writer.release()


def main():
    ap = argparse.ArgumentParser(description="Xóa watermark tĩnh khỏi video.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--box", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                    help="Chỉ định vùng watermark thủ công thay vì tự dò")
    ap.add_argument("--corner", choices=CORNERS, default="bottom-right",
                    help="Góc chứa watermark khi tự dò (mặc định bottom-right)")
    ap.add_argument("--radius", type=int, default=5, help="Bán kính inpaint cho telea/ns")
    ap.add_argument("--method", choices=("fsr", "telea", "ns"), default="fsr",
                    help="fsr: giữ đường nét tốt nhất (mặc định); telea/ns: nhanh hơn")
    ap.add_argument("--save-mask", metavar="PNG", help="Lưu mask ra ảnh để kiểm tra")
    args = ap.parse_args()

    if args.box:
        mask = box_mask(args.input, *args.box)
    else:
        mask = detect_mask(args.input, args.corner)
    ys, xs = np.where(mask > 0)
    print(f"Vùng watermark: x={xs.min()} y={ys.min()} w={xs.max() - xs.min() + 1} h={ys.max() - ys.min() + 1}")
    if args.save_mask:
        cv2.imwrite(args.save_mask, mask)

    process(args.input, args.output, mask, args.radius, args.method)
    print(f"Xong: {args.output}")


if __name__ == "__main__":
    main()
