#!/usr/bin/env python3
"""Xóa watermark tĩnh (vd. "Veo", "Dola AI") khỏi video bằng OpenCV inpainting.

Cách hoạt động:
  1. Dò watermark: lấy trung vị theo thời gian của vùng góc video. Nền chuyển
     động bị nhòe đi, còn watermark đứng yên vẫn sắc nét -> lọc thông cao để
     tách watermark. Mỗi preset (veo, dola) chỉ tìm trong một ô nhỏ ở góc.
  2. Tạo mask từ vùng đó, nới rộng để phủ cả viền chữ lẫn bóng đổ.
  3. Khôi phục vùng watermark:
     - Watermark bán trong suốt (vd. Veo): tính ngược alpha để lấy lại đúng chi tiết
       nền bên dưới -> sắc nét, không bị mờ.
     - Watermark đục / nền đứng yên (vd. Dola AI): không tính ngược được -> inpaint
       (FSR) từ các điểm xung quanh.
     Chế độ auto tự kiểm tra và chọn cách phù hợp.
  4. Ghi video mới rồi ghép lại audio gốc.

Cài đặt:
  pip install opencv-contrib-python-headless numpy imageio-ffmpeg
  (bản contrib có thuật toán FSR giữ đường nét tốt hơn; nếu chỉ có opencv-python
   thì script tự dùng Telea)

Sử dụng:
  python remove_watermark.py input.mp4 output.mp4 --preset veo
  python remove_watermark.py input.mp4 output.mp4 --preset dola
  python remove_watermark.py thu_muc_video/ thu_muc_ra/ --preset veo       # xử lý cả thư mục
  python remove_watermark.py input.mp4 output.mp4                          # tự dò trong cả góc
  python remove_watermark.py input.mp4 output.mp4 --box 605 1232 100 34   # chỉ định vùng x y w h
  python remove_watermark.py input.mp4 output.mp4 --corner bottom-left --save-mask mask.png
"""
import argparse
import os
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


# Vùng tìm watermark cho từng loại (kích thước tính ở video 720p, tự co giãn theo
# độ phân giải). "grow" = số pixel nới mask để phủ viền/bóng đổ.
PRESETS = {
    "veo":  {"corner": "bottom-right", "win": (140, 70), "grow": 4},
    "dola": {"corner": "bottom-right", "win": (220, 100), "grow": 7},
}


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Không mở được video: {path}")
    info = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def search_window(w, h, corner, win=None):
    """Vùng (x, y, w, h) để tìm watermark ở một góc của frame."""
    if win:
        s = min(w, h) / 720
        rw, rh = min(int(win[0] * s), w), min(int(win[1] * s), h)
    else:
        rw, rh = int(w * 0.4), int(h * 0.12)
    x = w - rw if "right" in corner else 0
    y = h - rh if "bottom" in corner else 0
    return x, y, rw, rh


def detect_mask(path, corner, win=None, grow=6, samples=120):
    """Trả về mask (uint8, 0/255) kích thước bằng frame, đánh dấu vùng watermark.

    Ở mỗi frame lấy chi tiết cao tần (ảnh trừ ảnh mờ). Nền chuyển động nên chi
    tiết của nó thay đổi liên tục; watermark đứng yên nên chi tiết gần như không
    đổi. Điểm = trung bình / độ lệch chuẩn theo thời gian -> watermark nổi bật.
    """
    w, h, n = video_info(path)
    rx, ry, rw, rh = search_window(w, h, corner, win)
    s = min(w, h) / 720
    sigma = 3 * s

    cap = cv2.VideoCapture(path)
    step = max(1, n // samples)
    hps = []
    i = 0
    while True:
        if i % step:
            if not cap.grab():
                break
        else:
            ok, frame = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(frame[ry:ry + rh, rx:rx + rw], cv2.COLOR_BGR2GRAY).astype(np.float32)
            hps.append(g - cv2.GaussianBlur(g, (0, 0), sigma))
        i += 1
    cap.release()
    if len(hps) < 2:
        sys.exit("Video quá ngắn, không dò được watermark. Hãy dùng --box x y w h.")

    hps = np.array(hps)
    # Chữ watermark sáng hơn xung quanh -> chỉ lấy phần dương.
    # Ngưỡng kép: "strong" để chắc chắn là watermark, "weak" để lấy đủ phần chữ nhạt.
    score = hps.mean(0) / (hps.std(0) + 2)
    strong = score > max(0.25 * score.max(), 1.5)
    weak = (score > max(0.1 * score.max(), 1.0)).astype(np.uint8) * 255

    # Nối các chữ cái thành cụm
    k = max(3, int(9 * s)) | 1
    joined = cv2.morphologyEx(weak, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    has_strong = np.bincount(labels[strong], minlength=num) > 0
    seeds = [j for j in range(1, num) if has_strong[j] and stats[j, cv2.CC_STAT_AREA] >= 15 * s * s]
    if not seeds:
        sys.exit("Không tự dò được watermark. Hãy dùng --box x y w h.")

    # Watermark nằm sát góc: chọn cụm có mép gần góc khung tìm nhất
    def corner_dist(j):
        x, y, cw, ch = stats[j, :4]
        dx = rw - (x + cw) if "right" in corner else x
        dy = rh - (y + ch) if "bottom" in corner else y
        return dx + dy
    seed = min(seeds, key=corner_dist)
    sx, sy, sw, sh = stats[seed, :4]
    # Gom thêm các cụm cùng hàng chữ, sát cạnh (vd. "Dola" và "AI" tách rời)
    gap = 12 * s
    keep = [j for j in range(1, num)
            if stats[j, cv2.CC_STAT_AREA] >= 15 * s * s
            and stats[j, 1] < sy + sh and stats[j, 1] + stats[j, 3] > sy
            and stats[j, 0] < sx + sw + gap and stats[j, 0] + stats[j, 2] > sx - gap]
    region = np.isin(labels, keep).astype(np.uint8) * 255

    # Nới rộng để phủ hết viền chữ + bóng đổ
    g = max(1, int(grow * s))
    region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))

    mask = np.zeros((h, w), np.uint8)
    mask[ry:ry + rh, rx:rx + rw] = region
    return mask


def box_mask(path, x, y, bw, bh):
    w, h, _ = video_info(path)
    mask = np.zeros((h, w), np.uint8)
    mask[y:y + bh, x:x + bw] = 255
    return mask


def fit_unblend(src, sub_mask, box, samples=300):
    """Ước lượng watermark bán trong suốt để tính ngược ra nền gốc (giữ độ nét).

    Watermark trộn vào ảnh theo dạng: I = g * B + c  (g = 1 - alpha, c = alpha * màu chữ).
    Theo thời gian: mean(I) = g*mean(B) + c, std(I) = g*std(B). mean(B), std(B) bên
    trong mask được nội suy từ các pixel xung quanh (thống kê nền thay đổi chậm), từ đó
    suy ra g, c cho từng pixel và B = (I - c) / g.

    Trả về (g, c, weight, ok). ok=False nếu tính ngược không xóa được watermark
    (vd. chữ gần như đục, hoặc nền đứng yên) -> nên dùng inpaint.
    """
    x0, y0, x1, y1 = box
    _, _, n = video_info(src)
    step = max(1, n // samples)
    cap = cv2.VideoCapture(src)
    crops = []
    i = 0
    while True:
        if i % step:
            if not cap.grab():
                break
        else:
            ok, frame = cap.read()
            if not ok:
                break
            crops.append(frame[y0:y1, x0:x1].astype(np.float32))
        i += 1
    cap.release()
    m = sub_mask > 0
    if len(crops) < 10 or m.all():
        return None, None, None, False

    A = np.array(crops)
    mean, std = A.mean(0), A.std(0) + 1e-3
    outside = (~m).astype(np.float32)
    s = min(video_info(src)[:2]) / 720

    def fill_from_outside(v, sigma=6 * s):
        num = cv2.GaussianBlur(v * outside[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(outside, (0, 0), sigma)[..., None] + 1e-6
        return np.where(m[..., None], num / den, v)

    mean_b, std_b = fill_from_outside(mean), fill_from_outside(std)
    g = cv2.GaussianBlur(std / std_b, (0, 0), 0.7)
    g = np.clip(np.where(g > 0.92, 1.0, g), 0.05, 1.0)
    c = np.where(g >= 1.0, 0.0, mean - g * mean_b)
    # Chỗ chữ gần như đục (g nhỏ) không đủ thông tin -> để inpaint lo
    weight = np.clip((g.min(2) - 0.2) / 0.2, 0, 1)[..., None]
    weight[~m] = 0

    # Kiểm tra: sau khi tính ngược, chi tiết "đứng yên" của watermark phải biến mất
    def static_score(stack):
        gray = stack.mean(3)
        hp = gray - np.array([cv2.GaussianBlur(x, (0, 0), 1.5 * s) for x in gray])
        return np.abs(hp.mean(0)) / (hp.std(0) + 2)

    rec = np.clip((A - c) / g, 0, 255)
    rec = weight * rec + (1 - weight) * A
    solid = weight[..., 0] > 0.5
    before = np.percentile(static_score(A)[~m], 95)
    after = np.percentile(static_score(rec)[solid], 95) if solid.any() else np.inf
    ok = after < 2 * before + 0.2
    return g.astype(np.float32), c.astype(np.float32), weight.astype(np.float32), ok


def make_inpainter(method, sub_mask, radius):
    keep_orig = (sub_mask == 0)[..., None]
    if method == "fsr":
        # FSR lỗi màu khi mask sát mép ảnh -> thêm viền phản chiếu rồi cắt lại
        b = 16
        h, w = sub_mask.shape
        fsr_mask = cv2.copyMakeBorder(255 - sub_mask, b, b, b, b, cv2.BORDER_CONSTANT, value=255)
        fsr_out = np.empty((h + 2 * b, w + 2 * b, 3), np.uint8)

        def inpaint(img):
            big = cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_REFLECT)
            out = cv2.xphoto.inpaint(big, fsr_mask, fsr_out, cv2.xphoto.INPAINT_FSR_FAST)
            out = fsr_out if out is None else out
            # chỉ thay pixel trong mask, phần còn lại giữ nguyên ảnh gốc
            return np.where(keep_orig, img, out[b:-b, b:-b])
        return inpaint

    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return lambda img: cv2.inpaint(img, sub_mask, radius, flag)


def process(src, dst, mask, radius, method):
    w, h, n = video_info(src)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    fallback = "fsr" if hasattr(cv2, "xphoto") else "telea"
    if method == "fsr" and fallback != "fsr":
        print("Không có cv2.xphoto (cần opencv-contrib-python) -> dùng telea.")
        method = "telea"

    # Chỉ xử lý trên vùng cắt quanh mask cho nhanh
    ys, xs = np.where(mask > 0)
    pad = max(radius * 4 + 10, int(24 * min(w, h) / 720))
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
    sub_mask = mask[y0:y1, x0:x1]

    if method in ("auto", "unblend"):
        g, c, weight, ok = fit_unblend(src, sub_mask, (x0, y0, x1, y1))
        if g is None or (method == "auto" and not ok):
            print(f"Watermark không tính ngược được (chữ đục / nền tĩnh) -> dùng {fallback}.")
            method = fallback
        else:
            print("Dùng tính ngược alpha (giữ chi tiết nền dưới watermark).")

    if method in ("auto", "unblend"):
        fill = make_inpainter(fallback, sub_mask, radius)
        need_fill = bool((weight[sub_mask > 0] < 1).any())

        def inpaint(img):
            f = img.astype(np.float32)
            rec = np.clip((f - c) / g, 0, 255)
            base = fill(img).astype(np.float32) if need_fill else f
            return (weight * rec + (1 - weight) * base + 0.5).astype(np.uint8)
    else:
        inpaint = make_inpainter(method, sub_mask, radius)

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


VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")


def remove_one(src, dst, args):
    print(f"== {src}")
    if args.box:
        mask = box_mask(src, *args.box)
    elif args.preset == "auto":
        mask = detect_mask(src, args.corner)
    else:
        p = PRESETS[args.preset]
        mask = detect_mask(src, p["corner"], p["win"], p["grow"])
    ys, xs = np.where(mask > 0)
    print(f"Vùng watermark: x={xs.min()} y={ys.min()} w={xs.max() - xs.min() + 1} h={ys.max() - ys.min() + 1}")
    if args.save_mask:
        cv2.imwrite(args.save_mask, mask)
    process(src, dst, mask, args.radius, args.method)
    print(f"Xong: {dst}")


def main():
    ap = argparse.ArgumentParser(description="Xóa watermark tĩnh (Veo, Dola AI, ...) khỏi video.")
    ap.add_argument("input", help="File video hoặc thư mục chứa video")
    ap.add_argument("output", help="File video ra, hoặc thư mục ra nếu input là thư mục")
    ap.add_argument("--preset", choices=("auto", *PRESETS), default="auto",
                    help="Loại watermark: veo, dola, hoặc auto (tìm trong cả góc)")
    ap.add_argument("--box", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                    help="Chỉ định vùng watermark thủ công thay vì tự dò")
    ap.add_argument("--corner", choices=CORNERS, default="bottom-right",
                    help="Góc chứa watermark khi --preset auto (mặc định bottom-right)")
    ap.add_argument("--radius", type=int, default=5, help="Bán kính inpaint cho telea/ns")
    ap.add_argument("--method", choices=("auto", "unblend", "fsr", "telea", "ns"), default="auto",
                    help="auto (mặc định): tính ngược alpha nếu được, không thì fsr. "
                         "unblend: luôn tính ngược alpha. fsr/telea/ns: chỉ inpaint")
    ap.add_argument("--save-mask", metavar="PNG", help="Lưu mask ra ảnh để kiểm tra")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = sorted(f for f in os.listdir(args.input) if f.lower().endswith(VIDEO_EXTS))
        if not files:
            sys.exit(f"Không có video nào trong {args.input}")
        for f in files:
            name = os.path.splitext(f)[0] + "_clean.mp4"
            remove_one(os.path.join(args.input, f), os.path.join(args.output, name), args)
    else:
        remove_one(args.input, args.output, args)


if __name__ == "__main__":
    main()
