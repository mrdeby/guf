#!/usr/bin/env python3
"""Xóa watermark tĩnh (vd. "Veo", "Dola AI") khỏi video và ảnh.

Cách hoạt động:
  1. Dò watermark: lấy trung vị theo thời gian của vùng góc video. Nền chuyển
     động bị nhòe đi, còn watermark đứng yên vẫn sắc nét -> lọc thông cao để
     tách watermark. Mỗi preset (veo, dola) chỉ tìm trong một ô nhỏ ở góc.
  2. Tạo mask từ vùng đó, nới rộng để phủ cả viền chữ lẫn bóng đổ.
  3. Khôi phục vùng watermark:
     - Watermark trắng bán trong suốt (vd. Veo): tính ngược alpha
       B = (I - a*255) / (1 - a) để lấy lại đúng chi tiết nền bên dưới -> sắc nét.
       Alpha được ước lượng từ chính video (hồi quy qua các frame), hoặc lấy từ
       mẫu có sẵn trong thư mục watermarks/ (dùng cho ảnh tĩnh).
     - Watermark đục / nền đứng yên (vd. Dola AI): không tính ngược được -> inpaint
       (FSR) từ các điểm xung quanh.
     Chế độ auto tự kiểm tra và chọn cách phù hợp.
  4. Ghi video mới rồi ghép lại audio gốc.

Cài đặt:
  pip install -r requirements.txt
  (bản contrib có thuật toán FSR giữ đường nét tốt hơn; nếu chỉ có opencv-python
   thì script tự dùng Telea)

Sử dụng:
  python remove_watermark.py input.mp4 output.mp4 --preset veo
  python remove_watermark.py input.mp4 output.mp4 --preset dola
  python remove_watermark.py thu_muc_video/ thu_muc_ra/ --preset veo       # xử lý cả thư mục
  python remove_watermark.py input.mp4 output.mp4                          # tự dò trong cả góc
  python remove_watermark.py anh.png anh_sach.png --preset veo             # ảnh tĩnh (png/jpg/webp)
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
# "template" = mẫu alpha của logo (watermarks/*.png, 16-bit, alpha*65535) và
# "margin" = khoảng cách (phải, dưới) từ mẫu tới mép frame ở 720p.
PRESETS = {
    "veo":  {"corner": "bottom-right", "win": (140, 70), "grow": 4,
             "template": "veo.png", "margin": (12, 13)},
    "dola": {"corner": "bottom-right", "win": (220, 100), "grow": 7},
}
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watermarks")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


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


# ---------------------------------------------------------------------------
# Tính ngược alpha (reverse alpha blending)
#
# Watermark trắng bán trong suốt được trộn vào ảnh:  I = a*255 + (1 - a)*B
# Nếu biết alpha a của từng pixel thì lấy lại đúng nền gốc:
#     B = (I - a*255) / (1 - a)
# -> giữ nguyên chi tiết thật, không phải "đoán" như inpaint nên không bị mờ.
# ---------------------------------------------------------------------------

def unblend(img, alpha, fill=None, denoise=True):
    """Gỡ watermark trắng khỏi img (uint8 HxWx3) theo alpha (float HxW, 0..1).

    - Pixel có alpha quá cao (gần đục) không còn thông tin nền -> lấy từ `fill`
      (ảnh đã inpaint) nếu có.
    - Phép chia (1 - a) khuếch đại nhiễu nén video -> khử nhiễu nhẹ, giữ cạnh,
      mạnh dần theo alpha.
    """
    a = np.clip(alpha, 0, 0.95)[..., None]
    f = img.astype(np.float32)
    rec = np.clip((f - a * 255) / (1 - a), 0, 255)
    if denoise:
        smooth = cv2.bilateralFilter(rec.astype(np.uint8), 5, 20, 3).astype(np.float32)
        k = np.clip(a / 0.5, 0, 1) * 0.5
        rec = rec + k * (smooth - rec)
    if fill is not None:
        w = np.clip((a - 0.75) / 0.15, 0, 1)
        rec = w * fill.astype(np.float32) + (1 - w) * rec
    return (rec + 0.5).astype(np.uint8)


def static_score(stack, s=1.0):
    """Độ "đứng yên" của chi tiết cao tần theo thời gian (dùng để kiểm tra kết quả)."""
    gray = stack.astype(np.float32).mean(3)
    hp = gray - np.array([cv2.GaussianBlur(x, (0, 0), 1.5 * s) for x in gray])
    return np.abs(hp.mean(0)) / (hp.std(0) + 2)


def read_crops(src, box, samples):
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
            crops.append(frame[y0:y1, x0:x1].copy())
        i += 1
    cap.release()
    return np.array(crops)


def estimate_alpha(crops, sub_mask, fill_fn):
    """Ước lượng alpha của watermark trắng từ chính video.

    Mỗi frame: đoán nền B^ bằng inpaint (đúng ở tần số thấp), rồi hồi quy
    I - 255 = (1 - a) * (B^ - 255) trên mọi frame và 3 kênh màu -> alpha rất sạch.
    """
    m = sub_mask > 0
    A = crops.astype(np.float32)
    B = np.array([fill_fn(c) for c in crops], np.float32)
    num = ((A - 255) * (B - 255)).sum((0, 3))
    den = ((B - 255) ** 2).sum((0, 3)) + 1e-3
    alpha = 1 - num / den
    alpha[~m] = 0
    alpha[alpha < 0.03] = 0
    return np.clip(alpha, 0, 1).astype(np.float32)


def check_unblend(crops, alpha, fills, sub_mask, s):
    """True nếu sau khi tính ngược không còn dấu vết watermark đứng yên."""
    m = sub_mask > 0
    rec = np.array([unblend(c, alpha, f, denoise=False) for c, f in zip(crops, fills)])
    before = np.percentile(static_score(crops, s)[~m], 95)
    after = np.percentile(static_score(rec, s)[m], 95)
    return after < 2 * before + 0.2


# ---------------------------------------------------------------------------
# Mẫu alpha cố định (template) - dùng cho ảnh tĩnh hoặc video nền đứng yên
# ---------------------------------------------------------------------------

def load_template(preset, s):
    p = PRESETS.get(preset, {})
    if "template" not in p:
        return None
    path = os.path.join(TEMPLATE_DIR, p["template"])
    t = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if t is None:
        return None
    t = t.astype(np.float32) / 65535
    if abs(s - 1) > 1e-3:
        t = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    return t


def locate_template(gray, tpl, preset, search=10):
    """Tìm vị trí mẫu logo quanh vị trí dự kiến. Trả về (x, y, score)."""
    h, w = gray.shape
    s = min(w, h) / 720
    mr, mb = PRESETS[preset]["margin"]
    th, tw = tpl.shape
    ex, ey = int(round(w - mr * s - tw)), int(round(h - mb * s - th))
    r = int(search * s) + 2
    x0, y0 = max(ex - r, 0), max(ey - r, 0)
    x1, y1 = min(ex + tw + r, w), min(ey + th + r, h)
    region = gray[y0:y1, x0:x1].astype(np.float32)
    hp = region - cv2.GaussianBlur(region, (0, 0), 2 * s)
    t_hp = tpl - cv2.GaussianBlur(tpl, (0, 0), 2 * s)
    res = cv2.matchTemplate(hp, t_hp, cv2.TM_CCOEFF_NORMED)
    _, score, _, (bx, by) = cv2.minMaxLoc(res)
    return x0 + bx, y0 + by, float(score)


def remove_from_array(rgb, trusted=True, preset="veo", min_score=0.3):
    """Xóa watermark khỏi một ảnh (numpy HxWx3). Trả về (ảnh sạch, region | None).

    trusted=True: chắc chắn ảnh có watermark -> xử lý ở vị trí khớp nhất.
    trusted=False: chỉ xử lý khi khớp mẫu đủ tốt, không thì trả về (rgb, None).
    """
    h, w = rgb.shape[:2]
    s = min(w, h) / 720
    tpl = load_template(preset, s)
    if tpl is None:
        raise ValueError(f"Preset '{preset}' không có mẫu alpha trong {TEMPLATE_DIR}")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    x, y, score = locate_template(gray, tpl, preset)
    if not trusted and score < min_score:
        return rgb, None
    th, tw = tpl.shape
    x, y = max(0, min(x, w - tw)), max(0, min(y, h - th))
    out = rgb.copy()
    out[y:y + th, x:x + tw] = unblend(rgb[y:y + th, x:x + tw], tpl)
    return out, (x, y, tw, th)


def remove_watermark(data, ext, trusted=True, preset="veo"):
    """Xóa watermark khỏi ảnh dạng bytes. Giữ kênh trong suốt và định dạng gốc."""
    from PIL import Image
    import io

    with Image.open(io.BytesIO(data)) as im:
        im.load()
        has_alpha = im.mode in ("RGBA", "LA") or "transparency" in im.info
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)

    cleaned, region = remove_from_array(np.ascontiguousarray(rgba[..., :3]), trusted, preset)
    if region is None:
        return data, None

    if has_alpha:
        output_image = Image.fromarray(np.dstack([cleaned, rgba[..., 3]]), "RGBA")
    else:
        output_image = Image.fromarray(cleaned, "RGB")

    image_format = {"jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}.get(ext.lower().lstrip("."), "PNG")
    options = {"quality": 95} if image_format in ("JPEG", "WEBP") else {}
    output = io.BytesIO()
    output_image.save(output, image_format, **options)
    return output.getvalue(), region


# ---------------------------------------------------------------------------
# Inpaint (dự phòng cho watermark đục)
# ---------------------------------------------------------------------------

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


def template_alpha(src, preset, box, s):
    """Đặt mẫu alpha của preset vào vùng cắt `box` (dò vị trí trên frame trung vị)."""
    tpl = load_template(preset, s)
    if tpl is None:
        return None, 0.0
    w, h, _ = video_info(src)
    cap = cv2.VideoCapture(src)
    _, _, n = video_info(src)
    frames = []
    for i in np.linspace(0, max(n - 1, 0), min(15, max(n, 1))).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    gray = np.median(np.array(frames), axis=0).astype(np.uint8)
    x, y, score = locate_template(gray, tpl, preset)
    x0, y0, x1, y1 = box
    alpha = np.zeros((y1 - y0, x1 - x0), np.float32)
    th, tw = tpl.shape
    # chỉ phần mẫu nằm trong box
    ax0, ay0 = max(x, x0), max(y, y0)
    ax1, ay1 = min(x + tw, x1), min(y + th, y1)
    if ax1 > ax0 and ay1 > ay0:
        alpha[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = tpl[ay0 - y:ay1 - y, ax0 - x:ax1 - x]
    return alpha, score


def process(src, dst, mask, radius, method, preset="auto"):
    w, h, n = video_info(src)
    s = min(w, h) / 720
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    fallback = "fsr" if hasattr(cv2, "xphoto") else "telea"
    if method == "fsr" and fallback != "fsr":
        print("Không có cv2.xphoto (cần opencv-contrib-python) -> dùng telea.")
        method = "telea"

    # Chỉ xử lý trên vùng cắt quanh mask cho nhanh
    ys, xs = np.where(mask > 0)
    pad = max(radius * 4 + 10, int(24 * s))
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
    box = (x0, y0, x1, y1)
    sub_mask = mask[y0:y1, x0:x1]
    fill = make_inpainter(fallback, sub_mask, radius)
    alpha = None

    if method in ("auto", "unblend"):
        crops = read_crops(src, box, 60)
        fills = np.array([fill(c) for c in crops])
        est = estimate_alpha(crops, sub_mask, fill)
        if method == "unblend" or check_unblend(crops, est, fills, sub_mask, s):
            alpha = est
            print(f"Tính ngược alpha ước lượng từ video (alpha max {alpha.max():.2f}).")
        elif preset in PRESETS and "template" in PRESETS[preset]:
            tpl_alpha, score = template_alpha(src, preset, box, s)
            if tpl_alpha is not None and score > 0.3 and check_unblend(crops, tpl_alpha, fills, sub_mask, s):
                alpha = tpl_alpha
                print(f"Tính ngược alpha theo mẫu {preset} (độ khớp {score:.2f}).")
    if method == "template":
        alpha, score = template_alpha(src, preset, box, s)
        if alpha is None:
            sys.exit(f"Preset '{preset}' không có mẫu alpha.")
        print(f"Tính ngược alpha theo mẫu {preset} (độ khớp {score:.2f}).")

    if alpha is not None:
        need_fill = bool((alpha > 0.75).any())

        def inpaint(img):
            return unblend(img, alpha, fill(img) if need_fill else None)
    else:
        if method in ("auto", "unblend"):
            print(f"Watermark đục / không tính ngược được -> dùng {fallback}.")
            method = fallback
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
    process(src, dst, mask, args.radius, args.method, args.preset)
    print(f"Xong: {dst}")


def remove_image(src, dst, args):
    print(f"== {src}")
    preset = "veo" if args.preset == "auto" else args.preset
    with open(src, "rb") as f:
        data = f.read()
    out, region = remove_watermark(data, os.path.splitext(dst)[1], not args.untrusted, preset)
    if region is None:
        print("Không thấy watermark -> giữ nguyên ảnh.")
    else:
        print(f"Vùng watermark: x={region[0]} y={region[1]} w={region[2]} h={region[3]}")
    with open(dst, "wb") as f:
        f.write(out)
    print(f"Xong: {dst}")


def main():
    ap = argparse.ArgumentParser(description="Xóa watermark tĩnh (Veo, Dola AI, ...) khỏi video/ảnh.")
    ap.add_argument("input", help="File video/ảnh hoặc thư mục chứa video/ảnh")
    ap.add_argument("output", help="File ra, hoặc thư mục ra nếu input là thư mục")
    ap.add_argument("--preset", choices=("auto", *PRESETS), default="auto",
                    help="Loại watermark: veo, dola, hoặc auto (tìm trong cả góc)")
    ap.add_argument("--box", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                    help="Chỉ định vùng watermark thủ công thay vì tự dò")
    ap.add_argument("--corner", choices=CORNERS, default="bottom-right",
                    help="Góc chứa watermark khi --preset auto (mặc định bottom-right)")
    ap.add_argument("--radius", type=int, default=5, help="Bán kính inpaint cho telea/ns")
    ap.add_argument("--method", choices=("auto", "unblend", "template", "fsr", "telea", "ns"),
                    default="auto",
                    help="auto (mặc định): tính ngược alpha (ước lượng từ video, rồi tới mẫu), "
                         "không được thì fsr. unblend: luôn ước lượng alpha từ video. "
                         "template: luôn dùng mẫu alpha của preset. fsr/telea/ns: chỉ inpaint")
    ap.add_argument("--untrusted", action="store_true",
                    help="Ảnh: chỉ xử lý khi chắc chắn tìm thấy watermark")
    ap.add_argument("--save-mask", metavar="PNG", help="Lưu mask ra ảnh để kiểm tra")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = sorted(f for f in os.listdir(args.input) if f.lower().endswith(VIDEO_EXTS + IMAGE_EXTS))
        if not files:
            sys.exit(f"Không có video/ảnh nào trong {args.input}")
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in IMAGE_EXTS:
                remove_image(os.path.join(args.input, f), os.path.join(args.output, base + "_clean" + ext), args)
            else:
                remove_one(os.path.join(args.input, f), os.path.join(args.output, base + "_clean.mp4"), args)
    elif args.input.lower().endswith(IMAGE_EXTS):
        remove_image(args.input, args.output, args)
    else:
        remove_one(args.input, args.output, args)


if __name__ == "__main__":
    main()
