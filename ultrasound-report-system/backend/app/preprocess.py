"""Image preprocessing, copied from notebook 02 (02-preprocessing-splits).

An uploaded sheet goes through exactly the steps the training frames went
through: panel decomposition -> banner removal + sector crop -> CLAHE in LAB
space + median despeckle -> letterbox to 224 -> drop uninformative frames.
Keep these functions in sync with the notebook. If you change a value in
CFG here, the web app no longer matches the paper.
"""
from __future__ import annotations

import cv2
import numpy as np

# Values from notebook 02, cell 3 (saved there as preprocess_config.json).
# The bundle's config.json can override them (see bundle.py).
CFG = {
    "img_size": 224,
    "min_panel_area_frac": 0.010,
    "min_panel_side": 56,
    "close_kernel": 5,
    "banner_frac": 0.11,
    "clahe_clip": 2.5,
    "clahe_grid": 8,
    "speckle_filter": "median",
}


def foreground_mask(gray: np.ndarray, close: int | None = None) -> np.ndarray:
    close = close or CFG["close_kernel"]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    floor = max(18, int(thr * 0.5))
    th = cv2.bitwise_and(th, (blur > floor).astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
    return th


def split_components(mask: np.ndarray) -> list:
    h, w = mask.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < CFG["min_panel_area_frac"] * h * w:
            continue
        if bw < CFG["min_panel_side"] or bh < CFG["min_panel_side"]:
            continue
        if not (0.20 < bw / bh < 5.0):
            continue
        boxes.append((int(x), int(y), int(x + bw), int(y + bh)))
    return boxes


def _cut_points(profile: np.ndarray, length: int, min_gap: int, frac=0.04) -> list:
    thresh = frac * profile.max() if profile.max() > 0 else 0
    low = profile <= thresh
    cuts, start = [], None
    for i, v in enumerate(low):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_gap:
                cuts.append((start + i) // 2)
            start = None
    if start is not None and length - start >= min_gap:
        cuts.append((start + length) // 2)
    return cuts


def split_grid(mask: np.ndarray) -> list:
    h, w = mask.shape
    r_cuts = sorted({0, h, *_cut_points(mask.sum(1) / 255.0, h, max(5, h // 80))})
    c_cuts = sorted({0, w, *_cut_points(mask.sum(0) / 255.0, w, max(5, w // 80))})
    boxes = []
    for a, b in zip(r_cuts[:-1], r_cuts[1:]):
        for c, d in zip(c_cuts[:-1], c_cuts[1:]):
            if (b - a) < CFG["min_panel_side"] or (d - c) < CFG["min_panel_side"]:
                continue
            cell = mask[a:b, c:d]
            if (cell > 0).mean() < 0.08:
                continue
            ys, xs = np.where(cell > 0)
            boxes.append((c + int(xs.min()), a + int(ys.min()),
                          c + int(xs.max()) + 1, a + int(ys.max()) + 1))
    return boxes


def _refine_oversized(mask: np.ndarray, boxes: list) -> list:
    if len(boxes) < 2:
        return boxes
    areas = np.array([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes], float)
    med = np.median(areas)
    out = []
    for (x0, y0, x1, y1), a in zip(boxes, areas):
        if a > 1.8 * med:
            sub = split_grid(mask[y0:y1, x0:x1])
            if len(sub) > 1:
                out += [(x0 + sx0, y0 + sy0, x0 + sx1, y0 + sy1)
                        for sx0, sy0, sx1, sy1 in sub]
                continue
        out.append((x0, y0, x1, y1))
    return out


def detect_panels(bgr: np.ndarray) -> list:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mask = foreground_mask(gray)
    boxes = split_components(mask)
    if len(boxes) < 2:
        grid = split_grid(mask)
        if len(grid) > len(boxes):
            boxes = grid
    else:
        boxes = _refine_oversized(mask, boxes)
    if not boxes:
        boxes = [(0, 0, w, h)]
    row_h = max(1, h // 12)
    return sorted(boxes, key=lambda b: (b[1] // row_h, b[0]))


def crop_sector(panel: np.ndarray) -> np.ndarray:
    h, w = panel.shape[:2]
    top = int(CFG["banner_frac"] * h)
    body = panel[top:, :]
    if body.size == 0:
        body = panel
    gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return body
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.10 * body.shape[0] * body.shape[1]:
        return body
    x, y, bw, bh = cv2.boundingRect(c)
    pad = 2
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(body.shape[1], x + bw + pad), min(body.shape[0], y + bh + pad)
    out = body[y0:y1, x0:x1]
    return out if out.size else body


def enhance(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CFG["clahe_clip"],
                            tileGridSize=(CFG["clahe_grid"], CFG["clahe_grid"]))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if CFG["speckle_filter"] == "median":
        out = cv2.medianBlur(out, 3)
    elif CFG["speckle_filter"] == "nlm":
        out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)
    return out


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.zeros((size, size, 3), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def is_informative(frame: np.ndarray) -> bool:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return not (g.std() < 9 or (g > 25).mean() < 0.06)


def decode_image(data: bytes) -> np.ndarray:
    """Bytes of a JPEG/PNG/etc. -> BGR uint8 array (same as cv2.imread)."""
    buf = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the image. Upload a JPG or PNG file.")
    return img


def sheet_to_frames(bgr: np.ndarray, max_frames: int = 12) -> tuple[list, list]:
    """Notebook 02, cell 12, for a single uploaded sheet.

    Returns (frames, boxes): 224x224 BGR frames and their source boxes.
    """
    # Very large phone photos make panel detection slow; the training sheets
    # were scans of a few thousand pixels at most.
    h, w = bgr.shape[:2]
    if max(h, w) > 4000:
        s = 4000 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    size = CFG["img_size"]
    frames, boxes = [], []
    for (x0, y0, x1, y1) in detect_panels(bgr):
        panel = bgr[y0:y1, x0:x1]
        if panel.size == 0:
            continue
        frame = letterbox(enhance(crop_sector(panel)), size)
        if not is_informative(frame):
            continue
        frames.append(frame)
        boxes.append((x0, y0, x1, y1))
        if len(frames) >= max_frames:
            break
    if not frames:
        frames = [letterbox(enhance(bgr), size)]
        boxes = [(0, 0, bgr.shape[1], bgr.shape[0])]
    return frames, boxes


def jpeg_roundtrip(frame: np.ndarray, quality: int = 94) -> np.ndarray:
    """Training frames were saved as JPEG (quality 94) and re-read before
    embedding. Doing the same here keeps the inputs bit-for-bit comparable."""
    ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else frame
