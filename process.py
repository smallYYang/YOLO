"""
process_detailed_v2.py
Dock safety inspection with:
- YOLOv8: person / net / equipment(life vest)
- OpenCV: curb (danger zone) detection
- Rule-based orange life vest fallback
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
import argparse

# ===============================
# Geometry helpers
# ===============================
def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

# ===============================
# Hue entropy (for curb)
# ===============================
def hue_entropy(bgr_crop, bins=30):
    if bgr_crop is None or bgr_crop.size == 0:
        return 1e6
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    hist = cv2.calcHist([h], [0], None, [bins], [0, 180])
    hist = hist / (hist.sum() + 1e-6)
    return float(-np.sum(hist * np.log(hist + 1e-6)))

# ===============================
# Curb detection (unchanged)
# ===============================
def detect_curb(img):
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    left_bound = int(0.1 * W)
    right_bound = int(0.9 * W)
    y1, y2 = int(0.4 * H), int(0.9 * H)

    roi = gray[y1:y2, left_bound:right_bound]
    bgr_roi = img[y1:y2, left_bound:right_bound]

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (50, 40, 40), (65, 255, 255))
    if blue_mask.sum() / (blue_mask.size + 1e-6) > 0.1:
        return None

    sobely = np.abs(cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3))
    energy = np.std(sobely, axis=0)

    thresh = np.mean(sobely) + 0.5 * np.std(sobely)
    binary = sobely > thresh
    stripe_count = np.sum(np.abs(np.diff(binary.astype(np.int8), axis=0)), axis=0)

    score = energy * stripe_count
    if len(score) < 10:
        return None

    score = cv2.GaussianBlur(score.reshape(1, -1), (1, 31), 0).flatten()
    xs = np.where(score > np.mean(score) + 1.2 * np.std(score))[0]
    if len(xs) == 0:
        return None

    segments = []
    start = xs[0]
    for i in range(1, len(xs)):
        if xs[i] != xs[i-1] + 1:
            segments.append((start, xs[i-1]))
            start = xs[i]
    segments.append((start, xs[-1]))
    segments.sort(key=lambda s: s[1] - s[0], reverse=True)

    fallback = None
    for bx1, bx2 in segments:
        x1 = max(0, bx1 + left_bound - 5)
        x2 = min(W, bx2 + left_bound + 5)
        crop = img[y1:y2, x1:x2]
        if fallback is None:
            fallback = (x1, y1, x2, y2)
        if hue_entropy(crop) < 2.5:
            return (x1, y1, x2, y2)
    return fallback

# ===============================
# Life vest detection
# ===============================
def person_has_lifevest_yolo(person_box, equipment_boxes):
    for e in equipment_boxes:
        if iou(person_box, e) > 0.01:
            return True
    return False

def detect_orange_lifevest(img, person_box,
                           min_area_ratio=0.03):
    x1, y1, x2, y2 = map(int, person_box)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 宽 HSV 门限：橙 + 红橙
    mask1 = cv2.inRange(hsv, (5, 80, 80), (18, 255, 255))
    mask2 = cv2.inRange(hsv, (18, 80, 80), (35, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)

    # 轻量去噪
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8), iterations=1)

    orange_area = np.count_nonzero(mask)
    person_area = (x2 - x1) * (y2 - y1)
    return orange_area > min_area_ratio * person_area

# ===============================
# Main processing
# ===============================
def process_image(model, img_path, out_dir):
    img = cv2.imread(str(img_path))
    result = model(img)[0]

    persons, nets, equipments = [], [], []
    for b in result.boxes:
        cls = int(b.cls[0])
        box = b.xyxy[0].cpu().numpy().tolist()
        if cls == 0:
            persons.append(box)
        elif cls == 1:
            nets.append(box)
        elif cls == 2:
            equipments.append(box)

    curb = detect_curb(img)

    report = {
        "image": img_path.name,
        "persons": [],
        "alarm": False,
        "alarm_reasons": []
    }

    vis = img.copy()
    if curb:
        x1, y1, x2, y2 = map(int, curb)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    net_present = len(nets) > 0

    for i, p in enumerate(persons):
        cx, cy = box_center(p)
        in_curb = curb and curb[0] <= cx <= curb[2] and curb[1] <= cy <= curb[3]

        # --- Life vest detection ---
        if person_has_lifevest_yolo(p, equipments):
            vest_present = True
            vest_source = "yolo"
        elif detect_orange_lifevest(img, p):
            vest_present = True
            vest_source = "color"
        else:
            vest_present = False
            vest_source = "none"

        reasons = []
        if in_curb:
            if not net_present:
                reasons.append("in_curb_without_net")
            if not vest_present:
                reasons.append("in_curb_without_lifevest")

        if reasons:
            report["alarm"] = True
            report["alarm_reasons"].extend(reasons)

        report["persons"].append({
            "id": i,
            "in_curb": bool(in_curb),
            "lifevest": {
                "present": vest_present,
                "source": vest_source
            },
            "safe_net_present": net_present,
            "alarm": bool(reasons),
            "reasons": reasons if reasons else ["safe"]
        })

        color = (0, 255, 0) if not reasons else (0, 0, 255)
        x1, y1, x2, y2 = map(int, p)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"P{i} vest:{vest_present}({vest_source})",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / img_path.name), vis)
    with open(out_dir / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report

# ===============================
# CLI
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/test/images")
    ap.add_argument("--model", default="runs/detect/train2/weights/best.pt")
    ap.add_argument("--out", default="out_v2")
    args = ap.parse_args()

    model = YOLO(args.model)
    for img in Path(args.src).glob("*"):
        if img.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            r = process_image(model, img, Path(args.out))
            if r["alarm"]:
                print(f"[ALARM] {img.name} -> {r['alarm_reasons']}")

if __name__ == "__main__":
    main()
