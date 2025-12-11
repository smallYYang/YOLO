# process_detailed.py
"""
Detailed processing for dock safety task.
Outputs per-image JSON report, a CSV summary, visualization and alarm images.
Usage:
    python process_detailed.py --src data/test/images --model runs/detect/train/weights/best.pt
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import csv
import os
import time

# ------------------------
# Default config (can be overridden by roi_config.json)
# ------------------------
DEFAULT_CFG = {
    "roi": [],  # optional polygon [[x,y],...], pixel coords
    "use_normalized_roi": False,
    "conf_thresh": 0.25,
    "orange_lower_hsv": [5, 120, 150],
    "orange_upper_hsv": [18, 255, 255],
    "orange_ratio_threshold": 0.05,
    "upper_body_ratio": 0.45
}

# ------------------------
# Helpers
# ------------------------
def load_cfg(path=None):
    cfg = DEFAULT_CFG.copy()
    if path:
        p = Path(path)
        if p.exists():
            try:
                j = json.loads(p.read_text())
                cfg.update(j)
                print(f"Loaded config from {path}")
            except Exception as e:
                print("Failed to read cfg, using defaults:", e)
    return cfg

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def xyxy_to_int(box):
    return [int(round(x)) for x in box]

def box_center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2, (y1+y2)/2)

def point_in_polygon(pt, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), (int(pt[0]), int(pt[1])), False) >= 0

def upper_body_crop(img, box, ratio=0.45):
    x1,y1,x2,y2 = [int(round(v)) for v in box]
    h = y2 - y1
    if h <= 2: return None
    new_y2 = y1 + max(1, int(h * ratio))
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1]-1, x2); new_y2 = min(img.shape[0]-1, new_y2)
    if x2 <= x1 or new_y2 <= y1: return None
    return img[y1:new_y2, x1:x2]

def is_wearing_vest(crop, lower_hsv, upper_hsv, ratio_thr):
    if crop is None or crop.size == 0:
        return {"wearing": False, "orange_ratio": 0.0}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv, dtype=np.uint8), np.array(upper_hsv, dtype=np.uint8))
    orange_ratio = float(np.count_nonzero(mask) / mask.size)
    return {"wearing": orange_ratio >= ratio_thr, "orange_ratio": orange_ratio}

def draw_label(img, text, org, color=(0,255,0), scale=0.5, thickness=1, bg=True):
    x,y = int(org[0]), int(org[1])
    (w,h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    if bg:
        cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y), (0,0,0), -1)
    cv2.putText(img, text, (x+2, y-2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# ------------------------
# Main per-image logic
# ------------------------
def analyze_image(model, img_path: Path, cfg, out_dirs):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    H, W = img.shape[:2]

    # run model
    res = model(img, conf=cfg["conf_thresh"])[0]

    detections = []  # list of dict {cls, cls_name, conf, xyxy}
    persons = []; nets = []; equipments = []

    if hasattr(res, "boxes") and len(res.boxes) > 0:
        for b in res.boxes:
            # safe extraction supporting both tensor and numpy
            try:
                xy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], "cpu") else np.array(b.xyxy[0])
                conf = float(b.conf[0].cpu().numpy()) if hasattr(b.conf[0], "cpu") else float(b.conf[0])
                cls = int(b.cls[0].cpu().numpy()) if hasattr(b.cls[0], "cpu") else int(b.cls[0])
            except Exception:
                # fallback: try attribute directly
                xy = np.array(b.xyxy[0])
                conf = float(b.conf[0])
                cls = int(b.cls[0])
            x1,y1,x2,y2 = xy.tolist()
            d = {"cls": cls, "conf": conf, "xyxy": [x1,y1,x2,y2]}
            detections.append(d)
            if cls == 0: persons.append(d)
            elif cls == 1: nets.append(d)
            elif cls == 2: equipments.append(d)

    # build report skeleton
    report = {
        "image": img_path.name,
        "size": [W,H],
        "detections": detections,
        "summary_reasons": [],
        "alarms": [],
        "per_person": []
    }

    # default flags
    net_present = len(nets) > 0
    equip_present = len(equipments) > 0
    persons_present = len(persons) > 0

    # reasons for no alarm (or for alarm)
    if not persons_present:
        report["summary_reasons"].append("no_persons_detected")
    if not equip_present:
        report["summary_reasons"].append("no_equipment_detected")
    if not net_present:
        report["summary_reasons"].append("no_net_detected")

    # prepare visualization canvas
    vis = img.copy()
    # draw all detections
    for d in detections:
        cls = d["cls"]; conf = d["conf"]; xy = d["xyxy"]
        x1,y1,x2,y2 = xy_to_int = xyxy_to_int(xy)
        if cls == 0:
            color=(0,200,0)
            name="person"
        elif cls == 1:
            color=(0,200,200)
            name="net"
        elif cls == 2:
            color=(200,0,0)
            name="equipment"
        else:
            color=(100,100,100); name=f"cls{cls}"
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        draw_label(vis, f"{name} {conf:.2f}", (x1, y1), color=color)

    # if config ROI exists and no equipment detected, draw ROI
    if (not equip_present) and cfg.get("roi"):
        roi = cfg["roi"]
        if cfg.get("use_normalized_roi", False):
            roi_px = [[int(x * W), int(y * H)] for x,y in roi]
        else:
            roi_px = roi
        cv2.polylines(vis, [np.array(roi_px, np.int32)], isClosed=True, color=(0,128,255), thickness=2)
        report["summary_reasons"].append("used_config_roi" if not equip_present else "roi_present")

    # analyze each detected person
    for p_idx, p in enumerate(persons):
        xy = p["xyxy"]
        x1,y1,x2,y2 = [int(round(v)) for v in xy]
        cx, cy = box_center([x1,y1,x2,y2])
        person_entry = {
            "bbox": [x1,y1,x2,y2],
            "conf": p["conf"],
            "center": [cx, cy]
        }

        # check in equipment if any equipment present
        in_equipment = False
        if equip_present:
            for eq in equipments:
                ex1,ey1,ex2,ey2 = [int(round(v)) for v in eq["xyxy"]]
                if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
                    in_equipment = True
                    break
        else:
            # fallback to ROI if provided
            if cfg.get("roi"):
                if cfg.get("use_normalized_roi", False):
                    roi_px = [[int(x * W), int(y * H)] for x,y in cfg["roi"]]
                else:
                    roi_px = cfg["roi"]
                in_equipment = point_in_polygon((cx,cy), roi_px)

        person_entry["in_equipment"] = bool(in_equipment)

        # check vest on upper body crop
        crop = upper_body_crop(img, [x1,y1,x2,y2], ratio=cfg.get("upper_body_ratio",0.45))
        vest_info = is_wearing_vest(crop, cfg["orange_lower_hsv"], cfg["orange_upper_hsv"], cfg["orange_ratio_threshold"])
        person_entry.update(vest_info)

        # form reasons per person
        reasons = []
        if not in_equipment:
            reasons.append("person_not_in_equipment")
        else:
            if not net_present:
                reasons.append("person_in_equipment_but_no_net")
            if not vest_info["wearing"]:
                reasons.append("person_in_equipment_but_no_vest")
            if vest_info["wearing"] and net_present:
                reasons.append("person_in_equipment_and_safe")

        person_entry["reasons"] = reasons
        report["per_person"].append(person_entry)

        # draw annotation for this person
        color = (0,255,0) if vest_info["wearing"] else (0,0,255)
        cv2.rectangle(vis, (x1,y1),(x2,y2), color, 2)
        draw_label(vis, f"vest:{vest_info['orange_ratio']:.3f}", (x1, max(0,y1-12)), color=color)

        # if alarm reason exists for this person -> collect
        if any(r.startswith("person_in_equipment") and (r != "person_in_equipment_and_safe") for r in reasons):
            report["alarms"].append({"person_index": p_idx, "reasons": reasons})

    # global summary alarm: if any per-person alarm exists -> alarm true
    alarm = len(report["alarms"]) > 0
    if alarm:
        report["summary_reasons"].extend([r for a in report["alarms"] for r in a["reasons"]])
    report["alarm_flag"] = bool(alarm)

    # save outputs
    out_vis = out_dirs["vis"] / img_path.name
    out_report = out_dirs["reports"] / (img_path.stem + ".json")
    out_csv_row = {
        "image": img_path.name,
        "alarm": int(alarm),
        "num_persons": len(persons),
        "num_nets": len(nets),
        "num_equipments": len(equipments),
        "summary_reasons": ";".join(report["summary_reasons"])
    }

    cv2.imwrite(str(out_vis), vis)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # if alarm -> also save alarm image
    if alarm:
        out_alarm = out_dirs["alarms"] / img_path.name
        cv2.imwrite(str(out_alarm), vis)

    return out_csv_row, report

# ------------------------
# CLI & batch processing
# ------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, default="data/test/images", help="image folder")
    p.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt", help="model path")
    p.add_argument("--cfg", type=str, default="roi_config.json", help="optional cfg json")
    p.add_argument("--out", type=str, default="out", help="output base dir")
    p.add_argument("--conf", type=float, default=None, help="override conf thresh")
    args = p.parse_args()

    cfg = load_cfg(args.cfg)
    if args.conf is not None:
        cfg["conf_thresh"] = args.conf

    model = YOLO(args.model)

    src_dir = Path(args.src)
    out_base = Path(args.out)
    out_dirs = {
        "vis": out_base / "vis",
        "alarms": out_base / "alarms",
        "reports": out_base / "reports"
    }
    for d in out_dirs.values():
        ensure_dir(d)

    csv_path = out_base / "summary.csv"
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(["image","alarm","num_persons","num_nets","num_equipments","summary_reasons"])

    img_paths = sorted([p for p in src_dir.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    start = time.time()
    for img_p in img_paths:
        row, report = analyze_image(model, img_p, cfg, out_dirs)
        if row is None:
            continue
        csv_writer.writerow([row["image"], row["alarm"], row["num_persons"], row["num_nets"], row["num_equipments"], row["summary_reasons"]])
        if row["alarm"]:
            print(f"[ALARM] {row['image']} reasons: {report['summary_reasons']}")
    csv_f.close()
    elapsed = time.time() - start
    print(f"Done. Processed {len(img_paths)} images in {elapsed:.1f}s. Outputs in {out_base}")

if __name__ == "__main__":
    main()
