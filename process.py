"""
process_detailed_v2.py
Dock safety inspection with:
- YOLOv8: person / net / equipment(life vest)
- OpenCV: curb (danger zone) detection
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
    x1,y1,x2,y2 = box
    return ((x1+x2)/2, (y1+y2)/2)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)


def detect_curb(img):
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Sobel Y：提取水平纹理
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = np.abs(sobely)

    # 2. 仅统计下半区域（先验）
    y1, y2 = int(0.4 * H), int(0.9 * H)
    energy = np.sum(sobely[y1:y2, :], axis=0)

    # 3. 平滑能量曲线
    energy = cv2.GaussianBlur(energy.reshape(1,-1), (1,31), 0).flatten()

    # 4. 阈值选高能列
    thresh = np.mean(energy) + 1.5 * np.std(energy)
    xs = np.where(energy > thresh)[0]

    if len(xs) == 0:
        return None

    # 5. 合并连续列
    x_start, x_end = xs[0], xs[0]
    best = (x_start, x_end)
    max_len = 0

    for x in xs[1:]:
        if x == x_end + 1:
            x_end = x
        else:
            if x_end - x_start > max_len:
                best = (x_start, x_end)
                max_len = x_end - x_start
            x_start = x
            x_end = x
    if x_end - x_start > max_len:
        best = (x_start, x_end)

    bx1, bx2 = best

    # 6. 竖向 bbox（稍微扩展）
    pad = 5
    x1 = max(0, bx1 - pad)
    x2 = min(W, bx2 + pad)

    return (x1, y1, x2, y2)



# ===============================
# Equipment (life vest) logic
# ===============================
def person_has_lifevest(person_box, equipment_boxes):
    for e in equipment_boxes:
        if iou(person_box, e) > 0.2:
            return True
    return False

# ===============================
# Main processing
# ===============================
def process_image(model, img_path, out_dir):
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]

    result = model(img)[0]

    persons = []
    nets = []
    equipments = []

    for b in result.boxes:
        cls = int(b.cls[0])
        box = b.xyxy[0].cpu().numpy().tolist()
        if cls == 0:
            persons.append(box)
        elif cls == 1:
            nets.append(box)
        elif cls == 2:
            equipments.append(box)

    # ---- Detect curb ----
    curb = detect_curb(img)

    report = {
        "image": img_path.name,
        "detections": {
            "persons": len(persons),
            "nets": len(nets),
            "equipments": len(equipments),
            "curb_detected": curb is not None
        },
        "persons": [],
        "alarm": False,
        "alarm_reasons": []
    }

    vis = img.copy()

    # draw curb
    if curb:
        x1,y1,x2,y2 = map(int, curb)
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(vis,"CURB",(10,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    net_present = len(nets) > 0

    for i, p in enumerate(persons):
        px1,py1,px2,py2 = map(int,p)
        cx, cy = box_center(p)

        in_curb = False
        if curb:
            x1,y1,x2,y2 = curb
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                in_curb = True

        has_vest = person_has_lifevest(p, equipments)

        person_info = {
            "id": i,
            "in_curb": in_curb,
            "has_lifevest": has_vest,
            "safe_net_present": net_present
        }

        # ---- Alarm logic ----
        reasons = []
        if in_curb:
            if not net_present:
                reasons.append("in_curb_without_net")
            if not has_vest:
                reasons.append("in_curb_without_lifevest")

        if reasons:
            report["alarm"] = True
            report["alarm_reasons"].extend(reasons)
            person_info["alarm"] = True
            person_info["reasons"] = reasons
        else:
            person_info["alarm"] = False
            person_info["reasons"] = ["safe"]

        report["persons"].append(person_info)

        # draw person
        color = (0,255,0) if not reasons else (0,0,255)
        cv2.rectangle(vis,(px1,py1),(px2,py2),color,2)
        label = f"P{i} vest:{has_vest}"
        cv2.putText(vis,label,(px1,py1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    # draw nets
    for n in nets:
        x1,y1,x2,y2 = map(int,n)
        cv2.rectangle(vis,(x1,y1),(x2,y2),(255,255,0),2)
        cv2.putText(vis,"NET",(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)

    # save
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / img_path.name), vis)

    with open(out_dir / f"{img_path.stem}.json","w",encoding="utf-8") as f:
        json.dump(report,f,indent=2,ensure_ascii=False)

    return report

# ===============================
# CLI
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/test/images")
    ap.add_argument("--model", default="runs/detect/train/weights/best.pt")
    ap.add_argument("--out", default="out_v2")
    args = ap.parse_args()

    model = YOLO(args.model)
    src = Path(args.src)
    out = Path(args.out)

    for img in src.glob("*"):
        if img.suffix.lower() not in [".jpg",".png",".jpeg"]:
            continue
        report = process_image(model, img, out)
        if report["alarm"]:
            print(f"[ALARM] {img.name} -> {report['alarm_reasons']}")

if __name__ == "__main__":
    main()
