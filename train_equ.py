from ultralytics import YOLO

def main():
    # 使用更大的模型，对小目标更友好
    model = YOLO("yolov8m.pt")   # 显存够可用 yolov8x.pt

    model.train(
        data="data_equipment.yaml",

        # ---------- 核心参数 ----------
        epochs=100,                 # 小目标需要更长训练
        imgsz=960,                  # 关键：提高分辨率
        batch=8,                    # imgsz 大，batch 适当减
        device=6,
        workers=8,

        # ---------- 小目标友好 ----------
        mosaic=1.0,                 # 保留 mosaic
        close_mosaic=10,            # 后期关闭，提升精度
        mixup=0.1,

        # ---------- 颜色 & 外观增强 ----------
        hsv_h=0.05,
        hsv_s=0.7,
        hsv_v=0.4,

        # ---------- 学习率 ----------
        lr0=0.003,                  # 小一点更稳
        lrf=0.01,

        # ---------- 正则 ----------
        weight_decay=0.0005,
        patience=30,

        # ---------- 保存 ----------
        project="runs_equipment",
        name="yolov8m_equipment"
    )

    print("\n✅ Equipment 专用模型训练完成！")
    print("📌 best.pt 位于 runs_equipment/yolov8m_equipment/weights/best.pt")

if __name__ == "__main__":
    main()
