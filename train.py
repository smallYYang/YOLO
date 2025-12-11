# train.py
from ultralytics import YOLO

def main():
    # 使用预训练 YOLOv8s（建议）或 YOLOv8n（更小更快）
    model = YOLO("yolov8n.pt")     # 或改为 "yolov8n.pt"

    # 训练模型
    model.train(
        data="data.yaml",          # 指向你的 data.yaml
        epochs=50,                 # 训练轮数（可调）
        imgsz=640,                 # 输入图片大小
        batch=16,                  # batch size，不够显存则改为 8 或 4
        device=7,                  # GPU 编号，只有一个 GPU 就写 0
        workers=8,                 # 加载数据的线程数
        patience=20                # Early stopping
    )

    print("\n训练完成！最佳模型保存在：runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()
