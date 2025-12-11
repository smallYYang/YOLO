import torch
torch.cuda.is_available()
torch.version.cuda
print("CUDA is available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
print("OK")