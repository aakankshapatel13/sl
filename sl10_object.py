
import torch
from PIL import Image
import requests
from io import BytesIO

print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load the small YOLOv5 model
print("Model Loaded!")


img_path = "C:\\Users\\aakan\\OneDrive\\Pictures\\Screenshots\\Screenshot 2025-04-20 134129.png"  
img = Image.open(img_path)


img.show()
results = model(img)
results.print() 
results.show()  
results.save()  


labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]


print("Detected Labels (Class IDs):", labels)
print("Detected Bounding Boxes and Confidence Scores:", cords)



LABELS = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", 
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


for label, cord in zip(labels, cords):
    class_name = LABELS[int(label)]
    print(f"Detected Object: {class_name}, Confidence: {cord[-2]:.4f}")
import torch
from PIL import Image
import requests
from io import BytesIO

print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load the small YOLOv5 model
print("Model Loaded!")


img_path = "C:\\Users\\aakan\\OneDrive\\Pictures\\Screenshots\\Screenshot 2025-04-20 134129.png"  
img = Image.open(img_path)


img.show()
results = model(img)
results.print() 
results.show()  
results.save()  


labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]


print("Detected Labels (Class IDs):", labels)
print("Detected Bounding Boxes and Confidence Scores:", cords)



LABELS = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", 
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


for label, cord in zip(labels, cords):
    class_name = LABELS[int(label)]
    print(f"Detected Object: {class_name}, Confidence: {cord[-2]:.4f}")