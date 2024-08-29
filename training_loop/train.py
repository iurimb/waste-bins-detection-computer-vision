from ultralytics import YOLO 
import cv2
from PIL import Image

epochs_list = [50]


model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
data=r'YOUR_DATA.YAML_FILE'

for i in range(len(epochs_list)):
    print(f"We'll be training for {epochs_list[i]} epochs")
   
    results = model.train(data=data, epochs=epochs_list[i])

    # Evaluate the model's performance on the validation set
results = model.val()
