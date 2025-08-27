from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

def run_crop_hand(choose=0):
    image_folder_name = r"D:\PythonProject\HumanMoE\src\deepfashion\cond_text_image_samples"
    hand_model = YOLO(r"D:\PythonProject\HumanMoE\src\postprocess\hand_yolov9c.pt")
    image = Image.open(f"{image_folder_name}/result.png")
    image = np.array(image)
    parsing = Image.open(f"{image_folder_name}/parsing.png").resize((256, 512))
    parsing = np.array(parsing)
    pose = Image.open(f"{image_folder_name}/pose.png").resize((256, 512))
    pose = np.array(pose)
    hand_output = hand_model(image)
    hand_boxes = hand_output[0].boxes
    try:
        for box in hand_boxes:
            print(box.xyxy)
        x1, y1, x2, y2 = hand_boxes[choose].xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x11, y11, x22, y22 = x1, y1, x2, y2
        difference_width = 64 - (x2 - x1)
        difference_height = 64 - (y2 - y1)
        if difference_width % 2 == 1:
            x1 -= difference_width // 2
            if x1 < 0:
                x2 += abs(x1)
                x1 = 0
            x2 += difference_width // 2 + 1
            if x2 >= image.shape[1]:
                x1 -= abs(x2 - image.shape[1] + 1)
                x2 = image.shape[1] - 1
        else:
            x1 -= difference_width // 2
            if x1 < 0:
                x2 += abs(x1)
                x1 = 0
            x2 += difference_width // 2
            if x2 >= image.shape[1]:
                x1 -= abs(x2 - image.shape[1] + 1)
                x2 = image.shape[1] - 1

        if difference_height % 2 == 1:
            y1 -= difference_height // 2
            if y1 < 0:
                y2 += abs(y1)
                y1 = 0
            y2 += difference_height // 2 + 1
            if y2 >= image.shape[0]:
                y1 -= abs(y2 - image.shape[0] + 1)
                y2 = image.shape[0] - 1
        else:
            y1 -= difference_height // 2
            if y1 < 0:
                y2 += abs(y1)
                y1 = 0
            y2 += difference_height // 2
            if y2 >= image.shape[0]:
                y1 -= abs(y2 - image.shape[0] + 1)
                y2 = image.shape[0] - 1
        image = Image.fromarray(image[y1:y2, x1:x2]).resize((128, 128))
        image.save(f"{image_folder_name}/hand-ori.png")
        with open(f"{image_folder_name}/hand-box.txt", "w") as f:
            f.write(f"{x1},{y1},{x2},{y2}, {x11},{y11},{x22},{y22}")
        parsing = Image.fromarray(parsing[y1:y2, x1:x2]).resize((128, 128))
        parsing.save(f"{image_folder_name}/hand-parsing.png")
        pose = Image.fromarray(pose[y1:y2, x1:x2]).resize((128, 128))
        pose.save(f"{image_folder_name}/hand-pose.png")

    except:
        print(f"No hand detected!")
        pass
if __name__ == '__main__':
    run_crop_hand(choose=0)