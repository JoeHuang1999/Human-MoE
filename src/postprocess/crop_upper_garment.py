from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

def run_crop_upper_garment():
    image_folder_name = r"D:\PythonProject\HumanMoE\src\deepfashion\cond_text_image_samples"
    image = Image.open(f"{image_folder_name}/result.png")
    image = np.array(image)
    parsing = Image.open(f"{image_folder_name}/parsing.png").resize((256, 512))
    parsing = np.array(parsing)
    pose = Image.open(f"{image_folder_name}/pose.png").resize((256, 512))
    pose = np.array(pose)
    target_values = [1, 2, 4, 21, 23]
    image_background = Image.new('RGB', (512, 512), (255, 255, 255))
    parsing_background = Image.new('RGB', (512, 512), (255, 255, 255))
    pose_background = Image.new('RGB', (512, 512), (255, 255, 255))
    binary_mask = np.zeros_like(parsing, dtype=np.uint8)
    binary_mask[np.isin(parsing, target_values)] = 255
    binary_mask = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_x, center_y = 0, 0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x1, y1, w, h = cv2.boundingRect(largest_contour)
        if h <= 256:
            center_x = x1 + w // 2
            center_y = y1 + h // 2
            x2 = x1 + w
            y2 = y1 + h
            difference_width = 256 - w
            difference_height = 256 - h
            if difference_width % 2 == 1:
                x1 -= difference_width // 2
                if x1 < 0:
                    x2 += abs(x1)
                    x1 = 0
                x2 += difference_width // 2 + 1
                if x2 > parsing.shape[1]:
                    x1 -= abs(x2 - parsing.shape[1])
                    x2 = parsing.shape[1]
            else:
                x1 -= difference_width // 2
                if x1 < 0:
                    x2 += abs(x1)
                    x1 = 0
                x2 += difference_width // 2
                if x2 > parsing.shape[1]:
                    x1 -= abs(x2 - parsing.shape[1])
                    x2 = parsing.shape[1]

            if difference_height % 2 == 1:
                y1 -= difference_height // 2
                if y1 < 0:
                    y2 += abs(y1)
                    y1 = 0
                y2 += difference_height // 2 + 1
                if y2 > parsing.shape[0]:
                    y1 -= abs(y2 - parsing.shape[0])
                    y2 = parsing.shape[0]
            else:
                y1 -= difference_height // 2
                if y1 < 0:
                    y2 += abs(y1)
                    y1 = 0
                y2 += difference_height // 2
                if y2 > parsing.shape[0]:
                    y1 -= abs(y2 - parsing.shape[0])
                    y2 = parsing.shape[0]
            image = Image.fromarray(image[y1:y2, x1:x2]).resize((256, 256))
            image.save(f"{image_folder_name}/upper-garment-ori.png")
            parsing = Image.fromarray(parsing[y1:y2, x1:x2]).resize((256, 256))
            parsing.save(f"{image_folder_name}/upper-garment-parsing.png")
            pose = Image.fromarray(pose[y1:y2, x1:x2]).resize((256, 256))
            pose.save(f"{image_folder_name}/upper-garment-pose.png")

        else:
            x2 = x1 + w
            y2 = y1 + h
            scale = 256 / float(h)
            new_h = int(h * scale)
            new_w = int(w * scale)
            new_x1 = int(x1 * scale)
            new_y1 = int(y1 * scale)
            new_x2 = new_x1 + new_w
            new_y2 = new_y1 + new_h

            image = Image.fromarray(image[y1:y2, x1:x2]).resize((new_w, new_h))
            image_background.paste(image, ((256 - w) // 2, 0))
            image_background.save(f"{image_folder_name}/upper-ori.png")
            parsing = Image.fromarray(parsing[y1:y2, x1:x2]).resize((new_w, new_h))
            parsing_background.paste(parsing, ((256 - w) // 2, 0))
            parsing_background.save(f"{image_folder_name}/upper-parsing.png")
            pose = Image.fromarray(pose[y1:y2, x1:x2]).resize((new_w, new_h))
            pose_background.paste(pose, ((256 - w) // 2, 0))
            pose_background.save(f"{image_folder_name}/upper-pose.png")
        with open(f"{image_folder_name}/upper-garment-box.txt", "w") as f:
            f.write(f"{x1},{y1},{x2},{y2}")
    else:
        pass

if __name__ == '__main__':
    pass



