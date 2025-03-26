import numpy as np
from PIL import Image, ImageDraw
import math
import random
import os
from tqdm import tqdm

def RandomBrush(
    max_tries,
    height,
    width,
    min_num_vertex=4,
    max_num_vertex=18,
    mean_angle=2*math.pi / 5,
    angle_range=2*math.pi / 15,
    min_width=12,
    max_width=48):
    average_radius = math.sqrt(height*height + width*width) / 8
    mask = Image.new('L', (width, height), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width_line = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width_line)
        for v in vertex:
            draw.ellipse((v[0] - width_line//2,
                          v[1] - width_line//2,
                          v[0] + width_line//2,
                          v[1] + width_line//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask

def RandomMask(height, width, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)
    while True:
        mask = np.ones((height, width), np.uint8)
        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, width - w + ww), np.random.randint(-hh, height - h + hh)
            mask[max(y, 0): min(y + h, height), max(x, 0): min(x + w, width)] = 0
        def MultiFill(max_tries, max_size):
            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)
        MultiFill(int(4 * coef), min(height, width) // 2)
        MultiFill(int(2 * coef), min(height, width))
        mask = np.logical_and(mask, 1 - RandomBrush(int(8 * coef), height, width))  # hole denoted as 0, reserved as 1
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return mask[np.newaxis, ...].astype(np.float32)

def dir_not_exists_then_create(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def generate_mask(output_mask_path, height=512, width=256, hole_range=[0.2, 0.4]):
    dir_name = '/'.join(output_mask_path.split('/')[:-1])
    dir_not_exists_then_create(dir_name)

    mask = RandomMask(height, width, hole_range=hole_range)
    mask = mask * 255
    mask = mask.repeat(3, axis=0)
    mask = mask.transpose(1, 2, 0)
    mask = 255 - mask
    img = Image.fromarray(np.uint8(mask))
    img.save(output_mask_path)
for i in tqdm(range(500)):
    generate_mask(output_mask_path=f'./random-masks/{i}.png', height=1024, width=512, hole_range=[0.2, 0.5])
