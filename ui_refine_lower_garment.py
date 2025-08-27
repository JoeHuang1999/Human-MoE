from PIL import Image
import numpy as np
from src.my_utils.ldm_sample_folder_lower_garment import run_lower_garment
from src.postprocess.crop_lower_garment import run_crop_lower_garment
from src.postprocess.seamless_clone_lower_garment import run_seamless_clone_lower_garment
from src.postprocess.edge_refinement_lower_garment import run_edge_refinement_lower_garment
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os

def get_lower_garment_ssim():
    folder_name = r"D:\PythonProject\HumanMoE\src\deepfashion\cond_text_image_samples"
    image_gt = Image.open(f"{folder_name}/lower-garment-source.png").convert('L')
    image_generated = Image.open(f"{folder_name}/lower-garment-refined.png").convert('L')
    image_gt = np.array(image_gt)
    image_generated = np.array(image_generated)
    ssim_index, ssim_map = ssim(image_gt, image_generated, full=True)
    print(f'SSIM: {ssim_index}')
    return ssim_index

def read_lower_garment_box():
    file_name = r"D:\PythonProject\HumanMoE\src\deepfashion\cond_text_image_samples\lower-garment-box.txt"
    with open(file_name, "r") as file:
        # 讀取第一行並以逗號分割為整數列表
        line = file.readline()
        numbers = list(map(int, line.split(",")))
    return tuple(numbers)

def get_lower_garment_edge_gradient():
    folder_name = r"D:\PythonProject\HumanMoE\src\deepfashion\cond_text_image_samples"
    face_garment_list = [13, 14]
    hand_list = [15, 16, 17]
    upper_garment_list = [1, 2, 4, 21, 23]
    lower_garment_list = [3, 5, 6, 10]
    parsing = np.array(Image.open(f'{folder_name}/parsing.png').resize((256, 512)))
    parsing[np.isin(parsing, lower_garment_list)] = 255
    image = cv2.imread(f'{folder_name}/result.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(f'{folder_name}/lower-garment-mask.png', cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(mask, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel)
    # dilated_edges[parsing != 255] = 0
    image = image * (dilated_edges[:, :, np.newaxis] / 255)
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    image = Image.fromarray(image.astype('uint8')).convert('RGB')
    image.save(f'{folder_name}/gradient.png')
    edge_gradients = gradient_magnitude[dilated_edges > 0]
    if edge_gradients.size > 0:
        average_gradient = edge_gradients.mean()
    else:
        average_gradient = 0
    print(f'Average gradient: {average_gradient}')
    return average_gradient

def get_lower_garment_edge():
    folder_name = r"D:\PythonProject\UI\src\deepfashion\cond_text_image_samples"
    face_garment_list = [13, 14]
    hand_list = [15, 16, 17]
    upper_garment_list = [1, 2, 4, 21, 23]
    lower_garment_list = [3, 5, 6, 10]
    parsing = np.array(Image.open(f'{folder_name}/parsing.png').resize((256, 512)))
    parsing[np.isin(parsing, lower_garment_list)] = 255
    mask = cv2.imread(f'{folder_name}/lower-garment-mask.png', cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(mask, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_edges = cv2.dilate(edges, kernel)
    # dilated_edges[parsing != 255] = 0
    cv2.imwrite(f'{folder_name}/lower-garment-edge.png', dilated_edges)
    # cv2.imshow('Dilated Edges', dilated_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    folder_name = r"D:\PythonProject\HumanMoE\src\deepfashion\cond_text_image_samples"
    model = None
    vae = None
    gradient_list = []
    ssim_list = []
    run_crop_lower_garment()
    model, vae = run_lower_garment(model, vae)
    x1, y1, x2, y2 = read_lower_garment_box()
    for i in range(1, 7, 2):
        run_seamless_clone_lower_garment(crop=False, k=i)
        gradient_list.append(get_lower_garment_edge_gradient())
        image = Image.open(f'{folder_name}/result.png').save(f'{folder_name}/result-{len(gradient_list)}.png')
        mask = Image.open(f'{folder_name}/lower-garment-mask.png').save(f'{folder_name}/lower-garment-mask-{len(gradient_list)}.png')
        image = Image.open(f'{folder_name}/result.png').crop((x1, y1, x2, y2)).resize((256, 256)).save(f'{folder_name}/lower-garment-refined.png')
        ssim_list.append(get_lower_garment_ssim())

    run_seamless_clone_lower_garment(crop=True, k=1)
    gradient = get_lower_garment_edge_gradient()
    image = Image.open(f'{folder_name}/result.png').crop((x1, y1, x2, y2)).resize((256, 256)).save(f'{folder_name}/face-refined.png')
    ssim = get_lower_garment_ssim()

    best_index = ssim_list.index(max(ssim_list))
    if ( ssim - max(ssim_list) < 0.03 and gradient_list[best_index] < 350 ):
        image = Image.open(f'{folder_name}/result-{best_index}.png').save(f'{folder_name}/result.png')
        mask = Image.open(f'{folder_name}/lower-garment-mask-{best_index}.png').save(f'{folder_name}/lower-garment-mask.png')
        image = Image.open(f'{folder_name}/result.png').crop((x1, y1, x2, y2)).resize((256, 256)).save(f'{folder_name}/lower-garment-refined.png')
    else:
        pass
    # get_lower_garment_edge()
    # run_edge_refinement_lower_garment()


