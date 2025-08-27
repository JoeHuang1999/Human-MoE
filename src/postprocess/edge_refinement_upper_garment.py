import torch
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import numpy as np

def run_edge_refinement_face():
    folder_name = r"D:\PythonProject\HumanMoE\src\deepfashion\cond_text_image_samples"
    seed = 99
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"cuda available: {torch.cuda.is_available()}")
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-inpainting",
        cache_dir=r"D:\PythonProject\HumanMoE\src\deepfashion\cond_text_image_samples\model_cache",
    ).to("cuda")
    def dummy(images, **kwargs):
        return images, [False]
    pipeline.safety_checker = dummy
    init_image = Image.open(f"{folder_name}/result.png")
    mask_image = Image.open(f"{folder_name}/upper-garment-edge.png")
    prompt = "best quality,"
    negative_prompt = "bad quality, blurry, artifacts"
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image, guidance_scale=3, strength=0.6).images[0]
    mask_image = np.array(mask_image)
    mask_image[mask_image < 128] = 0
    mask_image[mask_image >= 128] = 1
    mask_image = np.stack([mask_image] * 3, axis=-1)
    image = np.array(image.resize((256, 512)))
    print(image.shape)
    image = init_image * (1 - mask_image) + image * mask_image
    image = Image.fromarray(image.astype(np.uint8))
    image.save(f"{folder_name}/result.png")

if __name__ == "__main__":
    run_edge_refinement_face()