import torch
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import numpy as np
import time
def getModelSize(model):
	param_size = 0
	param_sum = 0
	for param in model.parameters():
		param_size += param.nelement() * param.element_size()
		param_sum += param.nelement()
	buffer_size = 0
	buffer_sum = 0
	for buffer in model.buffers():
		buffer_size += buffer.nelement() * buffer.element_size()
		buffer_sum += buffer.nelement()
	all_size = (param_size + buffer_size) / 1024 / 1024
	print('Model size: {:.3f}MB'.format(all_size))
	print(f'Sum of parameters: {param_sum}')
	return (param_size, param_sum, buffer_size, buffer_sum, all_size)

def run_edge_refinement_face():
    folder_name = r"D:\PythonProject\HumanMoE\src\deepfashion\cond_text_image_samples"
    seed = 99
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"cuda available: {torch.cuda.is_available()}")
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-inpainting",
        cache_dir=r"D:\PythonProject\UI\src\deepfashion\cond_text_image_samples\model_cache",
    ).to("cuda")
    def dummy(images, **kwargs):
        return images, [False]
    pipeline.safety_checker = dummy
    init_image = Image.open(f"{folder_name}/result.png")
    mask_image = Image.open(f"{folder_name}/face-edge.png")
    prompt = "best quality,"
    negative_prompt = "bad quality, blurry, artifacts"
    start_time = time.time()
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image, guidance_scale=3, strength=0.6, num_inference_steps=20).images[0]
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    mask_image = np.array(mask_image)
    mask_image[mask_image < 128] = 0
    mask_image[mask_image >= 128] = 1
    mask_image = np.stack([mask_image] * 3, axis=-1)
    image = np.array(image.resize((256, 512)))
    print(image.shape)
    image = init_image * (1 - mask_image) + image * mask_image
    image = Image.fromarray(image.astype(np.uint8))
    image.save(f"{folder_name}/result.png")
    getModelSize(pipeline.unet)
    getModelSize(pipeline.vae)
    getModelSize(pipeline.text_encoder)

if __name__ == "__main__":
    run_edge_refinement_face()