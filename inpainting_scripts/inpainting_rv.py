from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch
from diffusers import AutoencoderKL
import cv2
from PIL import Image
import glob

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
pipe = StableDiffusionInpaintPipeline.from_single_file(
    "https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/Realistic_Vision_V5.1-inpainting.safetensors", vae=vae, torch_dtype=torch.float16)

# speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

pipe.enable_model_cpu_offload()
pipe.safety_checker = None
pipe.requires_safety_checker = False


img_paths = sorted(glob.glob('/root/egobody/egobody_extracted_wo_blur/*.jpg'))
for img_path in img_paths:
    id = img_path.split('/')[-1].split('.')[0]
    original_image = cv2.imread('/root/egobody/egobody_extracted_wo_blur/'+id+'.jpg')
    original_image = original_image[...,::-1]
    h, w, _ = original_image.shape
    original_image_extended = np.zeros((2*h, w, 3))
    original_image_extended[:h, :w, :] = original_image
    mask_image = cv2.imread('/root/egobody/egobody_extracted_wo_blur_masks/'+id+'.png')
    mask_image_extended = np.zeros((2*h, w, 3))
    mask_image_extended[:h, :w, :] = mask_image
    # mask_image_extended[:int(h*0.95), :w, :] = 255
    mask_image_extended = (mask_image_extended//255).astype(bool)
    mask_image_extended = (~mask_image_extended).astype(np.int8)*255
    kernel = np.ones((10, 10), np.uint8)
    # mask_image_extended = cv2.erode(mask_image_extended, kernel, iterations=1)
    # cv2.imwrite('./original.png', mask_image_extended)
    # cv2.imwrite('./dilated.png', mask_image_extended_dilated)
    # print(mask_image_extended)
    # exit()
    original_image_extended = Image.fromarray(np.uint8(original_image_extended))
    mask_image_extended = Image.fromarray(np.uint8(mask_image_extended))
    # inpaint_condition_image = make_inpaint_condition(original_image_extended, mask_image_extended)

    # generate image
    generated_img = pipe(
        prompt="RAW photo, (a person standing on the ground, both legs are visible:1.8), no text, high detailed skin, empty background, 8k uhd, soft lighting, high quality",
        negative_prompt="(more than 1 person, extra person, naked, more than 2 legs, more than 2 shoes, missing legs, extra legs, extra feet, more than 2 arms, more than 2 hands, missing arms, extra arms, extra hands:1.16), (extra long legs, extra long arms, occlusion, text, broken legs, broken limbs, deformed iris, deformed pupils, semi-realistic, extra people, extra limbs, extra arms, extra fingers, missing fingers, missing arms, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, mutated hands, poorly drawn hands, mutated feet, poorly drawn feet, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, cloned face, disfigured, gross proportions, malformed limbs, xfused fingers, too many fingers, long neck", 
        num_inference_steps=20,
        generator=torch.Generator(device="cpu").manual_seed(42),
        guidance_scale=6.0,
        eta=1.0,
        height=1080,
        width=960,
        num_images_per_prompt=1,
        image=original_image_extended,
        mask_image=mask_image_extended
    ).images[0]
    # for (idx, img) in enumerate(generated_imgs):
    #     img.save('/root/egobody/egobody_empty_bg_generated/' + id + '_' + str(idx) + '.png')
    
    generated_img.save('/root/egobody/egobody_empty_bg_generated/' + id + '.png')
    
    # exit()
