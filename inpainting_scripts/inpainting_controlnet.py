from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


# control_image = make_inpaint_condition(init_image, mask_image)
id = '06'
original_image = load_image('./images/'+id+'_extended.png')
mask_image = load_image('./masks/'+id+'_extended.png')
inpaint_condition_image = make_inpaint_condition(original_image, mask_image)

openpose_condition_image = load_image('./poses/'+id+'_poses.png')
controlnet_inpaint = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
)

controlnet_openpose = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_openpose', 
                                                       torch_dtype=torch.float16)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", controlnet=controlnet_inpaint, torch_dtype=torch.float16
)

# speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None
pipe.requires_safety_checker = False
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

pipe.enable_model_cpu_offload()

# generate image
generated_img = pipe(
    prompt="RAW photo, (a person standing on the ground:1.2), no text, high detailed skin, empty background, 8k uhd, soft lighting, high quality",
    negative_prompt="(more than 1 person, extra person, more than 2 legs, missing legs, extra legs, extra feet:1.16), (extra long legs, extra long arms, occlusion, text, broken legs, broken limbs, deformed iris, deformed pupils, semi-realistic, extra people, extra limbs, extra arms, extra fingers, missing fingers, missing arms, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, mutated hands, poorly drawn hands, mutated feet, poorly drawn feet, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, cloned face, disfigured, gross proportions, malformed limbs, xfused fingers, too many fingers, long neck", 
    num_inference_steps=20,
    # generator=torch.Generator(device="cpu").manual_seed(1),
    guidance_scale=6.0,
    eta=1.0,
    image=original_image,
    mask_image=mask_image,
    control_image=inpaint_condition_image,
).images[0]

generated_img.save('./test_ehf.png')
