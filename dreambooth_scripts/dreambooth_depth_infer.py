from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline, ControlNetModel, DDIMScheduler, DPMSolverMultistepScheduler
import torch
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector
from PIL import Image
import cv2
import numpy as np
device = torch.device('cuda:0')
controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1p_sd15_depth', torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "./sergey_masked_dreambooth_2x_pp_cr_checkpoints_2000",
    controlnet=controlnet,
    safety_checker = None,
    requires_safety_checker = False,
    torch_dtype=torch.float16
).to(device)

# open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
# img = Image.open('/root/amass_priorpose/rom4_projected/000000.png').convert("RGB")
# openpose_img = open_pose(img, detect_resolution=img.size, image_resolution=img.size, hand_and_face=True)
# openpose_img.save('./sergey_full_body_pose_openpose.png'
img = cv2.imread('/root/amass_priorpose/rom4_projected/000001.png')
img = Image.fromarray(np.uint8(img))
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

for i in range(100):
    image = pipe(prompt="A photo of sks man, white background, no shadow, best quality",
                # prompt="A photo of sks woman, only one person, full body is visible, visible face, visible legs, visible shoes, RAW photo, 8k uhd, best quality, photorealistic, empty background", 
                negative_prompt="shadow, blurry, naked, painting, cartoon, anime", 
                # negative_prompt="text, partially observed, truncated, blurry, upper body only, naked, extra person, extra limbs, extra legs, more than 2 legs, more than 1 person, ugly, bad, unrealistic, cartoon, anime", 
                num_inference_steps=20, 
                width=img.size[0],
                height=img.size[1],
                image=img,
                guidance_scale=20,
                # generator=torch.Generator(device="cpu").manual_seed(42),
                ).images[0]

    image.save(f'./monica_test.png')
    exit()