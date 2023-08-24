from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline, ControlNetModel, DDIMScheduler, DPMSolverMultistepScheduler
import torch
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector
from PIL import Image
import cv2
device = torch.device('cuda:0')
controlnet_openpose = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_openpose', torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "./monica_masked_dreambooth_2x_pp_cr_checkpoints_800",
    controlnet=controlnet_openpose,
    safety_checker = None,
    requires_safety_checker = False,
    torch_dtype=torch.float16
).to(device)

open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
img = Image.open('./monica_full_body_pose.png').convert("RGB")
openpose_img = open_pose(img, hand_and_face=True)
openpose_img.save('./monica_full_body_pose_openpose.png')
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
conditional_image = load_image('./control_human_openpose.png')

for i in range(100):
    image = pipe(prompt="A photo of sks woman, empty background, best quality, visible shoes",
                # prompt="A photo of sks woman, only one person, full body is visible, visible face, visible legs, visible shoes, RAW photo, 8k uhd, best quality, photorealistic, empty background", 
                negative_prompt="blurry, naked, ugly, bad, unrealistic, cartoon, anime", 
                # negative_prompt="text, partially observed, truncated, blurry, upper body only, naked, extra person, extra limbs, extra legs, more than 2 legs, more than 1 person, ugly, bad, unrealistic, cartoon, anime", 
                num_inference_steps=50, 
                width=512,
                height=1200,
                image=openpose_img,
                # guidance_scale=7,
                # generator=torch.Generator(device="cpu").manual_seed(42),
                ).images[0]

    image.save(f'./monica_test.png')
    exit()
