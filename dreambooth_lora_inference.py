from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler, DPMSolverMultistepScheduler
import torch
from diffusers.utils import load_image
device = torch.device('cuda:0')
controlnet_openpose = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_openpose')

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stablediffusionapi/cyberrealistic", controlnet=controlnet_openpose,
    safety_checker = None,
    requires_safety_checker = False
).to(device)

pipe.load_lora_weights('/root/dreambooth/monica_dreambooth_all_cr_checkpoints/checkpoint-10000/')

# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DDIMScheduler.from_pretrained("stablediffusionapi/cyberrealistic", subfolder="scheduler")

conditional_image = load_image('./smplx-a-pose_openpose_body_hand.png')
# conditional_image.save('./openpose_downsampled.png')

for i in range(100):
    image = pipe(prompt="A photo of <Monica Geller> wearing <Monica's dress> standing on the ground, white background, empty background, empty floor, empty ground", 
                negative_prompt="extra limbs, extra legs, more than 2 legs, more than 1 person, ugly, bad, unrealistic, cartoon, anime, naked, kitchen, room, furnitures, decorations", 
                num_inference_steps=50, 
                width=768,
                height=768,
                image=conditional_image).images[0]

    image.save(f'./generations/monica_{i}.png')
