from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline, ControlNetModel, DDIMScheduler, DPMSolverMultistepScheduler
import torch
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector
from PIL import Image
import cv2, glob
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
from torchmetrics.functional.multimodal import clip_score
from functools import partial

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    images_int = (images).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

def yolov8_detection(model, image):
    results = model(image, stream=True)  # generator of Results objects

    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
    
    bbox_all = boxes.xyxy.tolist()

    bbox = []
    for box, cl, conf in zip(bbox_all, boxes.cls.tolist(), boxes.conf.tolist()):
      if cl == 0 and conf >= 0:
        width = box[2] - box[0]
        height = box[3] - box[1]
        # print(conf)
        bbox.append([box[0]-30, box[1]-30, box[2]+30, box[3]+30])
    return bbox

device = torch.device('cuda:0')
model_type = "vit_h"
sam_checkpoint = "/root/dreambooth/data/sam_vit_h_4b8939.pth"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
yolo_model=YOLO('/root/dreambooth/data/yolov8x.pt')

controlnet_depth = ControlNetModel.from_pretrained('lllyasviel/control_v11f1p_sd15_depth', 
                                                      torch_dtype=torch.float16)
controlnet_openpose = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_openpose', 
                                                       torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "./sergey_checkpoints/sergey_masked_dreambooth_2x_pp_cr_checkpoints_2000",
    controlnet=[controlnet_depth, controlnet_openpose],
    safety_checker = None,
    requires_safety_checker = False,
    torch_dtype=torch.float16
).to(device)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
openpose_img_all = sorted(glob.glob('/root/amass_priorpose/all_poses/*.png'))
depth_img_all = sorted(glob.glob('/root/amass_priorpose/all_pose_smpl_vis/*.png'))
keypoints_all = sorted(glob.glob('/root/amass_priorpose/all_poses_npy/*.npy'))

for i in range(len(openpose_img_all)):
    openpose_img = Image.open(openpose_img_all[i]).convert("RGB")
    depth_img = Image.open(depth_img_all[i]).convert("RGB")
    assert openpose_img_all[i].split('_')[-1].split('.')[0] == depth_img_all[i].split('_')[-1].split('.')[0]
    w, h = openpose_img.size
    got_good_generation = False
    
    while not got_good_generation:
        print(i)
        image = pipe(prompt="A photo of sks man wearing black t-shirt, black shorts, and black shoes, visible shoes, standing on the floor in a white background, best quality, empty scene, no shadow",
                    negative_prompt="dark, shadow, blurry, naked, cartoon, anime, missing limbs, missing feet, extra limbs, more than 1 person, more than 2 feet, more than 2 legs, more than 2 shoes", 
                    num_inference_steps=20, 
                    width=openpose_img.size[0],
                    height=openpose_img.size[1],
                    image=[depth_img, openpose_img],
                    controlnet_conditioning_scale=(0.25, 0.75),
                    guidance_scale=7.5,
                    # generator=torch.Generator(device="cpu").manual_seed(42),
                    ).images[0]

        image_np = np.array(image)[:, :, ::-1]
        
        predictor.set_image(image_np)
        yolov8_boxes = yolov8_detection(yolo_model, image_np)
        if len(yolov8_boxes) != 1:
            continue
        input_boxes = torch.tensor(yolov8_boxes, device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image_np.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        mask = (masks[0].cpu().numpy().astype(np.int8)*255)[0]
        image_np_masked = image_np
        image_np_masked[mask==0] = 255
        
        clip_prompts = ["A photo of sks man wearing black t-shirt, black shorts, and black shoes"]
        clip_images = image_np_masked[None]
        sd_clip_score = calculate_clip_score(clip_images, clip_prompts)
        
        x_min, y_min, x_max, y_max = yolov8_boxes[0]
        if sd_clip_score >= 28 and y_max < h and x_max < w:
            got_good_generation = True
            image.save(f'./sergey_results/sergey_amass_multicontrolnet_0.25_0.75_guidance_7.5_clip_26/pose_{i}.png')
            cv2.imwrite(f'./sergey_results/sergey_amass_multicontrolnet_0.25_0.75_guidance_7.5_clip_26_masks/pose_{i}.png', mask)