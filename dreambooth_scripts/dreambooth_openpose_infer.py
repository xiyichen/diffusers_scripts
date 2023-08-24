from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline, ControlNetModel, DDIMScheduler, DPMSolverMultistepScheduler
import torch
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector
from PIL import Image
import cv2, glob
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

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

controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_openpose', torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "./sergey_masked_dreambooth_2x_pp_cr_checkpoints_2000",
    controlnet=controlnet,
    safety_checker = None,
    requires_safety_checker = False,
    torch_dtype=torch.float16
).to(device)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

# open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
# img = Image.open('/root/amass_priorpose/rom4_projected/000001.png').convert("RGB")
# openpose_img = open_pose(img, detect_resolution=img.size, image_resolution=img.size, include_body=True, include_hand=True, include_face=True)
openpose_img_all = sorted(glob.glob('/root/amass_priorpose/all_poses/*.png'))
keypoints_all = sorted(glob.glob('/root/amass_priorpose/all_poses_npy/*.npy'))

for i in range(len(openpose_img_all)):
    openpose_img = Image.open(openpose_img_all[i]).convert("RGB")
    w, h = openpose_img.size
    keypoints_np = np.load(keypoints_all[i])
    labels_np = np.ones((1, keypoints_np.shape[0]))
    keypoints = torch.from_numpy(keypoints_np)[None].to(device)
    labels = torch.from_numpy(labels_np).to(device)
    got_good_generation = False
    
    while not got_good_generation:
        print(i)
        image = pipe(prompt="A photo of sks man wearing black t-shirt, black shorts, and black shoes, visible legs and shoes, standing on the floor in a white background, best quality, smooth light, no shadow",
                    negative_prompt="shadow, blurry, naked, cartoon, anime, missing limbs, missing feet, extra limbs, more than 1 person, more than 2 feet, more than 2 legs, more than 2 shoes", 
                    num_inference_steps=20, 
                    width=openpose_img.size[0],
                    height=openpose_img.size[1],
                    image=openpose_img,
                    guidance_scale=25,
                    # generator=torch.Generator(device="cpu").manual_seed(42),
                    ).images[0]

        # image.save(f'./sergey_amass/pose_{i}.png')
        # image = cv2.cvtColor(cv2.imread(f'./sergey_amass/pose_{i}.png'), cv2.COLOR_BGR2RGB)
        image_np = np.array(image)
        
        yolov8_boxes = yolov8_detection(yolo_model, image_np)
        x_min, y_min, x_max, y_max = yolov8_boxes[0]
        if len(yolov8_boxes) == 1 and y_max < h and x_max < w:
            got_good_generation = True
            image.save(f'./sergey_amass/pose_{i}.png')
            # input_boxes = torch.tensor(yolov8_boxes, device=predictor.device)
            
            # predictor.set_image(image)
            # masks, _, _ = predictor.predict_torch(
            #     # point_coords=keypoints,
            #     # point_labels=labels,
            #     point_coords=None,
            #     point_labels=None,
            #     boxes=input_boxes,
            #     multimask_output=False,
            # )
            
            # mask = masks[0][0].cpu().numpy().astype(np.int8)*255
            # for idx_, loc in enumerate(keypoints_np):
            #     x = int(loc[0])
            #     y = int(loc[1])
            #     cv2.circle(mask, (x, y), 10, (255, 0, 0), -1)
            # mask = cv2.rectangle(mask, [int(input_boxes[0][0]), int(input_boxes[0][1])], 
            #                      [int(input_boxes[0][2]), int(input_boxes[0][3])], (255, 0, 0), 2)
            # cv2.imwrite('./test_mask.png', mask)
            # exit()