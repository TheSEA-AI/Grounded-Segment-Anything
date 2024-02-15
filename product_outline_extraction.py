import os, sys

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "ControlNOLA"))

import argparse
import copy
from pathlib import Path

from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
#from transformers import pipeline

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor, build_sam_hq, build_sam_hq_vit_h, build_sam_hq_vit_b, build_sam_hq_vit_l
import cv2
import numpy as np
import matplotlib.pyplot as plt


## annotation
from ControlNOLA.annotator.hed import HEDdetector, nms
from annotator.util import HWC3, resize_image

# diffusers
import PIL
import requests
import torch
from io import BytesIO
from torchvision import transforms
import wget

from huggingface_hub import hf_hub_download
import locale
locale.getpreferredencoding = lambda: "UTF-8"

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of product position extraction based on Grouned SAM.")

    parser.add_argument("--input_dir", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="Path to the image.")

    parser.add_argument("--output_dir", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="Path to the image.")
    
    parser.add_argument("--img_format", 
                        default='png', 
                        type=str, 
                        help="Path to the image.")
    
    parser.add_argument("--product_type", 
                        default=None, 
                        type=str, 
                        required=False,
                        help="The type of the product.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args
    

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


# detect object using grounding DINO
def detect(image, image_source, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25):
  boxes, logits, phrases = predict(
      model=model,
      image=image,
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold
  )

  annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
  annotated_frame = annotated_frame[...,::-1] # BGR to RGB
  return annotated_frame, boxes

def segment(image, sam_model, boxes, device):
  sam_model.set_image(image)
  H, W, _ = image.shape
  boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

  transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
  masks, _, _ = sam_model.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,
      )
  return masks.cpu()


def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def get_product_position(mask):
  row_position = 0
  col_position = 0
  for row in range(0,mask.shape[0]):
    if any(x == False for x in mask[row,:]):
      row_position = row
      break

  for column in range(0,mask.shape[1]):
    if any(x == False for x in mask[:,column]):
      col_position = column
      break
  return row_position, col_position


### 
#inupt: 
#   path to image, 
#   prodction type
#output: 
#   row positon: the row where the very top pixel of the product is located
#   col positon: the column where the very left pixel of the product is located
###
def product_mask_extraction(img_path, product_type = "cosmetic product"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)

    sam_checkpoint_file = Path("./sam_hq_vit_h.pth")
    if not sam_checkpoint_file.is_file():
        sam_hq_vit_url = "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"
        wget.download(sam_hq_vit_url)

    sam_checkpoint = "sam_hq_vit_h.pth"
    sam_predictor = SamPredictor(build_sam_hq_vit_h(checkpoint=sam_checkpoint).to(device))

    image_source, image = load_image(img_path)
    _, detected_boxes = detect(image, image_source, text_prompt=product_type, model=groundingdino_model)
    mask_all = np.full((image_source.shape[1],image_source.shape[1]), True, dtype=bool)

    if detected_boxes.size(0) != 0:
        segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes, device=device)

        for mask in segmented_frame_masks:
            mask_all = mask_all & ~mask[0].cpu().numpy()
    else:
        raise ValueError("the product cannot be extracted.")

    mask_all = np.stack((mask_all,)*3, axis=-1)

    return mask_all

def product_outline_extraction(intput_dir, output_dir, img_format = '.png', product_type = "cosmetic product", image_resolution = 1024):

    Path(output_dir).mkdir(parents=True, exist_ok=True) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)

    sam_checkpoint_file = Path("./sam_hq_vit_h.pth")
    if not sam_checkpoint_file.is_file():
        sam_hq_vit_url = "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"
        wget.download(sam_hq_vit_url)

    sam_checkpoint = "sam_hq_vit_h.pth"
    sam_predictor = SamPredictor(build_sam_hq_vit_h(checkpoint=sam_checkpoint).to(device))

    image_filename_list = [i for i in os.listdir(intput_dir)]
    images_path = [os.path.join(intput_dir, file_path)
                        for file_path in image_filename_list]

    hedDetector = HEDdetector()
    for img_path, img_name in zip(images_path, image_filename_list):
        #mask = product_mask_extraction(img_path, product_type)
        #####################################
        #extract mask
        image_source, image = load_image(img_path)
        _, detected_boxes = detect(image, image_source, text_prompt=product_type, model=groundingdino_model)
        mask_all = np.full((image_source.shape[1],image_source.shape[1]), True, dtype=bool)

        if detected_boxes.size(0) != 0:
            segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes, device=device)

            for mask in segmented_frame_masks:
                mask_all = mask_all & ~mask[0].cpu().numpy()
        else:
            raise ValueError("the product cannot be extracted.")

        mask_all = np.stack((mask_all,)*3, axis=-1)
        ################

        mask_all = mask_all * np.uint8(255) 

        img = Image.open(img_path).convert("RGB")
        image_array = np.asarray(img)
        image_array = np.where(image_array == 0, 255, image_array)
        #print(f'image_array before = {image_array}')
        image_array = image_array * mask_all 
        image_array = np.where(image_array == 0, image_array, 255)
        #print(f'image_array after = {image_array}')
        
        hed = HWC3(image_array)
        hed = hedDetector(hed)
        hed = HWC3(hed)
        hed = cv2.resize(hed, (image_resolution, image_resolution),interpolation=cv2.INTER_LINEAR)
        img_masked = Image.fromarray(hed)
        img_save_path = output_dir + '/' + img_name
        img_masked.save(img_save_path, img_format)


if __name__ == "__main__":
    args = parse_args()
    product_outline_extraction(args.input_dir, args.output_dir, args.img_format)
    print(f'process finished.')
    #row_position, col_position = row_col_position(args.img_path, args.product_type)
    #print(f'row_position={row_position},col_position={col_position}')