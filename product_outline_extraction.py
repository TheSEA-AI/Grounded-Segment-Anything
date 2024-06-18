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
from scipy import ndimage

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
                        help="Path to data instance.")
    
    parser.add_argument("--data_hed_dir", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="Path to data hed.")

    parser.add_argument("--output_dir", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="Path to data hed background.")
    
    parser.add_argument("--img_format", 
                        default='png', 
                        type=str, 
                        help="Path to the image.")
    
    #parser.add_argument("--product_images",
    #                    nargs='+', 
    #                    default=[],
    #                    required=False,
    #                    help="The background image with the product")

    parser.add_argument("--product_images",
                    nargs='*', 
                    default=None,
                    required=False,
                    help="The background image with the product")

    parser.add_argument("--similarity_threshold", 
                        default=0.916, 
                        type=float, 
                        required=False,
                        help="The threshold to remove hed images")

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

def product_outline_extraction_by_mask(intput_dir, output_dir, img_format = 'png', product_type = "beauty product", image_resolution = 1024):

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
    kernel = np.ones((3, 3), np.uint8)
    image_dim = 1024
    for img_path, img_name in zip(images_path, image_filename_list):
        #####################################
        #extract mask
        image_source, image = load_image(img_path, image_dim)
        _, detected_boxes = detect(image, image_source, text_prompt=product_type, model=groundingdino_model)
        mask_all = np.full((image_source.shape[1],image_source.shape[1]), True, dtype=bool)
        
        if detected_boxes.size(0) != 0:
            segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes, device=device)

            for mask in segmented_frame_masks:
                im = np.stack((mask[0].cpu().numpy(),)*3, axis=-1)
                im = im.astype(np.uint8)*255
                imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(imgray, 127, 255, 0)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                print(f'img_name={img_name},len(contours)={len(contours)}')
                if len(contours) >= 20:
                    continue

                mask_all = mask_all & ~mask[0].cpu().numpy()
        else:
            raise ValueError("the product cannot be extracted.")

        mask_all = np.stack((mask_all,)*3, axis=-1)
        ################
    
        mask = ~mask_all
        mask = mask.astype(np.uint8)     
        mask = cv2.dilate(mask, kernel, iterations=3) 
        mask = np.array(mask, dtype=bool)

        img = Image.open(img_path).convert("RGB")
        img = img.resize((image_dim, image_dim), Image.LANCZOS)
        image_array = np.asarray(img)

        white_array = np.ones_like(image_array) * 180
        white_array = white_array * mask_all
        white_array = white_array * mask

        hed = HWC3(image_array)
        hed = hedDetector(hed) * mask_all[:,:,0]
        hed = hed*mask[:,:,0]
        hed = HWC3(hed)
        hed = np.where(hed<100, white_array, hed)
        hed = cv2.resize(hed, (image_resolution, image_resolution),interpolation=cv2.INTER_LINEAR)
        img_masked = Image.fromarray(hed)
        img_save_path = output_dir + '/' + img_name
        img_masked.save(img_save_path, img_format)

##the latest version with multiple product types and filling holes etc.
##the holes are becasue of SAM noise
def product_outline_extraction_by_mask_multiple_product_types(intput_dir, output_dir, img_format = 'png', image_resolution = 1024):

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
    kernel = np.ones((3, 3), np.uint8)
    image_dim = 1024
    for img_path, img_name in zip(images_path, image_filename_list):
        #####################################
        #extract mask

        image_source, image = load_image(img_path, image_dim)
        product_types = ["beauty product", "cosmetic product", "skincare product", "makeup product"]
        mask_all = np.full((image_source.shape[1],image_source.shape[1]), True, dtype=bool)
        for product_type in product_types:
            _, detected_boxes = detect(image, image_source, text_prompt=product_type, model=groundingdino_model)
            if detected_boxes.size(0) != 0:
                segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes, device=device)

                for mask in segmented_frame_masks:
                    im = np.stack((mask[0].cpu().numpy(),)*3, axis=-1)
                    im = im.astype(np.uint8)*255
                    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(imgray, 127, 255, 0)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) >= 50:
                        continue
                    mask_all = mask_all & ~mask[0].cpu().numpy()
            else:
                raise ValueError("the product cannot be extracted.")


        ##### fill holes inside product #######
        mask_all = ~mask_all
        mask_all = mask_all.astype(int)
        mask_all = ndimage.binary_fill_holes(mask_all).astype(int)
        mask_all = mask_all.astype(bool)
        mask_all = ~mask_all
        ##### fill holes inside product #######

        ##### fill small holes outside product #######
        ite = 8
        mask_all = mask_all.astype(int)
        mask_all = ndimage.binary_closing(mask_all,iterations=ite).astype(int)
        mask_all = mask_all.astype(bool)
        ##### fill small holes outside product #######
        
        ##### flip surrounding pixels due to previous fill small holes outside product #######
        mask_all[0:ite+2, :] = True
        mask_all[:, 0:ite+2] = True
        mask_all[1024-ite-1:1024, :] = True
        mask_all[:, 1024-ite-1:1024] = True
        ##### flip surrounding pixels due to previous fill small holes outside product #######

        mask_all = np.stack((mask_all,)*3, axis=-1)
        ################
    
        mask = ~mask_all
        mask = mask.astype(np.uint8)     
        mask = cv2.dilate(mask, kernel, iterations=3) 
        mask = np.array(mask, dtype=bool)

        img = Image.open(img_path).convert("RGB")
        img = img.resize((image_dim, image_dim), Image.LANCZOS)
        image_array = np.asarray(img)

        white_array = np.ones_like(image_array) * 180
        white_array = white_array * mask_all
        white_array = white_array * mask

        hed = HWC3(image_array)
        hed = hedDetector(hed) * mask_all[:,:,0]
        hed = hed*mask[:,:,0]
        hed = HWC3(hed)
        hed = np.where(hed<100, white_array, hed)
        hed = cv2.resize(hed, (image_resolution, image_resolution),interpolation=cv2.INTER_LINEAR)
        img_masked = Image.fromarray(hed)
        img_save_path = output_dir + '/' + img_name
        img_masked.save(img_save_path, img_format)

def filter_hed(product_images, data_hed_background_dir, data_similarity_dict, similarity_threshold):

    large_value = 100

    image_filename_list = [i for i in os.listdir(data_hed_background_dir)]
    images_path = [os.path.join(data_hed_background_dir, file_path)
                        for file_path in image_filename_list]

    ## make a copy of origial hed images
    image_dirs = data_hed_background_dir.split('/')
    new_image_dir = '/'+image_dirs[0]
    for i in range(1, len(image_dirs)-1):
        new_image_dir += image_dirs[i] + '/'
    new_image_dir += 'data_hed_background_original'
    Path(new_image_dir).mkdir(parents=True, exist_ok=True)
    #print(f'data_hed_background_dir={data_hed_background_dir}')
    #print(f'new_data_hed_background_dir={new_image_dir}')

    for img_name, img_path in zip(image_filename_list, images_path):
        img = Image.open(img_path).convert("RGB")
        img.save(new_image_dir+'/'+img_name, 'png')

    ## calculate similarities
    img_similarity_dict_all = {}
    for product_image in product_images:
        img1 = cv2.imread(os.path.join(data_hed_background_dir, product_image), cv2.IMREAD_GRAYSCALE)
        img1[img1 > 80] = 160
        img1[img1 <= 80] = 0
        ret1, thresh1 = cv2.threshold(img1, 127, 255,0)
        contours1,hierarchy1 = cv2.findContours(thresh1,2,1)
        cnt1 = contours1[0]

        img_similarity_dic = {}
        for img_name, img_path in zip(image_filename_list, images_path):
            if img_name not in product_images:
                img2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img2[img2 > 80] = 160
                img2[img2 <= 80] = 0
                ret2, thresh2 = cv2.threshold(img2, 127, 255,0)
                contours2,hierarchy2 = cv2.findContours(thresh2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours2) <=2:
                    cnt2 = contours2[0]
                    ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
                    if img_name not in img_similarity_dic:
                        img_similarity_dic[img_name] = ret
                    else:
                        if img_similarity_dic[img_name] > ret:
                            img_similarity_dic[img_name] = ret
                else:
                    img_similarity_dic[img_name] = large_value

        img_similarity_dict_all[product_image] = img_similarity_dic


    ##do filtering
    for img_name, img_path in zip(image_filename_list, images_path):
        if img_name not in product_images:
            remove = []

            for k, v in img_similarity_dict_all.items():
                data_simi_list = list(data_similarity_dict[k].values())
                for k_product, v_product in v.items():
                    if img_name == k_product:
                        if min(data_simi_list) <= 0.1:
                            v_product = v_product*10.0
                        if v_product >= similarity_threshold:
                            remove.append(True)
                
            if False not in remove:
                os.remove(img_path)

## the filtering for data is not enabled
## this is mainly for calculating data similarities to be used in hed filtering
def filter_data(product_images, hed_background_dir, hed_dir):

    print(f'product_images={product_images}')
    image_filename_list = [i for i in os.listdir(hed_dir)]
    images_path = [os.path.join(hed_dir, file_path)
                        for file_path in image_filename_list]

    ## make a copy of origial hed images
    image_dirs = hed_dir.split('/')
    new_image_dir = '/'+image_dirs[0]
    for i in range(1, len(image_dirs)-1):
        new_image_dir += image_dirs[i] + '/'
    new_image_dir += 'data_hed_original'
    Path(new_image_dir).mkdir(parents=True, exist_ok=True)

    for img_name, img_path in zip(image_filename_list, images_path):
        img = Image.open(img_path).convert("RGB")
        img.save(new_image_dir+'/'+img_name, 'png')

    ### calculate similarities
    img_similarity_dict_all = {}
    for product_image in product_images:
        img1 = cv2.imread(os.path.join(hed_background_dir, product_image), cv2.IMREAD_GRAYSCALE)
        img1[img1 > 80] = 160
        img1[img1 <= 80] = 0
        ret1, thresh1 = cv2.threshold(img1, 127, 255,0)
        contours1,hierarchy1 = cv2.findContours(thresh1,2,1)
        cnt1 = contours1[0]

        img_similarity_dic = {}
        for img_name, img_path in zip(image_filename_list, images_path):
            if img_name not in product_images:
                img2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img2[img2 > 80] = 160
                img2[img2 <= 80] = 0
                ret2, thresh2 = cv2.threshold(img2, 127, 255,0)
                contours2, hierarchy2 = cv2.findContours(thresh2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #print(f'img_name={img_name}, contours2 len={len(contours2)}')
                for cnt in contours2:
                    ret = cv2.matchShapes(cnt1,cnt,1,0.0)
                    if img_name not in img_similarity_dic:
                        img_similarity_dic[img_name] = ret
                    else:
                        if img_similarity_dic[img_name] > ret:
                            img_similarity_dic[img_name] = ret
        
        img_similarity_dict_all[product_image] = img_similarity_dic

    return img_similarity_dict_all

if __name__ == "__main__":
    args = parse_args()
    product_outline_extraction_by_mask_multiple_product_types(args.input_dir, args.output_dir, args.img_format)
    print(f'similarity={args.similarity_threshold}')
    if len(args.product_images) > 0:
       data_similarity_dict_all = filter_data(args.product_images, args.output_dir, args.data_hed_dir)
       filter_hed(args.product_images, args.output_dir, data_similarity_dict_all, args.similarity_threshold)
    print(f'process finished.')
    #row_position, col_position = row_col_position(args.img_path, args.product_type)
    #print(f'row_position={row_position},col_position={col_position}')