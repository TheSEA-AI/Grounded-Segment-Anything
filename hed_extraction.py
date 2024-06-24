import os
import random
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.ImageOps import exif_transpose
import cv2
from ControlNOLA.annotator.hed import HEDdetector
from ControlNOLA.annotator.util import HWC3, resize_image
import argparse
import subprocess

#subprocess.call(["sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py"], shell=True)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of product position extraction based on Grouned SAM.")

    parser.add_argument("--input_dir", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="Path to the input image.")

    parser.add_argument("--data_clean_dir", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="Path to the clean data.")

    parser.add_argument("--output_dir", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="Path to output folder.")

    parser.add_argument("--start_index", 
                        default=None, 
                        type=int, 
                        required=True, 
                        help="image index to start with")
    
    parser.add_argument("--end_index", 
                        default=None, 
                        type=int, 
                        required=True, 
                        help="ending image index.")
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args


def main(input_dir, data_clean_dir, output_dir, start_index, end_index):
    #image_filename_list = [i for i in os.listdir(input_dir) if i.endswith('.png')]
    #image_paths = [os.path.join(input_dir, file_path)
    #                for file_path in image_filename_list]

    #num_images = len(image_paths)
    #print(f'num_images={num_images}')

    Path(output_dir).mkdir(parents=True, exist_ok=True) 

    image_resolution = 1024
    hedDetector = HEDdetector()

    num = 0
    #for img_name, img_path in zip(image_filename_list, image_paths):
    for index in range(start_index, end_index+1):
        img_name = 'f' + str(index)
        img_path = os.path.join(input_dir, img_name)
        file_size = os.path.getsize(img_path)
        print(f'img_path={img_path}, file_size={file_size}')
        num += 1
        if file_size > 30*1024:
            img = Image.open(img_path).convert("RGB")
            img.save(data_clean_dir+'/'+img_name, 'png')

            image_array = np.asarray(img)
            hed = HWC3(image_array)
            hed = hedDetector(hed)
            hed = HWC3(hed)
            hed = cv2.resize(hed, (image_resolution, image_resolution),interpolation=cv2.INTER_LINEAR)
            img_hed = Image.fromarray(hed)
            img_hed.save(output_dir+'/'+img_name, 'png')
            
        if num % 100 == 0:
            print(f'num={num}')
    

if __name__ == "__main__":
    args = parse_args()
    main(args.input_dir, args.data_clean_dir, args.output_dir, args.start_index, args.end_index)