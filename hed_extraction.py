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
from annotator.util import HWC3, resize_image
import argparse
import subprocess

subprocess.call(["sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py"], shell=True)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of product position extraction based on Grouned SAM.")

    parser.add_argument("--input_dir", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="Path to the input image.")

    parser.add_argument("--output_dir", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="Path to output folder.")
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args


def main(input_dir, output_dir):
    image_filename_list = [i for i in os.listdir(input_dir) if i.endswith('.png')]
    image_paths = [os.path.join(input_dir, file_path)
                    for file_path in image_filename_list]

    num_images = len(image_paths)
    print(f'num_images={num_images}')

    Path(output_dir).mkdir(parents=True, exist_ok=True) 

    image_resolution = 1024
    hedDetector = HEDdetector()

    num = 0
    for img_name, img_path in zip(image_filename_list, image_paths):
        file_size = os.path.getsize(img_path)
        print(f'img_path={img_path}, file_size={file_size}')
        
        #if file_size < 10000000:
        #    os.remove(img_path)
        #    break

        img = Image.open(img_path).convert("RGB")
        image_array = np.asarray(img)
        hed = HWC3(image_array)
        hed = hedDetector(hed)
        hed = HWC3(hed)
        hed = cv2.resize(hed, (image_resolution, image_resolution),interpolation=cv2.INTER_LINEAR)
        img_hed = Image.fromarray(hed)
        img_hed.save(output_dir+'/'+img_name, 'png')
        num += 1
        if num % 100 == 0:
            print(f'num={num}')
        break

if __name__ == "__main__":
    args = parse_args()
    main(args.input_dir, args.output_dir)