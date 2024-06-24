import os
import random
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose
import cv2
import argparse


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
    #image_filename_list = [i for i in os.listdir(input_dir) if i.endswith('.png')]
    #image_paths = [os.path.join(input_dir, file_path)
    #                for file_path in image_filename_list]
    #num_images = len(image_paths)

    files = os.scandir(input_dir)

    info_file = output_dir + '/image_name_path.txt'
    num = 0
    with open(info_file, 'w') as f:
        #for img_name, img_path in zip(image_filename_list, image_paths):
        #file_size = os.path.getsize(img_path)
        #if file_size > 30*1024:
        #    f.write(f"{img_name}, {img_path}\n")

        for file in files:
            file_size = os.path.getsize(file.path)
            if file_size > 30*1024:
                num += 1
                f.write(f"{file.name}, {file.path}\n")
                if num % 100 == 0:
                    print(f'num={num}')

    print('process finshed.')

if __name__ == "__main__":
    args = parse_args()
    main(args.input_dir, args.output_dir)