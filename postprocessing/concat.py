'''
concatenate results horizontally or vertically
'''

import numpy as np
from glob import glob
import cv2
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import argparse

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
    
def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst
    
def concat(args):
    
    first_path = args.first_path
    second_path = args.second_path
    ref_path = args.ref_path
    outpath = args.out_path
    ext = args.ext

    if not os.path.exists(outpath):
        os.mkdir(outpath)
        
    for filename in sorted(glob(second_path + "/*." + ext)):
        print(filename)
        image_name = filename.replace(second_path,first_path)
        ref_name =  image_name.replace(first_path,ref_path)
        
        img = Image.open(image_name)
        overlay = Image.open(filename)
        ref_img = Image.open(ref_name)

        width , height = ref_img.size
        overlay = overlay.resize((width , height))
        
        if args.mode == "horizontal":
            img_mask = get_concat_h(img, overlay)
        elif args.mode == "vertical":
            img_mask = get_concat_v(img, overlay)
            
        out_name = filename.replace(second_path,outpath)
        img_mask.save(out_name)


if __name__ == "__main__":
    parser  =  argparse.ArgumentParser(description = "concatenate results with original images")

    # Dataset related settings
    parser.add_argument("--ref_path", default = "/projects/patho1/sparse_segmentation/dataset/split/test_set"
                        , help = "path to base image for up-sampling")
    parser.add_argument("--first_path", default = "/projects/patho1/sparse_segmentation/dataset/split/test_set"
                        , help = "path to first images")
    parser.add_argument("--second_path", default = "/projects/patho1/sparse_segmentation/results/merged/stage_1_2_DE_2_EP"
                        , help = "path to second images")
    parser.add_argument("--out_path", default = "/projects/patho1/sparse_segmentation/results/concat/stage_1_2_DE_2_EP"
                    , help="path to concat images")
    parser.add_argument('--ext', default= "png", help = "image extension without '.' ")

    parser.add_argument("--mode", default = "horizontal", choices = ["horizontal", "vertical"]
                    , help="which way of concating")
    
    args  =  parser.parse_args()
    concat(args)
