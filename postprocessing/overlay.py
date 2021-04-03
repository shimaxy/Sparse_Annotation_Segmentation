import numpy as np
from glob import glob
import cv2
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
import argparse

def overlay_fcn(im1,im2,over_color_list):
    overlay_img = im1.copy()
    
    for color in over_color_list:
        pixels_mask = np.all(im2 == color, axis=-1)
        overlay_img[pixels_mask] =  im2[pixels_mask] 

    return overlay_img
    
    
def overlay(args):
    #RGB color of labels
    colors =  {"UL" : (0, 0, 0),        #Unlabeled
               "BG" : (128, 128, 128),  #Background
               "COR" : (255, 0, 255),   #Corneum
               "EP" : (0, 0, 255),      #Epidermis
               "DE" : (255, 255, 0),    #Dermis
               "DMN" : (0, 255, 0),     #Dermal Nests
               "EPN" : (0, 85, 0)}      #Epidermal Nests

    path_1 = args.base_image_path
    path_2 = args.overlay_image_path
    out_path = args.out_path
    over_label = args.classes
    
    over_color_list = []
    for label in over_label:
        over_color_list.append(colors[label])
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
       
            
    for filename_2 in sorted(glob(path_2 + "/*.png")):  
        print(filename_2)
        filename_1 = filename_2.replace(path_2,path_1)

        im1 = cv2.imread(filename_1)
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
              
        im2 = cv2.imread(filename_2)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    
        overlay_img = overlay_fcn(im1,im2,over_color_list)
        overlay_name = filename_2.replace(path_2,out_path)
        plt.imsave(overlay_name,overlay_img)
            



if __name__ == "__main__":
    parser  =  argparse.ArgumentParser(description = "overlay stage 2 results" +
                                     "on stage 1 segmentation mask")

    # Dataset related settings
    parser.add_argument("--base_image_path", default = "/projects/patho1/sparse_segmentation/results/stage_1"
                        , help = "path to base image")
    parser.add_argument("--overlay_image_path", default = "/projects/patho1/sparse_segmentation/results/stage_2_DE"
                        , help = "image to be overlayed")
    parser.add_argument("--out_path", default = "/projects/patho1/sparse_segmentation/results/overlay/overlay_1_2_DE"
                    , help="overlayed output path")
    parser.add_argument('--classes', nargs='+', help='classes to be included', required=True)

    
    args  =  parser.parse_args()
    overlay(args)
