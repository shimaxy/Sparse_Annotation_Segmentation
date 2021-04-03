'''
Removing dermis for stage 2-Epidermis
and epidermis for stage 2-Dermis
Using dermis and epidermis mask

the option of extracting instead of removing is also available
'''

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
from glob import glob
import os
import argparse

def apply_mask(img,label, color,mode):
    mask_int = np.zeros(img.shape)
  
    if mode == "include":
        mask = np.all(label==color,axis=2)
    elif mode == "exclude":
        mask = np.all(label!=color,axis=2)

    mask = mask.astype(int)

    mask_int[:,:,0] = mask
    mask_int[:,:,1] = mask
    mask_int[:,:,2] = mask
    
    im_masked = img * (mask_int/255)
    im_masked = im_masked*255
 
    return im_masked



def conversion(args):
    image_path = args.image_path
    mask_path = args.mask_path
    out_path_DE = args.out_path_DE
    out_path_EP = args.out_path_EP
    ext = args.ext

    if not os.path.exists(out_path_DE):
        os.mkdir(out_path_DE)
    if not os.path.exists(out_path_EP):
        os.mkdir(out_path_EP)
    
    for filename in sorted(glob(mask_path + "/*." + ext)):
        stage_2_DE = filename.replace(mask_path,out_path_DE)
        stage_2_EP = filename.replace(mask_path,out_path_EP)
        image_name = filename.replace(mask_path,image_path)
        print(image_name)
        
        img = Image.open(image_name).convert('RGB')
        label_img = Image.open(filename)
        
        
        label_img = label_img.resize(img.size)
        
        im = np.array(img)
        label = np.array(label_img)
        
        ### Remove epidermis from stage 2-Dermis
        # color = (255,255,0) #dermis - change to "include"
        color = (0,0,255) #epidermis
        
        im_masked = apply_mask(im,label, color,"exclude")
        masked_out = Image.fromarray(im_masked.astype(np.uint8))
        masked_out.save(stage_2_DE)
        
        ### Remove dermis from stage 2-Epidermis
        # color = (0,0,255) #epidermis - change to "include"
        color = (255,255,0) #dermis
        
        im_masked = apply_mask(im,label, color,"exclude")
        masked_out = Image.fromarray(im_masked.astype(np.uint8))
        masked_out.save(stage_2_EP)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Removing dermis for stage 2-Epidermis " +
                                     "and epidermis for stage 2-Dermis")

    # General settings
    parser.add_argument("--image_path", default= "/projects/patho1/sparse_segmentation/dataset/originals",
                        help="path to the original images")
    parser.add_argument("--mask_path", default= "/projects/patho1/sparse_segmentation/dataset/stage_1_masks" ,
                        help="path to masks of stage 1 (containg full dermis and epidermis)")
    parser.add_argument("--out_path_DE", default= "/projects/patho1/sparse_segmentation/dataset/stage_2_DE_masks",
                        help="path to save the images for stage 2-DE (removed EP)")
    parser.add_argument("--out_path_EP", default= "/projects/patho1/sparse_segmentation/dataset/stage_2_EP_masks" ,
                        help="path to save the images for stage 2-EP (removed DE)")
    parser.add_argument('--ext', default= "png", help = "image extension without '.' ")
    
    args = parser.parse_args()
    conversion(args)