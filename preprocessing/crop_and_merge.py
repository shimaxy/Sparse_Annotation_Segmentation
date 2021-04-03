'''
cropping big ROIs or WSI (--mode "crop")
or merging back the cropped images/results (--mode "merge")
'''

import numpy as np
from glob import glob
import cv2
import os
import math
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import argparse

def crop_fcn(image,w,h,base_out_name):
    W,H = image.size      
    counter_w = 1
    for start_w in range(1,W+1,w):
        counter_h = 1
        for start_h in range(1,H+1,h):
            end_w = start_w + w
            end_h = start_h + h           
            if end_w > W:
                end_w = W                              
            if end_h > H:
                end_h = H                
            
            curr_crop = image.crop((start_w,start_h,end_w,end_h))           
            out_name = base_out_name.replace(".png", "_w_" + str(counter_w) + "_h_" +  
                                             str(counter_h)+ ".png")
            curr_crop.save(out_name)      
            counter_h += 1
        counter_w += 1
        

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

def merge_fcn(image,w,h,result_name,merge_name):
    W,H = image.size   
    try:
        counter_w = 1
        for start_w in range(1,W+1,w):
            counter_h = 1
            for start_h in range(1,H+1,h):
                end_w = start_w + w
                end_h = start_h + h            
                if end_w > W:
                    end_w = W+1                              
                if end_h > H:
                    end_h = H+1                                       
                crop_name = result_name.replace(".png", "_w_" + str(counter_w) + "_h_" + str(counter_h) + ".png")
                
                curr_crop = Image.open(crop_name) 
                curr_resize = curr_crop                        
                curr_resize = curr_crop.resize((end_w - start_w,end_h - start_h))
                
                if (counter_h == 1):
                    merge_h = curr_resize
                else:
                    merge_h = get_concat_v(merge_h,curr_resize)
                counter_h += 1            
            if (counter_w == 1):
                merged = merge_h
            else:
                merged = get_concat_h(merged, merge_h)
                
            counter_w += 1
            
        merged.save(merge_name)



    except NameError:       
        curr_crop = Image.open(result_name)
        curr_crop.save(merge_name) 
    
def crop_merge(args):  
    image_path = args.ref_image_path
    curr_path = args.current_image_path
    
    if args.mode == "crop":
        crop_path = args.crop_out
        if not os.path.exists(crop_path):
            os.mkdir(crop_path)
            
    elif args.mode == "merge":
        results_path = args.ref_crop
        merge_back = args.merge_out
        if not os.path.exists(merge_back):
            os.mkdir(merge_back)
                    
                        
    times = args.crop_times
    out_w = 480 * times
    out_h = 360 * times
    
    for ref_name in sorted(glob(image_path + "/*." + args.ext)):  
        filename = ref_name.replace(image_path,curr_path)
        print(filename)    

        image = Image.open(filename)  
        
        if args.mode == "crop":
            image_name = ref_name.replace(image_path, crop_path)  
            image_name = image_name.replace("tif","png")
            crop_fcn(image,out_w,out_h,image_name)
        
        elif args.mode == "merge":
            ref_name = ref_name.replace("tif","png")
            result_name = ref_name.replace(image_path, results_path)
            merge_name = ref_name.replace(image_path, merge_back)
            merge_fcn(image,out_w,out_h,result_name,merge_name)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="crop or merge images")


    parser.add_argument("--ref_image_path", default = "/projects/patho1/sparse_segmentation/dataset/originals/" ,
                         type = str, help="refence image path")    
    parser.add_argument("--current_image_path", default = "/projects/patho1/sparse_segmentation/dataet/images/",
                        type = str, help="current image to crop")
    parser.add_argument("--crop_out", default =  "/projects/patho1/sparse_segmentation/dataset/cropped/images/",
                         type = str, help="where to save cropped images")
    
    parser.add_argument("--ref_crop", default =  "/projects/patho1/sparse_segmentation/results/cropped/stage_1/",
                          type = str, help="cropped images path (to be merged)")
    parser.add_argument("--merge_out", default =  "/projects/patho1/sparse_segmentation/results/merged/stage_1/",
                          type = str, help="where to save merged back images")   
    
    parser.add_argument("--mode", default= "crop" ,choices = ["crop", "merge"],
                         type = str, help = "crop or merge")  
    parser.add_argument("--crop_times", default= 3,
                        type = int,  help = "how many times in (480,360)")  
    parser.add_argument("--ext", default= "tif",
                         type = str, help = "extension of original images without '.'")  
    
    args = parser.parse_args()
    crop_merge(args)
        
        
        