'''
Correcting color masks in general, 
or generating color masks for stage 1, stage 2-Dermis, or stage 2-Epidermis
'''

import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
import argparse

def correction(args):
    inpath = args.in_path
    out_path = args.output_path
    ext = args.ext
    
    colors =  {"UL" : (0, 0, 0),        #Unlabeled
               "BG" : (128, 128, 128),  #Background
               "COR" : (255, 0, 255),   #Corneum
               "EP" : (0, 0, 255),      #Epidermis
               "DE" : (255, 255, 0),    #Dermis
               "DMN" : (0, 255, 0),     #Dermal Nests
               "EPN" : (0, 85, 0),      #Epidermal Nests
               "BV" : (255, 0, 0),      #Blood Vessel
               "INF" : (255, 85, 0),    #Inflammatory Cells
               "ED" : (85, 0, 0)}       #Eccrine Ducts
        
    
    for filename in sorted(glob(inpath + "/*." + ext)):
        
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
        
        ### Stage_1
        if args.stage_1:
            stage_1_masks = out_path + "/masks_stage_1"
            im_stage_1 = image.copy()
            
            ### replacing entities inside DE with DE
            curr_list = ["BV","INF","ED","DMN"]
            for curr in curr_list:
                curr_color = colors[curr]
                rep_color = colors["DE"]
                im_stage_1[np.where((image==curr_color).all(axis=2))] = rep_color
        
            ### replacing entities inside EP with EP    
            curr_list = ["EPN"]
            for curr in curr_list:
                curr_color = colors[curr]
                rep_color = colors["EP"]
                im_stage_1[np.where((image==curr_color).all(axis=2))] = rep_color
            
            outname_stage_1 = filename.replace(inpath,stage_1_masks)
            cv2.imwrite(outname_stage_1, cv2.cvtColor(im_stage_1, cv2.COLOR_RGB2BGR)) 
          
            
        ### Stage_2_Dermis
        if args.stage_2_DE:
            stage_2_DE = out_path + "/masks_DE"
            im_DE = image.copy()
            
            ### removing everything other than BG, BV, INF, ED, DMN
            curr_list = ["COR","DE","EP","EPN"]
            for curr in curr_list:
                curr_color = colors[curr]
                rep_color = colors["UL"]
                im_DE[np.where((image==curr_color).all(axis=2))] = rep_color
            
                ###removing no-tissues parts from masks
            im_DE[np.where((im_stage_1==colors["UL"]).all(axis=2))] = colors["UL"]
         
            outname_DE = filename.replace(inpath,stage_2_DE)
            cv2.imwrite(outname_DE, cv2.cvtColor(im_DE, cv2.COLOR_RGB2BGR)) 
         
        ### Stage_2_Epidermis
        if args.stage_2_EP:
            stage_2_EP = out_path + "/masks_EP"
            im_EP = image.copy()
                ### removing everything other than BG and EPN
            curr_list = ["COR","DE","BV","INF","ED","DMN","EP"]
            for curr in curr_list:
                curr_color = colors[curr]
                rep_color = colors["UL"]
                im_EP[np.where((image==curr_color).all(axis=2))] = rep_color
                  
                ###removing no-tissues parts from masks
            im_EP[np.where((im_stage_1==colors["UL"]).all(axis=2))] = colors["UL"]
            
            outname_EP = filename.replace(inpath,stage_2_EP)
            cv2.imwrite(outname_EP, cv2.cvtColor(im_EP, cv2.COLOR_RGB2BGR)) 


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "f", "0", "no", "n", "False"}:
        return False
    elif value.lower() in {"True", "true", "t", "1", "yes", "y"}:
        return True
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate color masks from " +
                                     "labels or labels from color masks)")

    # General settings
    parser.add_argument("--stage_1", default = True, type = str_to_bool, nargs = "?", const = True,
                        help="generate stage 1 color masks")
    parser.add_argument("--stage_2_DE", default = True, type = str_to_bool, nargs = "?", const = True,
                        help="generate stage 2-Dermis color masks")
    parser.add_argument("--stage_2_EP", default = True, type = str_to_bool, nargs = "?", const = True,
                        help="generate stage 2-Epidermis color masks")
        
    parser.add_argument("--in_path", default= "/projects/patho1/sparse_segmentation/dataset/masks",
                        help="path to the images and xmls")
    parser.add_argument("--output_path", default= "/projects/patho1/sparse_segmentation/dataset/labels" ,
                        help="path to the images and xmls")
    parser.add_argument('--ext', default= "png", help = "image extension without '.' ")

    
    args = parser.parse_args()
    correction(args)    
