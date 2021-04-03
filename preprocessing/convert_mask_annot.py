'''
Converting color masks to labels (--mode "mask2label"), used to train the U-Net
or converting labels to color masks (--mode "label2mask") for visualization
'''

import numpy as np
import cv2
from glob import glob
import argparse


def conversion(args):
    mode = args.mode
    inpath = args.in_path
    out_path = args.output_path
    classes = args.classes
    ext = args.ext
    
    #RGB color of labels
    colors =  {"UL" : (0, 0, 0),        #Unlabeled
               "BG" : (128, 128, 128),  #Background
               "COR" : (255, 0, 255),   #Corneum
               "EP" : (0, 0, 255),      #Epidermis
               "DE" : (255, 255, 0),    #Dermis
               "DMN" : (0, 255, 0),     #Dermal Nests
               "EPN" : (0, 85, 0)}      #Epidermal Nests


    ### Converting color masks to labels
    if mode == "mask2label":
        for filename in sorted(glob(inpath + "/*." + ext)):
            
            image = cv2.imread(filename)            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            img_size = image.shape
            label = np.zeros(img_size, dtype="uint8")
                  
            for i in range(len(classes)):
                curr_color = colors[classes[i]]
                lower = np.array(curr_color, dtype = "uint8")
                upper = np.array(curr_color, dtype = "uint8")
                
                mask = cv2.inRange(image, lower, upper)            
                label[np.where((image==curr_color).all(axis=2))] = i
            
            outname = filename.replace(inpath, out_path) 
            cv2.imwrite(outname, label[:,:,0].astype(np.uint8)) 

    ### Converting labels to color masks                
    elif mode == "label2mask":
        for filename in sorted(glob(inpath + "/*." + ext)):
            print(filename)
            annot = cv2.imread(filename)
            img_size = image.shape
            mask = np.zeros(img_size, dtype="uint8")
               
        
            for label in range(len(classes)):      
                curr_color = [label,label,label]
                BGR = colors[classes[label]]
                mask[np.where((annot==curr_color).all(axis=2))] = BGR
                
            outname = filename.replace(inpath, out_path) 
            cv2.imwrite(outname, mask.astype(np.uint8)) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate color masks from " +
                                     "labels or labels from color masks)")

    # General settings
    parser.add_argument("--mode", default= "mask2label", choices = ["mask2label", "label2mask"],
                        help="generate labels from masks or masks from labels")
    parser.add_argument("--in_path", default= "/projects/patho1/sparse_segmentation/dataset/masks",
                        help="path to the masks")
    parser.add_argument("--output_path", default= "/projects/patho1/sparse_segmentation/dataset/labels" ,
                        help="path to the labels")
    parser.add_argument('--classes', nargs='+', help='classes to be included', required=False)
    parser.add_argument('--ext', default= "png", help = "image extension without '.' ")
    
    args = parser.parse_args()
    conversion(args)