"""
Segmentation of background using otsu thresholding

- Shima
"""
import numpy as np
from glob import glob
import cv2
from skimage.filters import threshold_multiotsu
import os
import imutils
import pickle
from PIL import Image
import argparse


def Otsu_thresh(args):
    in_path = args.in_path
    out_path = args.out_path
    categories = args.categories
    gauss_blur_size = args.gauss_blur_size
    save_pickle = args.save_pickle
    min_size = args.min_cc_size
    kernel_size = args.kernel_size
    iter_d = args.dialation_iteration
    choice = args.category_directory_or_single
    
    def threshold_output(filename):
        Img_O = cv2.imread(filename)
        Img = Img_O[:,:,1]
           
        blur = cv2.GaussianBlur(Img,gauss_blur_size,0)
        ret2,thresh_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh_img_inv = np.invert(thresh_img)           
        kernel = np.ones(kernel_size,np.uint8)
        curr = cv2.dilate(thresh_img_inv,kernel,iterations=iter_d)           
        cnts = cv2.findContours(curr.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        out_mask = np.ones(curr.shape, dtype="uint8")* 255            
        for c in cnts:
            if min_size<cv2.contourArea(c):
                cv2.drawContours(out_mask, [c], -1, 0, -1)          
        out_image = cv2.bitwise_not(out_mask)
 
           
        return out_image
        
    if choice == "category":
        for category in categories:
            for subpath in sorted(glob(os.path.join(in_path , category) + "/*")):
                for filename in sorted(glob(subpath + "/*.jpg")):
    
                    out_image = threshold_output(filename)
                    if not os.path.exists(out_path):
                        os.mkdir(out_path)
                    
                    image_name = filename.split("/")[-1]                  
                    out_name = os.path.join(out_path , image_name)
                    Image.fromarray(out_image).save(out_name)
    
                    if save_pickle == "yes":
                        pickle_name = out_name.replace(".jpg",".pkl")    
                        pickle.dump(out_image, open(pickle_name, "wb"))

    elif choice == "directory":        
        for filename in sorted(glob(in_path + "/*.jpg")):

            out_image = threshold_output(filename)
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            
            image_name = filename.split("/")[-1]                  
            out_name = os.path.join(out_path , image_name)
            Image.fromarray(out_image).save(out_name)

            if save_pickle == "yes":
                pickle_name = out_name.replace(".jpg",".pkl")    
                pickle.dump(out_image, open(pickle_name, "wb"))
                
    elif choice == "single":        
        out_image = threshold_output(filename)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        
        image_name = filename.split("/")[-1]                  
        out_name = os.path.join(out_path , image_name)
        Image.fromarray(out_image).save(out_name)

        if save_pickle == "yes":
            pickle_name = out_name.replace(".jpg",".pkl")    
            pickle.dump(out_image, open(pickle_name, "wb"))
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="resizing background masks from one " +
                                     "resolution to another resolution")

    # General settings
    parser.add_argument("--in_path",default= "/projects/patho1/melanoma_diagnosis/x10/segmented/Set1/",
                        help="path to the previous resolution masks")
    parser.add_argument("--out_path", default= "/projects/patho1/melanoma_diagnosis/x10/bg_masks/",
                        help="saving directory")
    
    parser.add_argument("--categories", default=["1","3","4","5"],
                        help="categories to generate masks on")
    
    parser.add_argument("--category_directory_or_single", default="category",
                        choices = ["category","directory","single"],
                help="are you running on a category, a directory, or a single image")
    
    parser.add_argument("--gauss_blur_size", default=(5,5),
                    help="size of first blurring on mask")
    
    parser.add_argument("--save_pickle", default="no", choices = ["yes","no"],
                    help="select yes if .pkl of the result should be saved") 
    
    parser.add_argument("--min_cc_size", default=100000,
                    help="minimum size of a connected component in the results") 
    
    parser.add_argument("--gauss_blur_size", default=(5,5),
                help="dilation kernel size")
    
    parser.add_argument("--dialation_iteration", default=1,
                help="iteration of applying dialation")
    
    args = parser.parse_args()
    Otsu_thresh(args)   