import os
os.environ["CUDA_VISIBLE_DEVICES"]  =  "2"
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp

from PIL import Image
import pandas as pd
from glob import glob

from dataloader import *
from visualization import *
from augmentation import *


class NameIter(BaseDataset):
    
    CLASSES = ["EL","BG","COR", "EP", "DE", "DMN","EPN"]

    def __init__(
            self, 
            images_dir, 
            masks_dir,
            classes = None, #CLASSES, 
            augmentation = None, 
            preprocessing = None,
            input_size = (480,360)
    ):
            
        with open(images_dir) as img_txt:
            self.images_fps = img_txt.read().splitlines()                
            
        with open(masks_dir) as mask_txt:
            self.masks_fps = mask_txt.read().splitlines()     
                
        # convert str names to class values on masks
        if classes:
            self.class_values = [self.CLASSES.index(cls) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.input_size = input_size
    
    def __getitem__(self, i):
        #read data
        image_name = self.images_fps[i].split("/")[-1]
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0) 

        image = cv2.resize(image,self.input_size)
        mask = cv2.resize(mask,self.input_size)
        
        
        # extract certain classes from mask (e.g. cars)    
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
            
        return image, mask, image_name
        
    def __len__(self):
        return len(self.images_fps)
    
    
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "f", "0", "no", "n", "False"}:
        return False
    elif value.lower() in {"True", "true", "t", "1", "yes", "y"}:
        return True
    #raise ValueError(f"{value} is not a valid boolean value")
        
def validation(args):

    
    class_color = {"UL":(0, 0, 0), 
                    "BG":(128, 128, 128), 
                    "COR":(255, 0, 255), 
                    "EP":(0, 0, 255), 
                    "DE":(255, 255, 0),
                    "DMN":(0, 255, 0), 
                    "EPN":(0, 85, 0), 
                    }

    
    curr_CLASSES = args.classes

    colors = []
    for c in curr_CLASSES:
        class2color = class_color[c]
        colors.append(class2color)
    
     
    preprocessing_fn  =  smp.encoders.get_preprocessing_fn(args.encoder, "imagenet")    
    DEVICE = "cuda" 
    best_model = torch.load(args.best_model)
    
    
    test_img_txt  = os.path.join(args.txt_folder, "test_images.txt")
    test_mask_txt  =  os.path.join(args.txt_folder, "test_masks.txt")   
    
    test_dataset_vis = NameIter(
        test_img_txt, test_mask_txt, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes= curr_CLASSES,
        input_size = args.inpSize)
    
    if not os.path.exists(args.results):
        os.mkdir(args.results)
        
    for n in range(len(test_dataset_vis)):
        
        image, gt_mask, image_name = test_dataset_vis[n]
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        pr_mask_stack = np.stack(pr_mask, axis=-1).astype('float')
        
        mask_add = np.zeros((pr_mask_stack.shape[0],pr_mask_stack.shape[1],3))

        for i in range(pr_mask_stack.shape[2]):
            
            mask_color = [colors[i][2],colors[i][1],colors[i][0]] #RGB to BGR (cv2 thing)
            curr = pr_mask_stack[:,:,i]
            
            curr_m = cv2.merge((curr,curr,curr))
            curr_overlay = mask_color*curr_m
            
            mask_add = cv2.addWeighted(mask_add,1,curr_overlay,1,0)
                
        out_name = os.path.join(args.results , image_name)
        cv2.imwrite(out_name, mask_add.astype(np.uint8))
        print(out_name) 

        if args.save_preds:
            out_array = args.results.replace("Stage_","preds/Stage_")
            out_array_name= os.path.join(out_array , image_name)
            out_array_name = out_array_name.replace("png","npy")
            
    
            if not os.path.exists(out_array):
                os.mkdir(out_array)
            
    
            pr_mask_data = np.asarray(pr_mask)
            np.save(out_array_name, pr_mask_data)    
        
        
         
        
if __name__ == "__main__":
    parser  =  argparse.ArgumentParser(description = "Validation script for" +
                                     "Sparse annotation segmentation")

    # Dataset related settings
    parser.add_argument("--data", default = "/projects/patho1/sparse_segmentation/split_set/"
                        , help = "path to dataset")
    parser.add_argument("--mask", default = "/projects/patho1/sparse_segmentation/split_set/masks"
                        , help = "path to masks")
    parser.add_argument("--label", default = "/projects/patho1/sparse_segmentation/split_set/"
                    , help="path to labels")
    parser.add_argument("--txt_folder", default = "/projects/patho1/sparse_segmentation/split_set/txt_files/"
                , help = "path to txt files")
    parser.add_argument("--results", default = "/projects/patho1/sparse_segmentation/results/no_bv_inf/"
                , help = "path to results folder")
    parser.add_argument("--save_preds", default = False, type = str_to_bool, nargs = "?", const = True, help = "save preds in .npy format ")

    parser.add_argument("--num_classes", default = 7, type = int, help = "Number of classes")

    parser.add_argument('--classes', nargs='+', help='classes to be included', required=True)
    ###default = ["UL","BG","COR", "EP", "DE","DMN","EPN"]

    parser.add_argument("--inpSize", default = (480,360), type = int, help = "Input image size (default: 480,360))")

    # Model settings
    parser.add_argument("--encoder", default = "resnet34", 
                        help = "model for encoder of segmentation model")
    parser.add_argument("--best_model", type = str, default = "/projects/patho1/sparse_segmentation/models/cor_epi_de/resnet34_softmax2d_lr0.0001_epoch_202.pth",
                        help = "directory to output saved checkpoint")
    parser.add_argument("--act_fcn", type = str, default = "softmax2d",
                    help = "could be None for logits or sigmoid or softmax2d for multicalss segmentation")
   
    ## Experiment related settings
    parser.add_argument("--use_gpu", default = True, type = str_to_bool, nargs = "?", const = True, help = "Use gpu for experiment")
    parser.add_argument("--gpu_id", nargs = "+", type = int)

    args  =  parser.parse_args()
    validation(args)
