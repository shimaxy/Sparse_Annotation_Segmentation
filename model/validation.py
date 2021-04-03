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

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "f", "0", "no", "n", "False"}:
        return False
    elif value.lower() in {"True", "true", "t", "1", "yes", "y"}:
        return True
        
def validation(args):

    # -------------------------------------------------------------------------
    # create segmentation model with pretrained encoder
    # -------------------------------------------------------------------------
        
    model = smp.Unet(
        encoder_name = args.encoder, 
        encoder_weights = "imagenet", 
        classes = len(args.classes), 
        activation=args.act_fcn,
    )        

    preprocessing_fn  =  smp.encoders.get_preprocessing_fn(args.encoder, "imagenet")
    
    DEVICE = "cuda" 
    
    best_model = torch.load(args.best_model)
    
    # -------------------------------------------------------------------------
    # dataset loading, txt files
    # -------------------------------------------------------------------------
    
    test_img_txt  = os.path.join(args.txt_folder, "test_images.txt")
    test_mask_txt  =  os.path.join(args.txt_folder, "test_masks.txt")

           
    test_dataset = Dataset(
        test_img_txt, test_mask_txt, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=args.classes,
        input_size = args.inpSize)
    
    test_dataloader = DataLoader(test_dataset)
    # -------------------------------------------------------------------------
    # parameters
    # -------------------------------------------------------------------------

    if args.loss_fcn == "Dice":
        loss  =  smp.utils.losses.DiceLoss()
        
        
    elif args.loss_fcn == "Jaccard":
        loss  =  smp.utils.losses.JaccardLoss()


    metrics = [smp.utils.metrics.IoU(threshold=0.5),]
    
    # -------------------------------------------------------------------------
    # Creating epoch runners
    # evaluate model on test set
    # -------------------------------------------------------------------------

    test_epoch = smp.utils.train.ValidEpoch(
        model = best_model, loss=loss,
        metrics = metrics, device=DEVICE)
    
    logs = test_epoch.run(test_dataloader)
           
        
if __name__ == "__main__":
    parser  =  argparse.ArgumentParser(description = "Validation script for" +
                                     "Sparse annotation segmentation")

    # Dataset related settings
    parser.add_argument("--data", default = "/projects/patho1/sparse_segmentation/split_set/"
                        , help = "path to dataset")
    parser.add_argument("--mask", default = "/projects/patho1/sparse_segmentation/split_set/"
                        , help = "path to masks")
    parser.add_argument("--label", default = "/projects/patho1/sparse_segmentation/split_set/"
                    , help="path to labels")
    parser.add_argument("--txt_folder", default = "/projects/patho1/sparse_segmentation/split_set/double_stage/txt_files/Stage_2_DE/"
                , help = "path to txt files")
    parser.add_argument("--results", default = "/projects/patho1/sparse_segmentation/results/"
                , help = "path to results folder")
    
    parser.add_argument("--num_classes", default = 7, type = int, help = "Number of classes")
    parser.add_argument('--classes', nargs='+', help='classes to be included', required=True)
    ###default = ["UL","BG","COR", "EP", "DE", "DMN","EPN"]
    
    parser.add_argument("--inpSize", default = (480,360), type = int, help = "Input image size (default: 480,360))")

    # Model settings
    parser.add_argument("--encoder", default = "resnet34", 
                        help = "model for encoder of segmentation model")
    parser.add_argument("--best_model", type = str, default = "/projects/patho1/sparse_segmentation/models/",
                        help = "directory to output saved checkpoint")
    parser.add_argument("--act_fcn", type = str, default = "softmax2d",
                    help = "could be None for logits or sigmoid or softmax2d for multicalss segmentation")
    parser.add_argument("--loss_fcn", default = "Dice", 
                        help = "loss function: Dice, Jaccard")
    ## Experiment related settings
    parser.add_argument("--use_gpu", default = True, type = str_to_bool, nargs = "?", const = True, help = "Use gpu for experiment")
    parser.add_argument("--gpu_id", nargs = "+", type = int)
    
    parser.add_argument("--save_results", default = False, type = str_to_bool, nargs = "?", const = True, help = "save results")

    args  =  parser.parse_args()
    validation(args)
