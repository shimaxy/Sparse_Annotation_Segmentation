import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from glob import glob
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
from dataloader import *
from visualization import *
from augmentation import *

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n', 'False'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    #raise ValueError(f'{value} is not a valid boolean value')
        
def main(args):
    

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # -------------------------------------------------------------------------
    # create segmentation model with pretrained encoder
    # -------------------------------------------------------------------------
    if args.finetune == False:
        
        model = smp.Unet(
            encoder_name = args.encoder, 
            encoder_weights = args.encoder_weights, 
            classes = len(args.classes), 
            activation=args.act_fcn,
        )

    else:
        model = torch.load(args.resume)
        
        
        model.segmentation_head = smp.base.SegmentationHead(
                            in_channels=16,
                            out_channels=len(args.classes), 
                            activation=args.act_fcn,
                            kernel_size=3,
                        )

                
    preprocessing_fn  =  smp.encoders.get_preprocessing_fn(args.encoder, "imagenet")
    
    
    DEVICE = "cuda" 
    
    # -------------------------------------------------------------------------
    # dataset loading, txt files
    # -------------------------------------------------------------------------

    train_img_txt  = os.path.join(args.txt_folder, "train_images.txt")
    train_mask_txt  =  os.path.join(args.txt_folder, "train_masks.txt")
    val_img_txt  =  os.path.join(args.txt_folder, "val_images.txt")
    val_mask_txt  =  os.path.join(args.txt_folder, "val_masks.txt")
    
    # -------------------------------------------------------------------------
    # loading the dataset
    # -------------------------------------------------------------------------

    train_dataset  =  Dataset(
        train_img_txt, train_mask_txt, 
        augmentation = get_training_augmentation(), 
        preprocessing = get_preprocessing(preprocessing_fn),
        classes = args.classes,
        input_size = args.inpSize)
    
    valid_dataset  =  Dataset(
        val_img_txt, val_mask_txt, 
        augmentation = get_validation_augmentation(), 
        preprocessing = get_preprocessing(preprocessing_fn),
        classes = args.classes,
        input_size = args.inpSize)
    
    train_loader  =  DataLoader(train_dataset, batch_size = args.train_batch_size,
                                shuffle = True, num_workers = args.train_workers)
    
    valid_loader  =  DataLoader(valid_dataset, batch_size = args.valid_batch_size,
                                shuffle = False, num_workers = args.valid_workers)
    
    
    # -------------------------------------------------------------------------
    # parameters
    # -------------------------------------------------------------------------

    if args.loss_fcn == "Dice":
        loss  =  smp.utils.losses.DiceLoss()
        
    elif args.loss_fcn == "Jaccard":
        loss  =  smp.utils.losses.JaccardLoss()

     
    metrics  =  [smp.utils.metrics.IoU(threshold = 0.5),]     
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    else:
        scheduler = None

    # -------------------------------------------------------------------------
    # Creating epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    # -------------------------------------------------------------------------
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )
    
    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )
    

    # -------------------------------------------------------------------------
    # Training and Evaluation
    # -------------------------------------------------------------------------
    if os.path.isdir(args.model_dir) == False:
        os.mkdir(args.model_dir)
    max_score  =  0    
    for i in range(0, args.epochs):        
        print("\nEpoch: {}".format(i))
        train_logs  =  train_epoch.run(train_loader)
        valid_logs  =  valid_epoch.run(valid_loader)
        
        if args.scheduler:
            scheduler.step()
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs["iou_score"]:
            max_score  =  valid_logs["iou_score"]
            torch.save(model, os.path.join(args.model_dir , args.encoder + "_" +args.act_fcn +
                                            "_lr" + str(args.lr) + "_epoch_" + str(i) + ".pth"))
            
            torch.save(model, os.path.join(args.model_dir , "best_model.pth"))
            print("Model saved!")


        
if __name__ == "__main__":
    parser  =  argparse.ArgumentParser(description = "Training script for" +
                                     "Sparse annotation segmentation")
    # General settings
    parser.add_argument("--mode", default = "train", choices = ["train", "test"],
                        help = "Experiment mode")
    parser.add_argument("--finetune", default = False, type = str_to_bool, 
                        help = "Freeze batch norm layer for fine tuning")
    parser.add_argument("--resume", default = None, 
                        type = str, help = "path to latest checkpoint (default: none)")

    parser.add_argument("--train_workers", default = 12, type = int,
                        help = "number of data loading workers (default: 12)")
    parser.add_argument("--train_batch_size", default = 8, type = int,
                        help = "batch size (default: 8)")
    
    parser.add_argument("--valid_workers", default = 4, type = int,
                    help = "number of data loading workers (default: 4)")
    parser.add_argument("--valid_batch_size", default = 1, type = int,
                        help = "batch size (default: 1)")

    # Dataset related settings
    parser.add_argument("--data", default = "/projects/patho1/sparse_segmentation/dataset/split/images"
                        , help = "path to dataset")
    parser.add_argument("--mask", default = "/projects/patho1/sparse_segmentation/dataset/split/masks"
                        , help = "path to masks")
    parser.add_argument("--label", default = "/projects/patho1/sparse_segmentation/dataset/split/labels"
                    , help="path to labels")
    parser.add_argument("--txt_folder", default = "/projects/patho1/sparse_segmentation/split_set/txt_files/"
                , help = "path to txt files")
    parser.add_argument("--num_classes", default = 7, type = int, help = "Number of classes")

    parser.add_argument('--classes', nargs='+', help='classes to be included', required=True)
    ###default = ["UL","BG","COR", "EP", "DE","DMN","EPN"]

    parser.add_argument("--inpSize", default = (480,360), type = int, help = "Input image size (default: 480,360))")

    # Hyperparameters
    # LR scheduler settings
    parser.add_argument("--epochs", default = 50,
                        type = int, help = "number of maximum epochs to run")
    parser.add_argument("--lr", default = 0.0001, type = float,
                        help = "initial learning rate")
    parser.add_argument("--lr_decay", default = 0.5, type = float,
                        help = "factor by which lr should be decreased")
    parser.add_argument("--step_size", default = 50, type = float,
                        help = "lr step size")    
    parser.add_argument("--scheduler",default = False, type = str_to_bool, 
                        nargs = "?", const = True, help = "Learning rate scheduler")

    # Model settings
    parser.add_argument("--encoder", default = "resnet34", 
                        help = "model for encoder of segmentation model")
    parser.add_argument("--encoder_weights", default = "imagenet", 
                        help = "encoder weights for segmentation model")
    parser.add_argument("--model_dir", type = str, default = "/projects/patho1/sparse_segmentation/models/",
                        help = "directory to output saved checkpoint")
    parser.add_argument("--act_fcn", type = str, default = "softmax2d",
                    help = "could be None for logits or sigmoid or softmax2d for multicalss segmentation")
    parser.add_argument("--loss_fcn", default = "Dice", 
                        help = "loss function: Dice, Jaccard, BCE, BCEwithLL, NLL")

    ## Experiment related settings
    parser.add_argument("--use_gpu", default = True, type = str_to_bool, nargs = "?", const = True, help = "Use gpu for experiment")
    parser.add_argument("--gpu_id", nargs = "+", type = int)    
    args  =  parser.parse_args()
    main(args)
