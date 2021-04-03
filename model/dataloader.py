import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """ Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. normalization, shape manipulation, etc.)
    
    """
    
    CLASSES = ["UL","BG","COR", "EP", "DE","DMN","EPN"]
    

    def __init__(
            self, 
            images_dir, 
            masks_dir,
            img_or_txt = "txt",
            classes = None, 
            augmentation = None, 
            preprocessing = None,
            input_size = (480,360)
    ):
            
        if img_or_txt =="txt":
            with open(images_dir) as img_txt:
                self.images_fps = img_txt.read().splitlines()                
                
            with open(masks_dir) as mask_txt:
                self.masks_fps = mask_txt.read().splitlines()
                
        elif img_or_txt == "img":
            self.ids = os.listdir(images_dir)            
            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]       
                
        # convert str names to class values on masks
        # if classes:
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
        
        
        # extract certain classes from mask 
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
            
        return image, mask
        
    def __len__(self):
        return len(self.images_fps)