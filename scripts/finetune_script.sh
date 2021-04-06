#!/usr/bin/env bash

cd ..
cd /model


python3 main.py \
--txt_folder "/projects/patho1/sparse_segmentation/dataset/txt_files/stage_2_DE/" \
--epochs 100 \
--lr 0.0001 \
--encoder "resnet34" \
--act_fcn "softmax2d" \
--loss_fcn "Dice" \
--classes "UL" "BG" "DMN" \
--finetune True \
--resume "/projects/patho1/sparse_segmentation/models/stage_1/best_model.pth" \
--model_dir "/projects/patho1/sparse_segmentation/models/stage_2_DE/" 



python3 main.py \
--txt_folder "/projects/patho1/sparse_segmentation/dataset/txt_files/stage_2_EP/" \
--epochs 100 \
--lr 0.0001 \
--encoder "resnet34" \
--act_fcn "softmax2d" \
--loss_fcn "Dice" \
--classes "UL" "BG" "EPN" \
--finetune True \
--resume "/projects/patho1/sparse_segmentation/models/stage_1/best_model.pth" \
--model_dir "/projects/patho1/sparse_segmentation/models/stage_2_EP/" 


