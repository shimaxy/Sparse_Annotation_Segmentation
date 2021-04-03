#!/usr/bin/env bash

python3 main.py \
--txt_folder "/projects/patho1/sparse_segmentation/dataset/txt_files/stage_1/" \
--use_gpu True --gpu_id 2 \
--epochs 1000 \
--encoder "resnet34" \
--act_fcn "softmax2d" \
--loss_fcn "Dice" \
--classes "UL" "BG" "COR" "EP" "DE" \
--model_dir "/projects/patho1/sparse_segmentation/models/stage_1/"
