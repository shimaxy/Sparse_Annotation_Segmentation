#!/usr/bin/env bash

echo "Stage_1"

python3 validation.py \
--txt_folder "/projects/patho1/sparse_segmentation/dataset/txt_files/stage_1/" \
--use_gpu True --gpu_id 3 \
--classes "UL" "BG" "COR" "EP" "DE" \
--loss_fcn "Dice" \
--results "/projects/patho1/sparse_segmentation/results/stage_1/" \
--best_model "/projects/patho1/sparse_segmentation/models/stage_1/best_model.pth" \



echo "Stage_2_DE"

python3 validation.py \
--txt_folder "/projects/patho1/sparse_segmentation/dataset/txt_files/stage_2_DE/" \
--use_gpu True --gpu_id 3 \
--classes "UL" "BG" "DMN" \
--loss_fcn "Dice" \
--results "/projects/patho1/sparse_segmentation/stuff/report/MIA/results/stage_2_DE/" \
--best_model "/projects/patho1/sparse_segmentation/models/stage_2_DE/best_model.pth" \



echo "Stage_2_EP"

python3 validation.py \
--txt_folder "/projects/patho1/sparse_segmentation/dataset/txt_files/stage_2_EP/" \
--use_gpu True --gpu_id 3 \
--classes "UL" "BG" "EPN" \
--loss_fcn "Dice" \
--results "/projects/patho1/sparse_segmentation/results/stage_2_EP_previous_93/" \
--best_model "/projects/patho1/sparse_segmentation/models/stage_2_EP/best_model.pth" \

