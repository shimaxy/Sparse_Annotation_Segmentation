#!/usr/bin/env bash


python3 plot_preds.py \
--txt_folder "/projects/patho1/sparse_segmentation/dataset/txt_files/stage_1/" \
--use_gpu True --gpu_id 3 \
--classes "UL" "BG" "COR" "EP" "DE" \
--results "/projects/patho1/sparse_segmentation/results/stage_1/" \
--best_model "/projects/patho1/sparse_segmentation/models/stage_1/best_model.pth" \


python3 plot_preds.py \
--txt_folder "/projects/patho1/sparse_segmentation/dataset/txt_files/stage_2_DE/" \
--use_gpu True --gpu_id 3 \
--classes "UL" "BG" "DMN" \
--results "/projects/patho1/sparse_segmentation/results/stage_2_DE/" \
--best_model "/projects/patho1/sparse_segmentation/models/stage_2_DE/best_model.pth" \



python3 plot_preds.py \
--txt_folder "/projects/patho1/sparse_segmentation/dataset/txt_files/stage_2_EP/" \
--use_gpu True --gpu_id 3 \
--classes "UL" "BG" "EPN" \
--results "/projects/patho1/sparse_segmentation/results/stage_2_EP/" \
--best_model "/projects/patho1/sparse_segmentation/models/stage_2_EP/best_model.pth" \




