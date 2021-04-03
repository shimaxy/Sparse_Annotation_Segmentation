#!/usr/bin/env bash




python3 overlay.py \
--base_image_path "/projects/patho1/sparse_segmentation/results/stage_1/" \
--overlay_image_path "/projects/patho1/sparse_segmentation/results/stage_2_DE/" \
--out_path "/projects/patho1/sparse_segmentation/results/overlay/overlay_1_2_DE/"  \
--classes "DMN"


python3 overlay.py \
--base_image_path "/projects/patho1/sparse_segmentation/results/overlay/overlay_1_2_DE/" \
--overlay_image_path "/projects/patho1/sparse_segmentation/results/stage_2_EP/" \
--out_path "/projects/patho1/sparse_segmentation/results/overlay/overlay_1_2_DE_EP/"  \
--classes "EPN"
