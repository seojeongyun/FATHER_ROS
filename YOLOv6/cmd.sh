#!/bin/bash
clear
# ----- YOLOv6 -----
# For yolov6s
# --- COMMAND IN TERMINAL ---
base="base"
if [ "$1" = "$base" ]; then
	python core_train.py --batch-size 16 --data-path ./data/WIDER_FACE.yaml --conf-file ./configs/repopt/yolov6s_finetune.py --output-dir ./runs/train --name 6s --gpu-id 0 --worker 10 --fuse_ab --check-images --check-labels --eval-interval 10
elif [ "$1" = "base_no_fuse" ]; then
	python core_train.py --batch-size 16 --data-path ./data/WIDER_FACE.yaml --conf-file ./configs/repopt/yolov6s_hs.py --output-dir ./runs/train --name 6s_no_fuse --gpu-id 0 --worker 10 --check-images --check-labels --eval-interval 10
else
	echo "INVALID INPUT... POSSIBLE INPUT: [base, base_no_fuse]"
	echo "CURRENT INPUT:" $1
fi
# -------------------------------------------------
