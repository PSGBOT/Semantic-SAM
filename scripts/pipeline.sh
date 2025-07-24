#!/bin/bash

# Check if dataset name argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Example: $0 indoor"
    exit 1
fi

DATASET_NAME=$1

python ./psg_data/segmenter.py --input ../Data/$DATASET_NAME --level "2 3 4 5 6" --save_masks --output_dir ./output/$DATASET_NAME/level_seg_dataset --worker 1 --summary
python ./psg_data/post_process.py --input ./output/$DATASET_NAME/level_seg_dataset/segmentation_results.json --output ./output/$DATASET_NAME/part_seg_dataset --level "2 3 4 5 6"
# cp -r ./output/$DATASET_NAME ../Data/$DATASET_NAME
# python relation_generator.py --dataset_dir ../Data/$DATASET_NAME/part_seg_dataset/part_seg_dataset.json --src_image_dir ../Data/$DATASET_NAME/part_seg_dataset --output_dir relations/$DATASET_NAME
