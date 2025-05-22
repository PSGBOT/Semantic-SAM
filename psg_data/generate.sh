#!/bin/sh
python ./psg_data/segmenter.py --input ./input --level "2 3 4 5 6" --save_masks --output_dir ./output/level_seg_dataset --worker 1 --summary
python ./psg_data/post_process.py --input ./output/level_seg_dataset/segmentation_results.json --output ./output/part_seg_dataset --level "2 3 4 5 6"
