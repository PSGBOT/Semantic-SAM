# --------------------------------------------------------
# Semantic-SAM: Segment and Recognize Anything at Any Granularity
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------

import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from utils.visualizer import Visualizer
from typing import Tuple
from PIL import Image
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
import cv2
import io
from .automatic_mask_generator import SemanticSamAutomaticMaskGenerator
metadata = MetadataCatalog.get('coco_2017_train_panoptic')

def interactive_infer_image(model, image,level,all_classes,all_parts, thresh,text_size,hole_scale,island_scale,semantic, refimg=None, reftxt=None, audio_pth=None, video_pth=None, visualize=None):
    t = []
    t.append(transforms.Resize(int(text_size), interpolation=InterpolationMode.BICUBIC))
    transform1 = transforms.Compose(t)
    image_ori = transform1(image)

    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    mask_generator = SemanticSamAutomaticMaskGenerator(model,points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            min_mask_region_area=10,
            level=level,
        )

    outputs = mask_generator.generate(images)
    if visualize:
        fig=plt.figure(figsize=(10, 10))
        plt.imshow(image_ori)
        show_anns(outputs)
        fig.canvas.draw()
        im=Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    else:
        im=image
    return im, outputs

def interactive_infer_image_multilevel(model, image, levels, all_classes,all_parts, thresh,text_size,hole_scale,island_scale,semantic, refimg=None, reftxt=None, audio_pth=None, video_pth=None, visualize=None):
    t = []
    t.append(transforms.Resize(int(text_size), interpolation=InterpolationMode.BICUBIC))
    transform1 = transforms.Compose(t)
    image_ori = transform1(image)

    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    outputs = {}

    for level in levels:
        mask_generator = SemanticSamAutomaticMaskGenerator(model,points_per_side=32,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.92,
                min_mask_region_area=10,
                level=[level],
            )
        output = mask_generator.generate(images)
        outputs["level{}".format(level)]=output

    return outputs

def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True

def show_anns(anns):
    if len(anns) == 0:
        print("No instances to display")
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=False)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        plt.imshow(np.dstack((img, m*0.35)))

# a function to visualize the masks
def visualize_masks(anns, image_ori=None, figsize=(15, 10)):
    """
    Visualize segmentation masks one by one in black and white form.

    Args:
        anns: List of annotation dictionaries containing segmentation masks
        image_ori: Original image as a numpy array (optional)
        figsize: Size of the figure (width, height) in inches

    Returns:
        None - displays the masks directly

    Usage in Jupyter:
        visualize_masks(outputs)
    """
    if len(anns) == 0:
        print("No instances to display")
        return

    # Sort by area for better visualization
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    # Calculate grid dimensions
    n_masks = len(sorted_anns)
    n_cols = min(4, n_masks)
    n_rows = (n_masks + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Display each mask
    for i, ann in enumerate(sorted_anns):
        if i >= len(axes):
            break

        m = ann['segmentation']

        # Display mask in black and white
        axes[i].imshow(m, cmap='gray')

        # Add area information
        area_text = f"Area: {ann['area']:.0f}"
        if 'predicted_iou' in ann:
            area_text += f"\nIoU: {ann['predicted_iou']:.2f}"
        if 'stability_score' in ann:
            area_text += f"\nStability: {ann['stability_score']:.2f}"

        axes[i].set_title(f"Mask {i+1}\n{area_text}")
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(n_masks, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
