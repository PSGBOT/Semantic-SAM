import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch

def visualize_masks(anns, image_ori=None, figsize=(15, 10), save_path=None):
    """
    Visualize segmentation masks one by one in black and white form.

    Args:
        anns: List of annotation dictionaries containing segmentation masks
        image_ori: Original image as a numpy array (optional)
        figsize: Size of the figure (width, height) in inches
        save_path: Path to save the visualization (optional)

    Returns:
        None - displays the masks directly
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

    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")

    # Only show if not in batch mode
    if not plt.isinteractive():
        plt.close(fig)
    else:
        plt.show()


def save_masks(masks, output_dir):
    """
    Save individual masks as binary images.

    Args:
        masks: List of mask dictionaries
        output_dir: Directory to save masks
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, mask in enumerate(masks):
        # Convert boolean mask to uint8 (0 and 255)
        mask_img = mask['segmentation'].astype(np.uint8) * 255

        # Create a PIL image and save
        mask_pil = Image.fromarray(mask_img)
        mask_path = os.path.join(output_dir, f"mask_{i+1}.png")
        mask_pil.save(mask_path)

@torch.no_grad()
def iou(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return intersection / union

@torch.no_grad()
def contain(mask1 : torch.Tensor, mask2 : torch.Tensor):
    """
    if ->1, mask1 contains mask2
    """
    intersection = (mask1 * mask2).sum()
    return intersection / mask2.sum()

@torch.no_grad()
def contain_matrix(masks):
    n_masks = len(masks)
    contain_matrix = np.ones((n_masks, n_masks))
    for i in range(n_masks):
        for j in range(n_masks):
            if i == j:
                continue
            else:
                contain_matrix[i, j] = contain(masks[i], masks[j])
    return contain_matrix

@torch.no_grad()
def intersect(mask1 : torch.Tensor, mask2 : torch.Tensor):
    intersection_mask = (mask1 * mask2)
    return intersection_mask


def discard_submask(masks):
    """
    Discard those segmentations that are submasks of another result

    Args:
        seg_res: List of segmentation results(dict), bitmask is stored as seg_res[index]["segmentation"]
        contain_m: Containment matrix
    """
    mask_tensors = [torch.tensor(mask['segmentation'], dtype=torch.int).cuda() for mask in masks]
    matrix = contain_matrix(mask_tensors)
    #print(matrix)
    for i in range(len(masks)):
        for j in range(len(masks)):
            if i == j:
                continue
            else:
                if matrix[i, j] > 0.9 and matrix[j, i] < matrix[i, j]:
                    masks[j] = None
                    #print(f"Discard submask {j}, because it is submask of {i}")
    # discard None in list
    res = [mask for mask in masks if mask is not None]
    return res

def apply_mask(parent_masks, child_masks):
    parent_num = len(parent_masks)
    child_num = len(child_masks)
    valid_children = []
    for i in range(parent_num):
        for j in range(child_num):
            parent_masks[i]['segmentation'] = parent_masks[i]['segmentation'] * child_masks[j]['segmentation']
    return
