import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from PIL import Image

from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from utils.dist import init_distributed_mode
from utils.arguments import load_opt_from_config_file
from utils.constants import COCO_PANOPTIC_CLASSES

from tasks import interactive_infer_image_idino_m2m_auto, prompt_switch


def load_model(model_size='L'):
    """
    Load the Semantic SAM model.

    Args:
        model_size (str): Model size, either 'T' for Tiny or 'L' for Large

    Returns:
        model: Loaded model
    """
    ckpt = "./weights/swinl_only_sam_many2many.pth"
    cfgs = {
        'T': "configs/semantic_sam_only_sa-1b_swinT.yaml",
        'L': "configs/semantic_sam_only_sa-1b_swinL.yaml"
    }

    if model_size not in cfgs:
        raise ValueError(f"Model size must be one of {list(cfgs.keys())}")

    sam_cfg = cfgs[model_size]
    opt = load_opt_from_config_file(sam_cfg)
    model = BaseModel(opt, build_model(opt)).from_pretrained(ckpt).eval().cuda()

    return model


@torch.no_grad()
def inference(model, image, level=[0], *args, **kwargs):
    """
    Run inference on an image using the Semantic SAM model.

    Args:
        model: The Semantic SAM model
        image: PIL Image to segment
        level: Segmentation level or 'All Prompt' for all levels

    Returns:
        tuple: (image with masks, list of mask dictionaries)
    """
    if level == 'All Prompt':
        level = [1, 2, 3, 4, 5, 6]
    else:
        if isinstance(level, str):
            level = [level.split(' ')[-1]]

    print(f"Using segmentation level(s): {level}")

    # Default parameters
    text_size, hole_scale, island_scale = 640, 100, 100
    text, text_part, text_thresh = '', '', '0.0'
    semantic = False

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        result = interactive_infer_image_idino_m2m_auto(
            model, image, level, text, text_part, text_thresh,
            text_size, hole_scale, island_scale, semantic,
            *args, **kwargs
        )
        return result


def load_image_from_path(image_path):
    """
    Load an image from a file path and return it as a PIL Image object.

    Args:
        image_path (str): Path to the image file

    Returns:
        PIL.Image: Loaded image as a PIL Image object
    """
    try:
        # Open the image file
        img = Image.open(image_path)

        # Convert to RGB if the image is in RGBA mode (has transparency)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        return img
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        return None


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

    print(f"Saved {len(masks)} masks to {output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Semantic SAM Segmentation Pipeline")

    parser.add_argument("--image", type=str, default="./examples/truck.jpg",
                        help="Path to input image")
    parser.add_argument("--level", type=str, default="2",
                        help="Segmentation level (1-6) or 'All Prompt'")
    parser.add_argument("--model_size", type=str, default="L",
                        choices=["T", "L"], help="Model size: T (tiny) or L (large)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save output masks")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize masks")
    parser.add_argument("--save_masks", action="store_true",
                        help="Save individual masks as binary images")

    return parser.parse_args()


def main():
    """Main function to run the segmentation pipeline."""
    args = parse_args()

    # Load model
    print(f"Loading model (size: {args.model_size})...")
    model = load_model(model_size=args.model_size)

    # Load image
    print(f"Loading image from {args.image}...")
    image = load_image_from_path(args.image)
    if image is None:
        print("Failed to load image. Exiting.")
        return

    # Run inference
    print("Running inference...")
    _, masks = inference(model=model, image=image, level=args.level)
    print(f"Found {len(masks)} masks")

    # Create output directory if needed
    if args.save_masks or args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)

    # Save masks if requested
    if args.save_masks:
        save_masks(masks, args.output_dir)

    # Visualize masks if requested
    if args.visualize:
        vis_path = os.path.join(args.output_dir, "visualization.png") if args.save_masks else None
        visualize_masks(masks, save_path=vis_path)

    print("Done!")


if __name__ == "__main__":
    main()

"""
usage:
python ./psg_data/segment-pipeline.py --image ./examples/truck.jpg --level 2 --visualize --save_masks --output_dir ./output

Example usage:
python segment-pipeline.py --image ./examples/truck.jpg --level 2 --visualize --save_masks --output_dir ./output

Output format:
[
    {'segmentation': array([[False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        ...,
        [ True,  True,  True, ...,  True,  True,  True],
        [ True,  True,  True, ...,  True,  True,  True],
        [ True,  True,  True, ...,  True,  True,  True]]),
    'area': 208978,
    'bbox': [0, 379, 959, 260],
    'predicted_iou': 0.98974609375,
    'point_coords': [[0.984375,
        0.984375,
        0.004999999888241291,
        0.004999999888241291]],
    'stability_score': 0.9954926371574402,
    'crop_box': [0, 0, 960, 640]},
    ...
]
"""
