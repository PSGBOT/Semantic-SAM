import torch
import numpy as np
import argparse
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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


def find_images(input_path):
    """
    Find all image files in a directory or return a single image path.

    Args:
        input_path (str): Path to an image file or directory containing images

    Returns:
        list: List of image file paths
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    if os.path.isfile(input_path):
        # If input is a file, check if it's an image
        ext = os.path.splitext(input_path)[1].lower()
        if ext in image_extensions:
            return [input_path]
        else:
            print(f"Warning: {input_path} is not a recognized image file")
            return []

    elif os.path.isdir(input_path):
        # If input is a directory, find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))

        return sorted(image_files)

    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return []


def process_single_image(args, model, image_path):
    """
    Process a single image with the model.

    Args:
        args: Command line arguments
        model: The loaded model
        image_path: Path to the image file

    Returns:
        dict: Results containing image path and masks
    """
    # Create image-specific output directory
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_output_dir = os.path.join(args.output_dir, image_name)

    # Load image
    image = load_image_from_path(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return {"image_path": image_path, "success": False, "masks": []}

    # Run inference
    try:
        _, masks = inference(model=model, image=image, level=args.level)

        # Create output directory if needed
        if args.save_masks or args.visualize:
            os.makedirs(image_output_dir, exist_ok=True)

        # Save masks if requested
        if args.save_masks:
            save_masks(masks, image_output_dir)

        # Visualize masks if requested
        if args.visualize:
            vis_path = os.path.join(image_output_dir, "visualization.png") if args.save_masks else None
            visualize_masks(masks, save_path=vis_path)

        return {
            "image_path": image_path,
            "success": True,
            "masks": masks,
            "output_dir": image_output_dir
        }

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {"image_path": image_path, "success": False, "masks": []}


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

    print(f"Saved {len(masks)} masks to {output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Semantic SAM Segmentation Pipeline")

    parser.add_argument("--input", type=str, default="./examples/truck.jpg",
                        help="Path to input image or directory containing images")
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
    parser.add_argument("--batch", action="store_true",
                        help="Process all images in the input directory")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker threads for loading images (batch mode only)")
    parser.add_argument("--summary", action="store_true",
                        help="Generate a summary of all processed images")

    return parser.parse_args()


def main():
    """Main function to run the segmentation pipeline."""
    args = parse_args()

    # Load model
    print(f"Loading model (size: {args.model_size})...")
    model = load_model(model_size=args.model_size)

    # Find images to process
    image_paths = find_images(args.input)

    if not image_paths:
        print(f"No images found at {args.input}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process images
    results = []

    if len(image_paths) == 1 and not args.batch:
        # Single image mode
        print(f"Processing image: {image_paths[0]}")
        result = process_single_image(args, model, image_paths[0])
        results.append(result)
    else:
        # Batch mode
        print(f"Processing {len(image_paths)} images in batch mode...")

        # Use ThreadPoolExecutor for loading images in parallel
        # Note: The actual inference still runs sequentially on the GPU
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for image_path in image_paths:
                futures.append(executor.submit(process_single_image, args, model, image_path))

            # Process results as they complete
            for future in tqdm(futures, total=len(image_paths)):
                result = future.result()
                results.append(result)

    # Generate summary if requested
    if args.summary and len(results) > 0:
        summary_path = os.path.join(args.output_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Processed {len(results)} images\n\n")

            for result in results:
                if result["success"]:
                    f.write(f"Image: {result['image_path']}\n")
                    f.write(f"Output directory: {result['output_dir']}\n")
                    f.write(f"Number of masks: {len(result['masks'])}\n")

                    # Add mask statistics
                    if result["masks"]:
                        areas = [mask['area'] for mask in result['masks']]
                        f.write(f"Average mask area: {np.mean(areas):.2f}\n")
                        f.write(f"Largest mask area: {np.max(areas):.2f}\n")
                        f.write(f"Smallest mask area: {np.min(areas):.2f}\n")

                    f.write("\n")
                else:
                    f.write(f"Failed to process: {result['image_path']}\n\n")

        print(f"Summary saved to {summary_path}")

    print("Done!")


if __name__ == "__main__":
    main()

"""
Usage examples:

# Process a single image
python segment-pipeline.py --input ./examples/truck.jpg --level 2 --visualize --save_masks --output_dir ./output

# Process all images in a directory
python ./psg_data/segment-pipeline.py --input ./input --batch --summary --level 2 --save_masks --output_dir ./output --workers 2

Output format for each mask:
{
    'segmentation': array([[False, False, False, ..., False, False, False],
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
    'crop_box': [0, 0, 960, 640]
}
"""
