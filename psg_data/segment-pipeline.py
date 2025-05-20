import numpy as np
import argparse
import os
import torch  # Add torch import
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from psg_data.model import load_model
from psg_data.model import inference
from psg_data.model import inference_multilevel
from utils.psg_utils.image import load_image_from_path
from utils.psg_utils.image import find_images
from utils.psg_utils.mask import visualize_masks
from utils.psg_utils.mask import save_masks
from utils.psg_utils.mask import discard_submask


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

        masks = discard_submask(masks)
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

def process_single_image_multilevel(args, model, image_path, id):
    """
    Process a single image with multiple levels

    Args:
        args: Command line arguments
        model: The loaded model
        image_path: Path to the image file

    Returns:
        dict: different levels' masks
    """
    # Create image-specific output directory
    image_name = "id " + str(id)
    image_output_dir = os.path.join(args.output_dir, image_name)

    # Load image
    image = load_image_from_path(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return {"image_path": image_path, "success": False, "masks": []}

    # Run inference
    try:
        multilevel_masks = inference_multilevel(model=model, image=image, level=args.level)

        for level in multilevel_masks:
            multilevel_masks[level] = discard_submask(multilevel_masks[level])
            # Create output directory if needed
            level_dir = image_output_dir + "/" + level
            if args.save_masks or args.visualize:
                os.makedirs(level_dir, exist_ok=True)

            # Save masks if requested
            if args.save_masks:
                save_masks(multilevel_masks[level], level_dir)

            # Visualize masks if requested
            if args.visualize:
                vis_path = os.path.join(image_output_dir, "visualization.png") if args.save_masks else None
                visualize_masks(multilevel_masks[level], save_path=vis_path)

        return {
            "image_path": image_path,
            "success": True,
            "masks": multilevel_masks,
            "output_dir": image_output_dir
        }

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {"image_path": image_path, "success": False, "masks": None}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Semantic SAM Segmentation Pipeline")

    parser.add_argument("--input", type=str, default="./examples/truck.jpg",
                        help="Path to input image or directory containing images")
    parser.add_argument("--level", type=str, default="2 3 4 5 6",
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
    image_id = 0

    if len(image_paths) == 1 and not args.batch:
        # Single image mode
        print(f"Processing image: {image_paths[0]}")
        result = process_single_image_multilevel(args, model, image_paths[0], image_id)
        results.append(result)
    else:
        # Batch mode
        print(f"Processing {len(image_paths)} images in batch mode...")

        # Use ThreadPoolExecutor for loading images in parallel
        # Note: The actual inference still runs sequentially on the GPU
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for image_path in image_paths:
                futures.append(executor.submit(process_single_image_multilevel, args, model, image_path, image_id))
                image_id += 1

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
                    f.write(f"Number of levels: {len(result['masks'])}\n")

                    # Add mask statistics
                    for level in result['masks']:
                        if result["masks"][level]:
                            f.write(f"Level: {level}\n")
                            areas = [mask['area'] for mask in result['masks'][level]]
                            f.write(f"\tAverage mask area: {np.mean(areas):.2f}\n")
                            f.write(f"\tLargest mask area: {np.max(areas):.2f}\n")
                            f.write(f"\tSmallest mask area: {np.min(areas):.2f}\n")

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
