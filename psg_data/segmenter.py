import numpy as np
import os
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp
from functools import partial

from psg_data.model import load_model, inference_multilevel
from utils.psg_utils.image import load_image_from_path, find_images
from utils.psg_utils.mask import visualize_masks, save_masks, discard_subseg


class MultiLevelSegmenter:
    """
    A class for performing multi-level segmentation using Semantic SAM. Generate the level seg dataset.
    """

    def __init__(self, model_size="L", output_dir="./output", visualize=False,
                 save_masks=False, workers=1):
        """
        Initialize the MultiLevelSegmenter.

        Args:
            model_size (str): Model size: T (tiny) or L (large)
            output_dir (str): Directory to save output masks
            visualize (bool): Whether to visualize masks
            save_masks (bool): Whether to save individual masks as binary images
            workers (int): Number of worker threads for batch processing
        """
        self.model_size = model_size
        self.output_dir = output_dir
        self.visualize = visualize
        self.save_masks = save_masks
        self.workers = workers
        self.model = None

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self):
        """
        Load the Semantic SAM model.

        Returns:
            The loaded model
        """
        print(f"Loading model (size: {self.model_size})...")
        self.model = load_model(model_size=self.model_size)
        return self.model

    def process_image(self, image_path, levels="2 3 4 5 6", image_id=0):
        """
        Process a single image with multiple levels.

        Args:
            image_path (str): Path to the image file
            levels (str): Segmentation levels to use
            image_id (int): ID for the image (used for output directory naming)

        Returns:
            dict: Results containing image path and masks
        """
        if self.model is None:
            self.load_model()

        # Create image-specific output directory
        image_name = f"id_{image_id}" if image_id is not None else os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(self.output_dir, image_name)

        # Load image
        image = load_image_from_path(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return {"image_path": image_path, "success": False, "masks": []}

        # Run inference
        try:
            multilevel_masks = inference_multilevel(model=self.model, image=image, level=levels)

            for level in multilevel_masks:
                multilevel_masks[level] = discard_subseg(multilevel_masks[level])
                # Create output directory if needed
                level_dir = os.path.join(image_output_dir, level)
                if self.save_masks or self.visualize:
                    os.makedirs(level_dir, exist_ok=True)

                # Save masks if requested
                if self.save_masks:
                    save_masks(multilevel_masks[level], level_dir)

                # Visualize masks if requested
                if self.visualize:
                    vis_path = os.path.join(level_dir, "visualization.png") if self.save_masks else None
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

    def process_batch(self, image_paths, levels="2 3 4 5 6"):
        """
        Process multiple images in batch mode using multiple GPUs.

        Args:
            image_paths (list): List of image paths to process
            levels (str): Segmentation levels to use

        Returns:
            list: Results for each processed image
        """
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus <= 1:
            print(f"Only {num_gpus} GPU detected. Using ThreadPoolExecutor instead.")
            return self._process_batch_threads(image_paths, levels)

        print(f"Processing {len(image_paths)} images using {num_gpus} GPUs...")

        # Split images across GPUs
        chunks = self._split_list(image_paths, num_gpus)

        # Create process pool with one process per GPU
        mp.set_start_method('spawn', force=True)
        with mp.Pool(processes=num_gpus) as pool:
            # Create partial function with fixed arguments
            process_chunk_fn = partial(
                self._process_chunk,
                levels=levels
            )

            # Map chunks to processes (each with a different GPU)
            chunk_results = pool.starmap(
                process_chunk_fn,
                [(chunk, gpu_id) for gpu_id, chunk in enumerate(chunks)]
            )

        # Combine results from all processes
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)

        return results

    def _process_chunk(self, image_paths_chunk, gpu_id, levels):
        """
        Process a chunk of images on a specific GPU.

        Args:
            image_paths_chunk (list): Chunk of image paths to process
            gpu_id (int): GPU ID to use
            levels (str): Segmentation levels to use

        Returns:
            list: Results for processed images in this chunk
        """
        # Set device for this process
        torch.cuda.set_device(gpu_id)

        # Load model on this GPU
        model = load_model(model_size=self.model_size)

        results = []
        for image_id, image_path in enumerate(tqdm(image_paths_chunk,
                                                  desc=f"GPU {gpu_id}")):
            # Create image-specific output directory
            image_name = f"id_{image_id}" if image_id is not None else os.path.splitext(os.path.basename(image_path))[0]
            image_output_dir = os.path.join(self.output_dir, image_name)

            # Load image
            image = load_image_from_path(image_path)
            if image is None:
                results.append({"image_path": image_path, "success": False, "masks": []})
                continue

            # Run inference
            try:
                multilevel_masks = inference_multilevel(model=model, image=image, level=levels)

                for level in multilevel_masks:
                    multilevel_masks[level] = discard_subseg(multilevel_masks[level])
                    # Create output directory if needed
                    level_dir = os.path.join(image_output_dir, level)
                    if self.save_masks or self.visualize:
                        os.makedirs(level_dir, exist_ok=True)

                    # Save masks if requested
                    if self.save_masks:
                        save_masks(multilevel_masks[level], level_dir)

                    # Visualize masks if requested
                    if self.visualize:
                        vis_path = os.path.join(level_dir, "visualization.png") if self.save_masks else None
                        visualize_masks(multilevel_masks[level], save_path=vis_path)

                results.append({
                    "image_path": image_path,
                    "success": True,
                    "masks": multilevel_masks,
                    "output_dir": image_output_dir
                })

            except Exception as e:
                print(f"Error processing {image_path} on GPU {gpu_id}: {e}")
                results.append({"image_path": image_path, "success": False, "masks": None})

        return results

    def _process_batch_threads(self, image_paths, levels="2 3 4 5 6"):
        """Original thread-based implementation for single GPU"""
        if self.model is None:
            self.load_model()

        results = []
        print(f"Processing {len(image_paths)} images in batch mode...")

        # Use ThreadPoolExecutor for loading images in parallel
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for image_id, image_path in enumerate(image_paths):
                futures.append(executor.submit(
                    self.process_image, image_path, levels, image_id))

            # Process results as they complete
            for future in tqdm(futures, total=len(image_paths)):
                result = future.result()
                results.append(result)

        return results

    def _split_list(self, lst, n):
        """Split a list into n roughly equal chunks"""
        k, m = divmod(len(lst), n)
        return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

    def process_directory(self, input_dir, levels="2 3 4 5 6", generate_summary=False):
        """
        Process all images in a directory.

        Args:
            input_dir (str): Directory containing images to process
            levels (str): Segmentation levels to use
            generate_summary (bool): Whether to generate a summary of results

        Returns:
            list: Results for each processed image
        """
        # Find images to process
        image_paths = find_images(input_dir)

        if not image_paths:
            print(f"No images found at {input_dir}")
            return []

        # Process images
        results = self.process_batch(image_paths, levels)

        # Generate summary if requested
        if generate_summary and results:
            self._generate_summary(results)

        return results

    def _generate_summary(self, results):
        """
        Generate a summary of processing results.

        Args:
            results (list): List of processing results
        """
        summary_path = os.path.join(self.output_dir, "summary.txt")
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

    def Generate(self, image_path, levels):
        if os.path.isdir(image_path):
            results = segmenter.process_directory(image_path, levels, True)
        else:
            results = [segmenter.process_image(image_path, levels)]
        return results

    def SaveResult_dict(self, results, dir):
        """
        Save the results dictionary to a JSON file.

        Args:
            results (dict): The results dictionary to save.
            dir (str): The directory to save the file in.

        Returns:
            None
        """
        import json
        import os

        # Create directory if it doesn't exist
        os.makedirs(dir, exist_ok=True)

        # Create a serializable version of the results
        serializable_results = []

        for result in results:
            # Create a copy of the result to modify
            serializable_result = result.copy()

            # Instead of serializing mask data, store paths to mask images
            if result["success"] and "masks" in result:
                # Store only metadata and paths instead of actual mask data
                mask_metadata = {}

                for level, masks in result["masks"].items():
                    level_metadata = []

                    # Get the level directory path
                    level_dir = os.path.join(result.get("output_dir", ""), level)

                    for i, mask in enumerate(masks):
                        # Create metadata entry with important info but no binary data
                        mask_info = {
                            "id": i,
                            "area": mask.get("area", 0),
                            "bbox": mask.get("bbox", []),
                            "score": mask.get("stability_score", 0),
                            "mask_path": os.path.join(level_dir, f"mask_{i}.png") if self.save_masks else None
                        }
                        level_metadata.append(mask_info)

                    mask_metadata[level] = level_metadata

                serializable_result["masks"] = mask_metadata

                # Add visualization path if available
                if self.visualize and self.save_masks:
                    serializable_result["visualizations"] = {
                        level: os.path.join(os.path.join(result.get("output_dir", ""), level), "visualization.png")
                        for level in result["masks"]
                    }

            serializable_results.append(serializable_result)

        # Save to JSON file
        output_file = os.path.join(dir, "segmentation_results.json")
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {output_file}")





# Example usage
if __name__ == "__main__":
    import argparse

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

    args = parse_args()

    # Initialize the segmenter
    segmenter = MultiLevelSegmenter(
        model_size=args.model_size,
        output_dir=args.output_dir,
        visualize=args.visualize,
        save_masks=args.save_masks,
        workers=args.workers
    )

    # Process images
    result = segmenter.Generate(args.input, args.level)
    segmenter.SaveResult_dict(result, args.output_dir)
    print("Done!")
