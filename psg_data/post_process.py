import numpy as np
import os
import json
import torch
from tqdm import tqdm
from utils.psg_utils.mask import load_masks, apply_mask, discard_submask
from PIL import Image
import shutil


class PartlevelPostprocesser:
    """
    A class for postprocess the level-seg dataset into part-seg dataset. Masks are aligned in a hyerichical way.
    """

    def __init__(self, output_dir="./output", visualize=False):
        self.dataset_output_dir = output_dir
        self.visualize = visualize

        self.level_seg_dataset = {}

        # Create output directory
        os.makedirs(self.dataset_output_dir, exist_ok=True)

    def load_level_seg_dataset(self, dataset_json_path):
        """
        Load the level segmentation dataset from a JSON file.

        Args:
            dataset_json_path (str): Path to the JSON file containing the level segmentation dataset.

        Returns:
            dict: The loaded dataset or None if loading failed.
        """
        try:
            if not os.path.exists(dataset_json_path):
                print(f"Dataset file not found: {dataset_json_path}")
                return None

            with open(dataset_json_path, "r") as f:
                dataset = json.load(f)

            print(f"Loaded level segmentation dataset from {dataset_json_path}")
            print(f"Dataset contains {len(dataset)} images")

            self.level_seg_dataset = dataset
            return self.level_seg_dataset

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def _save_masks(self, masks, output_dir):
        """
        Save individual masks as binary images.

        Args:
            masks: List of mask dictionaries (can be numpy arrays or PyTorch tensors)
            output_dir: Directory to save masks
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, mask in enumerate(masks):
            # Convert to numpy if it's a tensor
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()

            # Convert boolean mask to uint8 (0 and 255)
            mask_img = mask.astype(np.uint8) * 255

            # Create a PIL image and save
            mask_pil = Image.fromarray(mask_img)
            mask_path = os.path.join(output_dir, f"mask{i}.png")
            mask_pil.save(mask_path)

    def _load_level_seg_masks(self, image_id, level=2):
        dir = os.path.join(
            self.level_seg_dataset[image_id]["output_dir"], f"level{level}"
        )
        return load_masks(dir)

    def process_mask(self, image_id, levels=[2, 3, 4, 5, 6]):
        multilevel_masks = {}
        for level in levels:
            multilevel_masks[level] = self._load_level_seg_masks(image_id, level)

        output_dir = os.path.join(self.dataset_output_dir, f"id {image_id}")
        cur_level = levels[0]
        top_level = levels[-1]

        self._process_mask_iterative(
            image_id,
            output_dir,
            multilevel_masks[cur_level],
            multilevel_masks[cur_level + 1],
            multilevel_masks,
            cur_level,
            top_level,
        )
        self._prune_masks(output_dir)

    def _process_mask_iterative(
        self,
        image_id,
        output_dir,
        parent_masks,
        child_masks,
        total_masks,
        level=2,
        top_level=6,
    ):
        self._save_masks(parent_masks, output_dir)
        if level == top_level:
            return
        valid_children = apply_mask(parent_masks, child_masks)
        for i, per_parent_children in enumerate(list(valid_children)):
            final_children = discard_submask(per_parent_children)
            if len(final_children) == 1:
                # skip this level
                final_children = [parent_masks[i]]
            child_output_dir = os.path.join(output_dir, f"mask{i}")
            self._process_mask_iterative(
                image_id,
                child_output_dir,
                final_children,
                total_masks[level + 1],
                total_masks,
                level + 1,
                top_level,
            )
        return

    def _prune_masks(self, image_dir):
        """
        If the mask{id} folder only has one mask png, delete the mask png and move the content in mask{id}/mask0/ into mask{id}, and delete mask{id}/mask0/

        Args:
            image_dir (str): Directory to prune masks from
        """
        # Check if the directory exists
        if not os.path.exists(image_dir):
            print(f"Directory not found: {image_dir}")
            return

        # Get all mask directories
        mask_dirs = [
            d
            for d in os.listdir(image_dir)
            if d.startswith("mask") and os.path.isdir(os.path.join(image_dir, d))
        ]

        # Process each mask directory
        for mask_dir in mask_dirs:
            mask_path = os.path.join(image_dir, mask_dir)
            self._prune_masks(mask_path)  # Recursively process subdirectories

        # Get all mask PNG files in the current directory
        mask_files = [
            f
            for f in os.listdir(image_dir)
            if f.endswith(".png") and os.path.isfile(os.path.join(image_dir, f))
        ]

        # Check if there's only one mask PNG and a mask0 directory
        mask0_dir = os.path.join(image_dir, "mask0")
        if (
            len(mask_files) == 1
            and os.path.exists(mask0_dir)
            and os.path.isdir(mask0_dir)
        ):
            try:
                # rename subfolder `mask0/` into `mask0_tmp/`
                mask0_tmp_dir = os.path.join(image_dir, "mask0_tmp")
                os.rename(mask0_dir, mask0_tmp_dir)

                # delete all other files in current folder
                for file in mask_files:
                    os.remove(os.path.join(image_dir, file))

                # move the content in `mask0_tmp` into current folder
                for item in os.listdir(mask0_tmp_dir):
                    src = os.path.join(mask0_tmp_dir, item)
                    dst = os.path.join(image_dir, item)
                    shutil.move(src, dst)

                # delete `mask0_tmp`
                os.rmdir(mask0_tmp_dir)
            except Exception as e:
                print(f"Error pruning masks in {image_dir}: {e}")
        elif len(mask_files) <= 1 and not (
            os.path.exists(mask0_dir) and os.path.isdir(mask0_dir)
        ):
            try:
                # Remove directory and all its contents
                shutil.rmtree(image_dir)
            except Exception as e:
                print(f"Error removing directory {image_dir}: {e}")

    def _generate_part_dataset_info(self):
        """
        Based on the processed file structure, generate the dataset info as `self.part_seg_dataset`,
        for each mask, generate and store the bbox information
        """
        self.part_seg_dataset = {}

        # Walk through the output directory
        for image_dir in os.listdir(self.dataset_output_dir):
            image_path = os.path.join(self.dataset_output_dir, image_dir)

            # Skip if not a directory or doesn't start with "id"
            if not os.path.isdir(image_path) or not image_dir.startswith("id"):
                continue

            image_id = image_dir.split(" ")[1]  # Extract ID from "id X"
            self.part_seg_dataset[f"id {image_id}"] = {"masks": {}}

            # Process masks recursively
            self._process_masks_recursive(
                image_path, self.part_seg_dataset[f"id {image_id}"]["masks"], ""
            )

        # Save the dataset to a JSON file
        output_path = os.path.join(self.dataset_output_dir, "part_seg_dataset.json")
        try:
            with open(output_path, "w") as f:
                json.dump(self.part_seg_dataset, f, indent=2)
            print(f"Part segmentation dataset saved to {output_path}")
        except Exception as e:
            print(f"Error saving dataset: {e}")

        return self.part_seg_dataset

    def _process_masks_recursive(self, dir_path, dataset_dict, parent_path):
        """
        Recursively process masks in a directory and its subdirectories

        Args:
            dir_path (str): Path to the directory
            dataset_dict (dict): Dictionary to store mask information
            parent_path (str): Path from root to parent directory
        """
        # Get all mask PNG files in the current directory
        mask_files = [
            f
            for f in os.listdir(dir_path)
            if f.endswith(".png") and os.path.isfile(os.path.join(dir_path, f))
        ]

        # Process each mask file
        for mask_file in mask_files:
            mask_path = os.path.join(dir_path, mask_file)
            try:
                # Load mask
                mask_img = Image.open(mask_path)
                mask_array = np.array(mask_img)

                # Calculate bounding box
                if mask_array.sum() > 0:
                    # Find non-zero pixels
                    rows = np.any(mask_array, axis=1)
                    cols = np.any(mask_array, axis=0)
                    y1, y2 = np.where(rows)[0][[0, -1]]
                    x1, x2 = np.where(cols)[0][[0, -1]]

                    # Calculate area
                    area = float(mask_array.sum()) / 255

                    # Store mask info
                    mask_id = os.path.splitext(mask_file)[0]  # Remove .png extension
                    mask_info = {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "area": area,
                        "path": os.path.join(parent_path, mask_file),
                    }
                    dataset_dict[mask_id] = mask_info
            except Exception as e:
                print(f"Error processing mask {mask_path}: {e}")

        # Process subdirectories
        for subdir in os.listdir(dir_path):
            subdir_path = os.path.join(dir_path, subdir)
            if os.path.isdir(subdir_path) and subdir.startswith("mask"):
                # Create entry for this mask directory
                dataset_dict[subdir]["children"] = {}

                # Process masks in this subdirectory
                new_parent_path = os.path.join(parent_path, subdir)
                self._process_masks_recursive(
                    subdir_path, dataset_dict[subdir]["children"], new_parent_path
                )

    def Process(self, levels):
        for i in tqdm(range(len(self.level_seg_dataset))):
            self.process_mask(i, levels)
            shutil.copyfile(
                self.level_seg_dataset[i]["image_path"],
                os.path.join(self.dataset_output_dir, f"id {i}.png"),
            )
        self._generate_part_dataset_info()


# example usage
if __name__ == "__main__":
    import argparse

    def parse_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="PSG Post Processor")

        parser.add_argument(
            "--input",
            type=str,
            default="./examples/truck.jpg",
            help="Path to input image or directory containing images",
        )
        parser.add_argument(
            "--level",
            type=str,
            default="2 3 4 5 6",
            help="Segmentation level (1-6) or 'All Prompt'",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="./output",
            help="Directory to save output masks",
        )

        return parser.parse_args()

    args = parse_args()
    levels = args.level
    if isinstance(levels, str):
        levels = [int(level_str) for level_str in levels.split(" ")]

    processor = PartlevelPostprocesser(args.output_dir)

    processor.load_level_seg_dataset(args.input)

    processor.Process(levels)

    print("Done!")
