import os
import glob
from PIL import Image

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
        # Ensure the image is in RGB format, so it always has 3 channels
        if img.mode != 'RGB':
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
