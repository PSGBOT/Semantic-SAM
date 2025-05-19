import torch
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from utils.arguments import load_opt_from_config_file

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
    from tasks import interactive_infer_image_idino_m2m_auto

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
