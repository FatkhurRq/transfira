# Third party modules
import torch
import torchvision.transforms as T
import onnx
from onnx2torch import convert

# Local modules
from train_utils.torch_utils import image_pipeline


def get_backbone(cfg):
    """
    Returns a backbone and transform consistent with the config options
    """

    if cfg["backbone_type"] == "onnx":
        onnx_model = onnx.load(cfg["model_path"])
        model = convert(onnx_model).train().cuda()
        cfg["xform"] = image_pipeline(
            p=0,
            u=cfg["aug"]["train_rgb_norm"],
            imgdim=cfg["aug"]["resize_dim"],
            imgcrop=cfg["aug"]["crop_dim"],
            img_std=cfg["aug"]["test_rgb_std"],
        )
        outdim = cfg["embedding_size"]
    else:
        raise ValueError("Invalid backbone type")

    return model, outdim, cfg
