import os
import pandas as pd
from PIL import Image, ImageOps

from torch.utils.data import Dataset
import torchvision.transforms as T


class ImageSet(Dataset):
    """
    Dataset class for test/inference - loads only images.

    Used in test.py and generate.py where only images are needed for feature
    extraction and score prediction.
    """

    def __init__(
        self, data_dir, annotations_file, transform=None, path_col_name="path", **kwargs
    ):
        self.annotations_df = pd.read_csv(annotations_file)
        self.files = self.annotations_df[path_col_name].values
        self.data_dir = data_dir
        self.xform = transform

    def __len__(self):
        return len(self.files)

    def process_image(self, fpath):
        fpath = os.path.join(self.data_dir, fpath)
        image = Image.open(fpath)
        image = ImageOps.exif_transpose(image)  # Handle EXIF orientation
        image = image.convert("RGB")
        if self.xform is not None:
            image = self.xform(image)
        return image

    def __getitem__(self, index):
        fpath = self.files[index]
        image = self.process_image(fpath)
        return image


class ImageRecognizabilitySet(Dataset):
    """
    Dataset class for training - loads images with CCS and CCAS labels.

    Used in train.py for TransFIRA recognizability prediction network training.
    Returns (image, ccs, ccas) tuples.
    """

    def __init__(
        self,
        data_dir,
        annotations_file,
        transform=None,
        path_col_name="path",
        ccs_col_name="ccs",
        ccas_col_name="ccas",
        **kwargs
    ):
        self.annotations_df = pd.read_csv(annotations_file)
        self.files = self.annotations_df[path_col_name].values
        self.ccs = self.annotations_df[ccs_col_name].values
        self.ccas = self.annotations_df[ccas_col_name].values
        self.data_dir = data_dir
        self.xform = transform

    def __len__(self):
        return len(self.files)

    def process_image(self, fpath):
        fpath = os.path.join(self.data_dir, fpath)
        image = Image.open(fpath)
        image = ImageOps.exif_transpose(image)  # Handle EXIF orientation
        image = image.convert("RGB")
        if self.xform is not None:
            image = self.xform(image)
        return image

    def __getitem__(self, index):
        fpath = self.files[index]
        image = self.process_image(fpath)
        ccs = self.ccs[index]
        ccas = self.ccas[index]
        return image, ccs, ccas


def image_pipeline(
    p=0, u=[0.4951, 0.4030, 0.3603], img_std=1.0, imgdim=112, imgcrop=112
):
    """
    Return image preprocessing pipeline for face recognition.

    Args:
        p: Probability for random horizontal flip augmentation
        u: RGB normalization mean values
        img_std: RGB normalization standard deviation
        imgdim: Target image dimension after cropping
        imgcrop: Initial resize dimension before center crop
    """
    xform = T.Compose(
        [
            T.Resize(imgcrop),
            T.CenterCrop((imgdim, imgdim)),
            T.RandomHorizontalFlip(p),
            # T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            T.ToTensor(),
            T.Normalize(mean=u, std=[img_std, img_std, img_std]),
        ]
    )

    return xform
