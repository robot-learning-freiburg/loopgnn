import os
import sys

from matching.third_party.mast3r.dust3r.dust3r.utils.device import to_numpy

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if base_path not in sys.path:
    sys.path.append(base_path)

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as tfm
from matching import THIRD_PARTY_DIR, BaseMatcher
from matching.utils import add_to_path, to_numpy
from PIL import Image

# add various keypoint model classes
add_to_path(THIRD_PARTY_DIR.joinpath("accelerated_features"))
from matching.im_models import handcrafted
from matching.im_models.lightglue import ALIKED, LightGlueBase, SuperPoint
from modules.xfeat import XFeat


class xFeatExtractor():
    def __init__(self, device="cpu", max_num_keypoints=4096, mode="sparse", *args, **kwargs):
        assert mode in ["sparse", "semi-dense"]

        self.model = XFeat()
        self.max_num_keypoints = max_num_keypoints
        self.mode = mode
        self.device = device

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        # return a [B, C, Hs, W] tensor
        # for sparse/semidense, want [C, H, W]
        while img.ndim < 4:
            img = img.unsqueeze(0)
        return self.model.parse_input(img)

    @staticmethod
    def load_image(path: str | Path, cut_below: int = 464, resize: int | Tuple = None) -> torch.Tensor:
        if isinstance(resize, int):
            resize = (resize, resize)
        img = tfm.ToTensor()(Image.open(path).convert("RGB"))
        # blank out the bottom part of the image
        if cut_below is not None:
            img[:,cut_below:] = 0
        if resize is not None:
            img = tfm.Resize(resize, antialias=True)(img)
        return img

    @torch.inference_mode()
    def forward(self, img):
        if isinstance(img, (str, Path)):
            img = BaseMatcher.load_image(img)

        assert isinstance(img, torch.Tensor), f"Expected torch.Tensor, got {type(img)}"
        img = img.to(self.device)
        
        img = self.preprocess(img)

        if self.mode == "semi-dense":
            output = self.model.detectAndComputeDense(img, top_k=self.max_num_keypoints)
        elif self.mode in ["sparse", "lighterglue"]:
            output = self.model.detectAndCompute(img, top_k=self.max_num_keypoints)[0]

        else:
            raise ValueError(f'unsupported mode for xfeat: {self.mode}. Must choose from ["sparse", "semi-dense"]')

        return (
            output["keypoints"].squeeze(),
            output["descriptors"].squeeze(),
        )
    

class ORBExtractor():
    def __init__(self, device="cpu", max_num_keypoints=512,  *args, **kwargs):
        self.det_descr = cv2.ORB_create(max_num_keypoints)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.k_neighbors = 2

    @staticmethod
    def preprocess(im_tensor: torch.Tensor) -> np.ndarray:
        # convert tensors to np 255-based for openCV
        im_arr = to_numpy(im_tensor).transpose(1, 2, 0)
        im = cv2.cvtColor(im_arr, cv2.COLOR_RGB2GRAY)
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

        return im

    @torch.inference_mode()
    def forward(self, img):

        img = self.preprocess(img)

        # find the keypoints and descriptors with ORB
        keypoints, descriptors = self.det_descr.detectAndCompute(img, None)

        kp_tensor = torch.from_numpy(np.array([kp.pt for kp in keypoints], dtype=np.float32))
        desc_tensor = torch.from_numpy(descriptors)
        
        return (
            kp_tensor,
            desc_tensor
        )    


class SIFTExtractor():
    def __init__(self, device="cpu", max_num_keypoints=2048,  *args, **kwargs):
        self.det_descr = cv2.SIFT_create(max_num_keypoints)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.k_neighbors = 2

    @staticmethod
    def preprocess(im_tensor: torch.Tensor) -> np.ndarray:
        # convert tensors to np 255-based for openCV
        im_arr = to_numpy(im_tensor).transpose(1, 2, 0)
        im = cv2.cvtColor(im_arr, cv2.COLOR_RGB2GRAY)
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

        return im

    @torch.inference_mode()
    def forward(self, img):

        img = self.preprocess(img)

        # find the keypoints and descriptors with ORB
        keypoints, descriptors = self.det_descr.detectAndCompute(img, None)
        kp_tensor = torch.from_numpy(np.array([kp.pt for kp in keypoints], dtype=np.float32))
        desc_tensor = torch.from_numpy(descriptors)

        return (
            kp_tensor,
            desc_tensor
        )



class SuperpointExtractor(LightGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.model = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.max_num_keypoints = max_num_keypoints
        self.device = device

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        # return a [B, C, Hs, W] tensor
        # for sparse/semidense, want [C, H, W]
        while img.ndim < 4:
            img = img.unsqueeze(0)
        return img

    @staticmethod
    def load_image(path: str | Path, cut_below: int = 464, resize: int | Tuple = None) -> torch.Tensor:
        if isinstance(resize, int):
            resize = (resize, resize)
        img = tfm.ToTensor()(Image.open(path).convert("RGB"))
        # blank out the bottom part of the image
        img[:,cut_below:] = 0
        if resize is not None:
            img = tfm.Resize(resize, antialias=True)(img)
        return img

    @torch.inference_mode()
    def forward(self, img):
        if isinstance(img, (str, Path)):
            img = LightGlueBase.load_image(img)

        assert isinstance(img, torch.Tensor), f"Expected torch.Tensor, got {type(img)}"
        img = img.to(self.device)
        
        img = self.preprocess(img)

        output = self.model.forward({"image": img})

        return (
            output["keypoints"].squeeze(),
            output["descriptors"].squeeze(),
        )


class ALIKEDExtractor(LightGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.model = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.max_num_keypoints = max_num_keypoints
        self.device = device

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        # return a [B, C, Hs, W] tensor
        # for sparse/semidense, want [C, H, W]
        while img.ndim < 4:
            img = img.unsqueeze(0)
        return img

    @staticmethod
    def load_image(path: str | Path, cut_below: int = 464, resize: int | Tuple = None) -> torch.Tensor:
        if isinstance(resize, int):
            resize = (resize, resize)
        img = tfm.ToTensor()(Image.open(path).convert("RGB"))
        # blank out the bottom part of the image
        img[:,cut_below:] = 0
        if resize is not None:
            img = tfm.Resize(resize, antialias=True)(img)
        return img

    @torch.inference_mode()
    def forward(self, img):
        if isinstance(img, (str, Path)):
            img = LightGlueBase.load_image(img)

        assert isinstance(img, torch.Tensor), f"Expected torch.Tensor, got {type(img)}"
        img = img.to(self.device)
        
        img = self.preprocess(img)

        output = self.model.forward({"image": img})

        return (
            output["keypoints"].squeeze(),
            output["descriptors"].squeeze(),
        )

    


def get_extractor(extractor_name="xfeat", device="cpu", max_num_keypoints=2048, *args, **kwargs):
    if "xfeat" in extractor_name:
        kwargs["mode"] = "semi-dense" if "star" in extractor_name else "sparse"
        return xFeatExtractor(device, max_num_keypoints=max_num_keypoints, *args, **kwargs)
    elif "orb" in extractor_name:
        return ORBExtractor(device, max_num_keypoints, *args, **kwargs)
    elif "sift" in extractor_name:
        return handcrafted.SIFTExtractor(device, max_num_keypoints, *args, **kwargs)
    elif "tiny-roma" in extractor_name:
        from matching.im_models import roma
        return roma.TinyRomaMatcher(device, max_num_keypoints, *args, **kwargs)
    elif "superpoint" in extractor_name:
        from matching.im_models import lightglue
        return lightglue.SuperpointExtractor(device, max_num_keypoints, *args, **kwargs)
    elif "aliked" in extractor_name:
        from matching.im_models import lightglue
        return lightglue.ALIKEDExtractor(device, max_num_keypoints, *args, **kwargs)
    