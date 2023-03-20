from .performance import performances_val
import os
import torch.nn.functional as F
import numpy as np
from PIL import ImageFilter
import random

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        if n > 0:
            self.sum += val * n
            self.cnt += n
            self.avg = self.sum / self.cnt


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def protocol_decoder(protocol):
    if protocol == "O_C_I_to_M":
        data_name_list_train = ["OULU", "CASIA_MFSD", "Replay_attack"]
        data_name_list_test = ["MSU_MFSD"]
    if protocol == "O_C_to_M":
        data_name_list_train = ["OULU", "CASIA_MFSD"]
        data_name_list_test = ["MSU_MFSD"]
    if protocol == "O_to_O":
        data_name_list_train = ["OULU"]
        data_name_list_test = ["OULU"]
    elif protocol == "O_M_I_to_C":
        data_name_list_train = ["OULU", "MSU_MFSD", "Replay_attack"]
        data_name_list_test = ["CASIA_MFSD"]
    elif protocol == "O_C_M_to_I":
        data_name_list_train = ["OULU", "CASIA_MFSD", "MSU_MFSD"]
        data_name_list_test = ["Replay_attack"]
    elif protocol == "I_C_M_to_O":
        data_name_list_train = ["MSU_MFSD", "CASIA_MFSD", "Replay_attack"]
        data_name_list_test = ["OULU"]
    elif protocol == "M_I_to_C":
        data_name_list_train = ["MSU_MFSD", "Replay_attack"]
        data_name_list_test = ["CASIA_MFSD"]
    elif protocol == "M_I_to_O":
        data_name_list_train = ["MSU_MFSD", "Replay_attack"]
        data_name_list_test = ["OULU"]
    return data_name_list_train, data_name_list_test

import torch
from torchvision.transforms.transforms import _setup_size


class NineCrop(torch.nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            tuple of 5 images. Image can be PIL Image or Tensor
        """
        return self.nine_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


    def nine_crop(self, img, size) :
        import numbers
        from torchvision.transforms.functional import _get_image_size, crop, center_crop

        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        elif isinstance(size, (tuple, list)) and len(size) == 1:
            size = (size[0], size[0])

        if len(size) != 2:
            raise ValueError("Please provide only two dimensions (h, w) for size.")

        image_width, image_height = _get_image_size(img)
        crop_height, crop_width = size
        if crop_width > image_width or crop_height > image_height:
            msg = "Requested crop size {} is bigger than input size {}"
            raise ValueError(msg.format(size, (image_height, image_width)))

        tl = crop(img, 0, 0, crop_height, crop_width)
        tm = crop(img, 0, (image_width - crop_width) // 2, crop_height, crop_width)
        tr = crop(img, 0, image_width - crop_width, crop_height, crop_width)

        ml = crop(img, (image_height - crop_height) // 2, 0, crop_height, crop_width)
        # mm = crop(img, (image_height - crop_height) // 2, (image_width - crop_width) // 2, crop_height, crop_width)
        mr = crop(img, (image_height - crop_height) // 2, image_width - crop_width, crop_height, crop_width)

        bl = crop(img, image_height - crop_height, 0, crop_height, crop_width)
        bm = crop(img, image_height - crop_height, (image_width - crop_width) // 2, crop_height, crop_width)
        br = crop(img, image_height - crop_height, image_width - crop_width, crop_height, crop_width)

        center = center_crop(img, [crop_height, crop_width])

        return tl, tm, tr, ml, center, mr, bl, bm, br


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x