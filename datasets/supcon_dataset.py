import os

import PIL.Image
import torch
import pandas as pd
import cv2
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import math
from glob import glob
import re
from utils.rotate_crop import crop_rotated_rectangle, inside_rect, vis_rotcrop
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt

from ylib.scipy_misc import imread, imsave
from .meta import DEVICE_INFOS

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)



def crop_face_from_scene(image, bbox, scale):

    x1,y1,x2,y2=[float(ele) for ele in bbox]
    h = y2-y1
    w = x2-x1
    # y2=y1+w
    # x2=x1+h
    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-h_scale/2.0
    x1=x_mid-w_scale/2.0
    y2=y_mid+h_scale/2.0
    x2=x_mid+w_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),h_img)
    x2=min(math.floor(x2),w_img)
    region=image[y1:y2, x1:x2]
    return region

class FaceDataset(Dataset):
    
    def __init__(self, dataset_name, root_dir, split='train', label=None, transform=None, scale_up=1.1, scale_down=1.0, map_size=32, UUID=-1):
        # self.landmarks_frame = pd.read_csv(info_list, delimiter=",", header=None)
        self.split = split
        self.video_list = os.listdir(root_dir)# list(filter(lambda x: split in x, os.listdir(root_dir)))
        if label is not None and label != 'all':
            self.video_list = list(filter(lambda x: label in x, self.video_list))
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        # self.map_root_dir = root_dir.replace("Train_files", "Depth/Train_files")
        self.transform = transform
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.map_size = map_size
        self.UUID = UUID
        self.face_width = 400

    def __len__(self):
        return len(self.video_list)

    def get_client_from_video_name(self, video_name):
        if 'msu' in self.dataset_name.lower() or 'replay' in self.dataset_name.lower():
            match = re.findall('client(\d\d\d)', video_name)
            if len(match) > 0:
                client_id = match[0]
            else:
                raise RuntimeError('no client')
        elif 'oulu' in self.dataset_name.lower():
            match = re.findall('(\d+)_\d$', video_name)
            if len(match) > 0:
                client_id = match[0]
            else:
                raise RuntimeError('no client')
        elif 'casia' in self.dataset_name.lower():
            match = re.findall('_(\d+)_[H|N][R|M]_\d$', video_name)
            if len(match) > 0:
                client_id = match[0]
            else:
                raise RuntimeError('no client')
        elif 'celeba' in self.dataset_name.lower():
            match = re.findall('_(\d+)$', video_name)
            if len(match) > 0:
                client_id = match[0]
            else:
                raise RuntimeError('no client')
        else:
            raise RuntimeError("no dataset found")
        return client_id
    
    def __getitem__(self, idx):
        # video_name = str(self.landmarks_frame.iloc[idx, 1])
        # spoofing_label = self.landmarks_frame.iloc[idx, 0]
        video_name = self.video_list[idx]
        spoofing_label = int('live' in video_name)
        if self.dataset_name in DEVICE_INFOS:
            if 'live' in video_name:
                patterns = DEVICE_INFOS[self.dataset_name]['live']
            elif 'spoof' in video_name:
                patterns = DEVICE_INFOS[self.dataset_name]['spoof']
            else:
                raise RuntimeError("STH WRONG")
            device_tag = None
            for pattern in patterns:
                if len(re.findall(pattern, video_name)) > 0:
                    if device_tag is not None:
                        raise RuntimeError("Multiple Match")
                    device_tag = pattern
            if device_tag is None:
                raise RuntimeError("No Match")
        else:
            device_tag = 'live' if spoofing_label else 'spoof'

        client_id = self.get_client_from_video_name(video_name)

        image_dir = os.path.join(self.root_dir, video_name)

        if self.split == 'train':
            image_x, info, _ = self.sample_image(image_dir)
            image_x_view1 = self.transform(PIL.Image.fromarray(image_x))
            image_x_view2 = self.transform(PIL.Image.fromarray(image_x))
        else:
            image_x, info, _ = self.sample_image(image_dir)
            image_x_view1 = self.transform(PIL.Image.fromarray(image_x))
            image_x_view2 = image_x_view1
            # import matplotlib.pyplot as plt
            # plt.imshow(image_x);plt.show()

        sample = {"image_x_v1": np.array(image_x_view1),
                  "image_x_v2": np.array(image_x_view2),
                  "label": spoofing_label,
                  "UUID": self.UUID,
                  'device_tag': device_tag,
                  'video': video_name,
                  'client_id': client_id,
                  'points': info['points']}
        return sample

    def sample_image(self, image_dir):
        frames = glob(os.path.join(image_dir, "org_*.jpg"))
        frames_total = len(frames)
        if frames_total == 0:
            raise RuntimeError(f"{image_dir}")


        for temp in range(500):
            if temp > 200:
                image_id = int(re.findall('_(\d+).jpg', frames[0])[0]) // 5
                # print(f"No {image_path} or {info_path} found, use backup id")
            else:
                image_id = np.random.randint(0, frames_total)

            image_name = f"crop_{image_id*5:04d}.jpg"
            # image_name = f"square400_{image_id*5:04d}.jpg"

            info_name = f"infov1_{image_id*5:04d}.npy"
            # image_name = "{}_{}_scene.jpg".format(video_name, image_id)
            image_path = os.path.join(image_dir, image_name)
            info_path = os.path.join(image_dir, info_name)

            if os.path.exists(image_path) and os.path.exists(info_path):
                break

        info = np.load(info_path, allow_pickle=True).item()
        image = imread(image_path)

        return image, info, image_id * 5

    def generate_square_images(self, image, info, range_scale=3):
        points = np.array(info['points'])
        dist = lambda p1, p2: int(np.sqrt(((p1 - p2) ** 2).sum()))
        width = dist(points[0], points[1])
        # height = max(dist(points[1], points[4]), dist(points[0], points[3]))
        center = tuple(points[2])

        angle = math.degrees(math.atan((points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0])))
        rect = (center, (int(width * range_scale), int(width * range_scale)), angle)
        img_rows = image.shape[0]
        img_cols = image.shape[1]

        round = 0
        initial_scale = range_scale
        scale = range_scale
        min_scale = (256 / self.face_width) * initial_scale + 0.3

        while True:
            if inside_rect(rect=rect, num_cols=img_cols, num_rows=img_rows):
                break

            if scale < min_scale:
                pad_size = 300
                image = np.array(tf.pad(PIL.Image.fromarray(image), pad_size, padding_mode='symmetric'))
                center = (center[0] + pad_size, center[1] + pad_size)
                rect = (center, (int(width * scale), int(width * scale)), angle)
                break

            scale = range_scale - round * 0.1
            rect = (center, (int(width * scale), int(width * scale)), angle)
            round += 1

        scaled_face_size = int(self.face_width * scale / initial_scale)
        image_square_cropped = crop_rotated_rectangle(image=image, rect=rect)
        # vis_rotcrop(image, image_square_cropped, rect, center)
        image_resized = cv2.resize(image_square_cropped, (scaled_face_size, scaled_face_size))
        return image_resized

    def get_single_image_x(self, image_dir):
        image, info = self.sample_image(image_dir)
        h_img, w_img = image.shape[0], image.shape[1]
        h_div_w = h_img / w_img
        image_x = cv2.resize(image, (self.face_width, int(h_div_w * self.face_width)))
        return image_x
