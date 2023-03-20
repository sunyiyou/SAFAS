from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import math
from utils.rotate_crop import crop_rotated_rectangle, inside_rect, vis_rotcrop
from ylib.scipy_misc import imread, imsave
import torchvision.transforms.functional as tf
import PIL
from ylib.scipy_misc import imsave
import glob, os, pickle
import torch
import re
os.environ['PATH'] += os.pathsep + '/usr/bin/ffprobe'

mtcnn = MTCNN(select_largest=False, device='cuda', image_size=224, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()

import ffmpeg

output_folder = 'preposess'


def check_rotation(path_video_file):

    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None

    if 'tags' in meta_dict['streams'][0] and 'rotate' in meta_dict['streams'][0]['tags']:
        if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
            rotateCode = cv2.ROTATE_180
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
        return rotateCode
    else:
        return -1


def proposess_video(video_path, savepath, sample_ratio=5, max=1000):

    v_cap = cv2.VideoCapture(video_path)

    rotateCode = check_rotation(video_path)

    frame_id = 0
    while True:
        success, frame = v_cap.read()
        if not success:
            break
        if frame_id > max:
            break
        if frame_id % sample_ratio == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rotateCode > -1:
                frame = cv2.rotate(frame, rotateCode)

            # Detect face
            batch_boxes, batch_probs, batch_points = mtcnn.detect(frame, landmarks=True)
            if batch_boxes is None:
                continue
            cropped_tsn = mtcnn.extract(frame, batch_boxes, None)
            if cropped_tsn is None:
                continue
            img_embedding = resnet(cropped_tsn.cuda().unsqueeze(0))

            prob = batch_probs[0]
            box = batch_boxes[0].astype(int)
            points = batch_points[0].astype(int)
            box = np.maximum(box, 0)
            points = np.maximum(points, 0)
            cropped = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

            imsave(os.path.join(savepath, f'org_{frame_id:04d}.jpg'), frame)
            imsave(os.path.join(savepath, f'crop_{frame_id:04d}.jpg'), Image.fromarray(cropped))
            info_dict = {
                'box': box,
                'detect_prob': prob,
                'points': points,
                'face_embed': img_embedding.data.cpu().numpy(),
                'frame_id': frame_id
            }
            np.save(os.path.join(savepath, f'infov1_{frame_id:04d}.npy'), info_dict, allow_pickle=True)
        frame_id += 1

def proposess_imglist(imglist, savepath):

    for i, img in enumerate(imglist):
        frame_id = i * 5
        frame = imread(img)
        # Detect face
        try:
            batch_boxes, batch_probs, batch_points = mtcnn.detect(frame, landmarks=True)
        except RuntimeError:
            print(f"error in handling {img}")
            continue
        if batch_boxes is None:
            continue
        cropped_tsn = mtcnn.extract(frame, batch_boxes, None)
        if cropped_tsn is None:
            continue
        img_embedding = resnet(cropped_tsn.cuda().unsqueeze(0))

        prob = batch_probs[0]
        box = batch_boxes[0].astype(int)
        points = batch_points[0].astype(int)
        box = np.maximum(box, 0)
        points = np.maximum(points, 0)
        cropped = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

        imsave(os.path.join(savepath, f'org_{frame_id:04d}.jpg'), frame)
        imsave(os.path.join(savepath, f'crop_{frame_id:04d}.jpg'), Image.fromarray(cropped))
        info_dict = {
            'box': box,
            'detect_prob': prob,
            'points': points,
            'face_embed': img_embedding.data.cpu().numpy(),
            'frame_id': frame_id
        }
        np.save(os.path.join(savepath, f'infov1_{frame_id:04d}.npy'), info_dict, allow_pickle=True)


def run_replay(rootpath):
    outpath = os.path.join(rootpath, output_folder)
    os.makedirs(outpath, exist_ok=True)

    file_list = glob.glob(rootpath + "**/*.mov", recursive=True)
    meta_info_list = []

    savepaths = []
    for i, filepath in enumerate(file_list):

        video_prefix = "_".join(filepath.split("/")[-2:]).split('.')[0]

        if "/real/" in filepath:
            live_or_spoof = 'live'
        elif "/attk/" in filepath:
            live_or_spoof = 'spoof'
        else:
            raise RuntimeError(f"What is wrong? {filepath}")

        if "/train/" in filepath:
            split = 'train'
        elif "/test/" in filepath:
            split = 'test'
        elif "/devel/" in filepath:
            split = 'dev'
        else:
            raise RuntimeError(f"What is wrong? {filepath}")

        name = f"replay_{split}_{live_or_spoof}_{video_prefix}"
        savepath = os.path.join(outpath, name)

        if os.path.exists(savepath) and len(os.listdir(savepath)) > 10:
            savepaths.append(savepath)
            # print(i, f"skip {savepath}")
            continue
        else:
            os.makedirs(savepath, exist_ok=True)
            print(f"make {savepath}")

        proposess_video(filepath, savepath)
        if i % 20 == 0:
            print(f"processed {i} / {len(file_list)}")

        meta_info_list.append((name, live_or_spoof, split))

    return meta_info_list


def run_msu(rootpath):
    outpath = os.path.join(rootpath, output_folder)
    os.makedirs(outpath, exist_ok=True)
    meta_info_list = []

    file_list = glob.glob(rootpath + "**/*.mov", recursive=True)
    file_list += glob.glob(rootpath + "**/*.mp4", recursive=True)

    test_list = np.loadtxt(os.path.join(rootpath, 'test_sub_list.txt')).astype(int)
    train_list = np.loadtxt(os.path.join(rootpath, 'train_sub_list.txt')).astype(int)

    for i, filepath in enumerate(file_list):

        video_prefix = filepath.split("/")[-1].split('.')[0]

        if "/real/" in filepath:
            live_or_spoof = 'live'
        elif "/attack/" in filepath:
            live_or_spoof = 'spoof'
        else:
            raise RuntimeError(f"What is wrong? {filepath}")

        id = int(re.search("client(\d\d\d)", filepath).group(1))

        if id in train_list:
            split = 'train'
        elif id in test_list:
            split = 'test'
        else:
            split = 'dev'

        name = f"replay_{split}_{live_or_spoof}_{video_prefix}"
        savepath = os.path.join(outpath, name)

        if os.path.exists(savepath) and len(os.listdir(savepath)) > 10:
            continue
        else:
            os.makedirs(savepath, exist_ok=True)
            print(f"make {savepath}")

        proposess_video(filepath, savepath)
        if i % 20 == 0:
            print(f"processed {i} / {len(file_list)}")

        meta_info_list.append((name, live_or_spoof, split))


def run_oulu(rootpath):
    outpath = os.path.join(rootpath, output_folder)
    os.makedirs(outpath, exist_ok=True)

    file_list = glob.glob(rootpath + "**/*.avi", recursive=True)
    meta_info_list = []

    for i, filepath in enumerate(file_list):

        video_prefix = filepath.split("/")[-1].split('.')[0]

        if "1.avi" in filepath:
            live_or_spoof = 'live'
        else:
            live_or_spoof = 'spoof'

        if "/Train_files/" in filepath:
            split = 'train'
        elif "/Test_files/" in filepath:
            split = 'test'
        elif "/Dev_files/" in filepath:
            split = 'dev'
        else:
            raise RuntimeError(f"What is wrong? {filepath}")

        name = f"oulu_{split}_{live_or_spoof}_{video_prefix}"
        savepath = os.path.join(outpath, name)
        if os.path.exists(savepath) and len(os.listdir(savepath)) > 10:
            continue
        else:
            os.makedirs(savepath, exist_ok=True)
            print(f"make {savepath}")

        proposess_video(filepath, savepath)
        if i % 20 == 0:
            print(f"processed {i} / {len(file_list)}")

        meta_info_list.append((name, live_or_spoof, split))

    return meta_info_list

def run_casia(rootpath):
    outpath = os.path.join(rootpath, output_folder)
    os.makedirs(outpath, exist_ok=True)

    file_list = glob.glob(rootpath + "**/*.avi", recursive=True)
    meta_info_list = []

    for i, filepath in enumerate(file_list):

        tokens = filepath.split("/")[-2:]
        if 'HR_' not in tokens[-1]:
            tokens[-1] = 'NM_' + tokens[-1]

        video_prefix = "_".join(tokens).split('.')[0]

        if "/1.avi" in filepath or "/2.avi" in filepath or "/HR_1.avi" in filepath:
            live_or_spoof = 'live'
        else:
            live_or_spoof = 'spoof'

        if "/train_release/" in filepath:
            split = 'train'
        elif "/test_release/" in filepath:
            split = 'test'
        else:
            raise RuntimeError(f"What is wrong? {filepath}")

        name = f"casia_{split}_{live_or_spoof}_{video_prefix}"
        savepath = os.path.join(outpath, name)
        if os.path.exists(savepath) and len(os.listdir(savepath)) > 10:
            continue
        else:
            os.makedirs(savepath, exist_ok=True)
            print(f"make {savepath}")

        proposess_video(filepath, savepath)

        if i % 20 == 0:
            print(f"processed {i} / {len(file_list)}")

        meta_info_list.append((name, live_or_spoof, split))

    return meta_info_list


def run_celeba(rootpath, saveroot):
    outpath = os.path.join(saveroot, 'preposess')
    os.makedirs(outpath, exist_ok=True)

    for split in ['train', 'test']:
        path_split = os.path.join(rootpath, split)
        for id in os.listdir(path_split):
            path_split_id = os.path.join(path_split, id)
            for live_or_spoof in ['live', 'spoof']:
                path_split_id_label = os.path.join(path_split_id, live_or_spoof)
                if os.path.exists(path_split_id_label):
                    img_file_list = glob.glob(path_split_id_label + "/*.jpg", recursive=False) + \
                                    glob.glob(path_split_id_label + "/*.png", recursive=False)

                    name = f"celeba_{split}_{live_or_spoof}_{id}"
                    savepath = os.path.join(outpath, name)


                    if os.path.exists(savepath) and len(os.listdir(savepath)) > 4:
                        print(f"skip {savepath}")
                        continue
                    else:
                        os.makedirs(savepath, exist_ok=True)
                        print(f"make {savepath}")

                    proposess_imglist(img_file_list, savepath)


def generate_square_images(image, info, face_width=400, range_scale=3):
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
    min_scale = (256 / face_width) * initial_scale + 0.2

    while True:
        if inside_rect(rect=rect, num_cols=img_cols, num_rows=img_rows):
            break

        if scale < min_scale:
            pad_size = 3000
            image = np.array(tf.pad(PIL.Image.fromarray(image), pad_size, padding_mode='symmetric'))
            center = (center[0] + pad_size, center[1] + pad_size)
            rect = (center, (int(width * scale), int(width * scale)), angle)
            break

        scale = range_scale - round * 0.1
        rect = (center, (int(width * scale), int(width * scale)), angle)
        round += 1


    scaled_face_size = int(face_width * scale / initial_scale)
    image_square_cropped = crop_rotated_rectangle(image=image, rect=rect)
    # vis_rotcrop(image, image_square_cropped, rect, center)
    image_resized = cv2.resize(image_square_cropped, (scaled_face_size, scaled_face_size))
    return image_resized

def generate_square_crop(rootpath, face_width=400):
    files = os.listdir(rootpath)
    for i, file in enumerate(files):
        print(i, file)
        video_clip_path = os.path.join(rootpath, file)
        frames_total = len(glob.glob(os.path.join(video_clip_path, "*.jpg")))

        for image_id in range(frames_total):
            image_name = f"org_{image_id * 5:04d}.jpg"

            # image_name = f"crop_{image_id*5:04d}.jpg"
            info_name = f"infov1_{image_id * 5:04d}.npy"
            # image_name = "{}_{}_scene.jpg".format(video_name, image_id)
            image_path = os.path.join(video_clip_path, image_name)
            info_path = os.path.join(video_clip_path, info_name)

            if not (os.path.exists(image_path) and os.path.exists(info_path)):
                continue
            try:
                info = np.load(info_path, allow_pickle=True).item()
            except pickle.UnpicklingError:
                print(f"*** error with {video_clip_path}")
                break
            square_image_path = os.path.join(video_clip_path, f"square{face_width}_{image_id * 5:04d}.jpg")
            if not os.path.exists(square_image_path):
                image = imread(image_path)
                square_image = generate_square_images(image, info, face_width=face_width)
                imsave(square_image_path, square_image)


if __name__ == '__main__':
    oulu_info = run_oulu(rootpath="datasets/FAS/OULU-NPU/")
    msu_info = run_msu(rootpath="datasets/FAS/MSU-MFSD/")
    casia_info = run_casia(rootpath="datasets/FAS/CASIA_faceAntisp/")
    replay_info = run_replay(rootpath="datasets/FAS/Replay/")
