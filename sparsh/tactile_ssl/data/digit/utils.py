# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
from glob import glob
import pickle
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

DIGIT_BGS_OBJECTS = {
    "004_sugar_box": 0,
    "005_tomato_soup_can": 1,
    "006_mustard_bottle": 2,
    "021_bleach_cleanser": 3,
    "025_mug": 4,
    "035_power_drill": 0,
    "037_scissors": 5,
    "042_adjustable_wrench": 6,
    "048_hammer": 8,
    "055_baseball": 8,
    "banana": 15,
    "bread": 11,
    "cheese": 16,
    "cookie": 17,
    "corn": 18,
    "lettuce": 17,
    "plum": 11,
    "strawberry": 17,
    "tomato": 16,
}

def get_path_dataset(config, dataset_name):
    return os.path.join(config.path_dataset, dataset_name)


def get_path_images(config, dataset_name):
    path_images = os.path.join(
        config.path_dataset,
        dataset_name,
        config.path_images,
    )
    ext = config.ext_images
    return sorted(glob(os.path.join(path_images, "*" + ext)))


def compute_diff(img1, img2, offset=0.0):
    img1 = np.int32(img1)
    img2 = np.int32(img2)
    diff = img1 - img2
    diff = diff / 255.0 + offset
    diff = np.clip(diff, 0.0, 1.0)
    diff = np.uint8(diff * 255.0)
    return diff


def pil_loader(path, img_bg=None):
    img = cv2.imread(path)
    if img_bg is not None:
        img = compute_diff(img, img_bg, offset=0.5)
    img = Image.fromarray(img)
    return img.convert("RGB")


def get_digit_intrinsics(img_sz):
    yfov = 60
    W, H = img_sz[1], img_sz[0]
    fx = H * 0.5 / np.tan(np.deg2rad(yfov) * 0.5)
    fy = fx
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return K


def get_resize_transform(img_size, is_anytouch=False, is_t3=False, sensor='digit', is_train=False, is_unitouch=False, input_diff=False):
    if is_anytouch:
        if input_diff:
            print("Using input diff transforms")
            t = transforms.Compose(
                [
                    transforms.ToTensor(),  # converts to [0 - 1]
                    transforms.Resize((224, 224), antialias=True),
                    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
                ]
            )
        else:
            print("Using input rgb transforms")
            t = transforms.Compose(
                [
                    transforms.ToTensor(),  # converts to [0 - 1]
                    transforms.Resize((224, 224), antialias=True),
                    # transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
                ]
            )
    elif is_t3:
        if sensor == 'gelsight':
            t = transforms.Compose(
                [   
                    transforms.ToTensor(),  # converts to [0 - 1]
                    transforms.Resize((256, 256), antialias=True),
                    transforms.CenterCrop((224, 224)),
                    transforms.Normalize([0.02172, 0.04152, 0.04152], [0.10158, 0.14423, 0.12185]),
                ]
            )
        else:
            t = transforms.Compose(
                [   
                    transforms.ToTensor(),  # converts to [0 - 1]
                    transforms.Resize((256, 256), antialias=True),
                    transforms.CenterCrop((224, 224)),
                    transforms.Normalize([0.42094, 0.44041, 0.44151], [0.14661, 0.08972, 0.09041]),
                ]
            )
        print("Using T3 transforms")
    elif is_unitouch:
        t = transforms.Compose(
                [   
                    transforms.Resize(
                        224, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    )
                ]
            )
    else:
        t = transforms.Compose(
            [
                transforms.Resize((img_size[0], img_size[1]), antialias=True),
                transforms.ToTensor(),  # converts to [0 - 1]
            ]
        )
    return t


def get_bg_img(config, sensor_type, dataset_name, remove_bg=True):
    bg = None
    if remove_bg:
        path_bgs = config.path_bgs
        if sensor_type == "digit":
            bg_id = DIGIT_BGS_OBJECTS[dataset_name.split("/")[0]]
            bg = cv2.imread("{0}/bg_{1}.jpg".format(path_bgs, bg_id))
        elif sensor_type == "gelsight_mini":
            bg = cv2.imread("{0}/bg_gs.jpg".format(path_bgs))
        else:
            raise ValueError("Unknown sensor type")
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    return bg


# functions to load digit dataset for training representation learning
# dataset is in the form of a pickle file
# images have to be decoded from binary format
def load_pickle_dataset(file_dataset):
    with open(file_dataset, "rb") as f:
        all_frames = pickle.load(f)
    return all_frames

def load_bin_image(io_buf) -> np.ndarray:
    img = Image.open(io.BytesIO(io_buf))
    img = np.array(img)
    return img

def load_sample(img, img_bg=None, enhance=False) -> Image:
    if img_bg is not None:
        img = compute_diff(img, img_bg, offset=0.5)
    if enhance:
        img = enhance_image(img, brightness=280, contrast=200)
    img = Image.fromarray(img)
    return img.convert("RGB")

def load_sample_from_buf(io_buf, img_bg=None, enhance=False, is_anytouch=False, sensor='digit', convert=False, is_train=False, is_force=False):
    img = load_bin_image(io_buf) if isinstance(io_buf, bytes) else io_buf
    assert isinstance(img, np.ndarray), ValueError("Image should be a numpy array")
    assert img.shape[2] == 3, ValueError("Image should have 3 channels")
    if is_anytouch:
        # print('Loading AnyTouch or t3 image')
        h, w, _ = img.shape
        if sensor == 'gelsight':
            # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            # if img_bg is not None:
            #     img_bg = cv2.rotate(img_bg, cv2.ROTATE_90_CLOCKWISE)
            h, w, _ = img.shape
        elif sensor == 'digit' and is_force:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if img_bg is not None:
                img_bg = cv2.rotate(img_bg, cv2.ROTATE_90_COUNTERCLOCKWISE)
            h, w, _ = img.shape
        elif sensor == 'digit' and not is_force:
            h, w, _ = img.shape

        if sensor == 'gelsight' or convert:
            img = img[:,:, ::-1]
            if img_bg is not None:
                img_bg = img_bg[:,:, ::-1]
        elif sensor != 'digit':
            raise ValueError("Unknown sensor type for anytouch: {}".format(sensor))

        if img_bg is not None:
            img = compute_diff(img, img_bg, offset=130.0 / 255.0)
            # print(np.min(img), np.max(img), np.mean(img), np.std(img))
        
    else:
        if img_bg is not None:
            img = compute_diff(img, img_bg, offset=0.5)
        h, w, _ = img.shape
        if sensor == 'gelsight' or convert:
            img = img[:,:, ::-1]
        if h < w:
            if enhance:
                img = enhance_image(img, brightness=280, contrast=200)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w, _ = img.shape
        
        h, w, _ = img.shape
        r = 4/3 # default aspect ratio
        if h/w != r:
            h2, w2 = int(h/r), w
            img = img[int((h-h2)/2):int((h+h2)/2), int((w-w2)/2):int((w+w2)/2)]
    
    img = Image.fromarray(img)
    return img.convert("RGB")

def enhance_image(img, brightness=255, contrast=127):
    # to enhance the diff image (especially for gelsight_mini sensor) 
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255)) 
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127)) 
  
    cal = img 
    if brightness != 0: 
        if brightness > 0: 
            shadow = brightness 
            max = 255
        else: 
            shadow = 0
            max = 255 + brightness 
        alpha = (max - shadow) / 255
        gamma = shadow 
        cal = cv2.addWeighted(img, alpha, img, 0, gamma) 
      
    if contrast != 0: 
        alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast)) 
        gamma = 127 * (1 - alpha) 
        cal = cv2.addWeighted(cal, alpha,  cal, 0, gamma) 
    return cal 


# to load force/slip dataset
def load_dataset_forces_digit_gelsight(config, dataset_name, sensor):
    path_data = os.path.join(config.path_dataset, dataset_name)
    path_force_slip = os.path.join(path_data, "dataset_slip_forces.pkl")
    path_images = sorted(glob(os.path.join(path_data, f"dataset_{sensor}*")))
    dataset_images = []
    for p in path_images:
        with open(p, "rb") as f:
            dataset_images.extend(pickle.load(f))
    with open(path_force_slip, "rb") as f:
        dataset_force_slip = pickle.load(f)

    return dataset_images, dataset_force_slip

def load_dataset_forces(config, dataset_name, sensor):
    return load_dataset_forces_digit_gelsight(config, dataset_name, sensor)


# to load feeling of success dataset
def load_feeling_success(config, dataset_name):
    path_data = os.path.join(config.path_dataset, f"{dataset_name:03d}.pkl")
    with open(path_data, "rb") as file:
        data = pickle.load(file)
    return data

# to load pose estimation dataset
def load_dataset_poses(config, dataset_name, finger_type, t_stride):
    path_data = os.path.join(config.path_dataset, f"{dataset_name}.pkl")

    with open(path_data, "rb") as file:
        data = pickle.load(file)

    idx_max = np.min(
        [
            len(data[f"digit_{finger_type}"]),
            len(data[f"object_{finger_type}_rel_pose_n{t_stride}"]),
        ]
    )

    dataset_digit = data[f"digit_{finger_type}"][:idx_max]
    dataset_poses = data[f"object_{finger_type}_rel_pose_n{t_stride}"][:idx_max]

    return dataset_digit, dataset_poses

# to load textile dataset
def load_textile_dataset(config, dataset_name):
    path_data = os.path.join(config.path_dataset, dataset_name, f"dataset_gelsight.pkl")
    with open(path_data, "rb") as file:
        data = pickle.load(file)
    path_metadata = os.path.join(config.path_dataset, dataset_name, f"metadata.txt")
    with open(path_metadata, "r") as file:
        metadata = file.read()
    return data, metadata