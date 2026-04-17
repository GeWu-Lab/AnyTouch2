# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
import time
import cv2
from omegaconf import DictConfig
import numpy as np
from PIL import Image

import torch
import matplotlib.pyplot as plt

from sparsh.tactile_ssl.data.digit.utils import (
    load_sample,
    get_resize_transform,
)

from digit_interface.digit import Digit


class DemoForceFieldData:
    def __init__(
        self,
        config: DictConfig,
        digit_serial: str,
        gelsight_device_id: int,
    ):

        super().__init__()
        self.config = config
        self.sensor = self.config.sensor
        self.digit_serial = digit_serial
        self.gelsight_device_id = gelsight_device_id
        self.enhance_diff_img = True if self.sensor == "gelsight_mini" else False

        self.remove_bg = (
            self.config.remove_bg if hasattr(self.config, "remove_bg") else False
        )
        self.out_format = self.config.out_format  # if output video
        assert self.out_format in [
            "video",
            "concat_ch_img",
            "single_image",
        ], ValueError(
            "out_format should be 'video' or 'concat_ch_img' or 'single_image'"
        )

        frame_stride = self.config.frame_stride
        self.num_frames = (
            1 if self.out_format == "single_image" else self.config.num_frames
        )
        self.frames_concat_idx = np.arange(
            0, self.num_frames * frame_stride, frame_stride
        )

        # load dataset
        self.loader = load_sample

        # transforms
        self.img_sz = self.config.transforms.resize
        self.transform_resize = get_resize_transform(self.img_sz)

        # tactile window. Make a FIFO buffer with length 5 using deque
        self.tactile_window = deque(maxlen=6)

        # connect to digit sensor
        self.fps = 30.0
        if self.sensor == "digit":
            self.touch_sensor = self.connect_digit()
            self._init_digit_sensor()
            self.bg = self.get_digit_image()
        elif "gelsight" in self.sensor:
            self.touch_sensor = self.connect_gelsight()
            self._init_gelsight_sensor()
            self.bg = self.get_gelsight_image()
        else:
            raise ValueError("Sensor not supported")

    def connect_gelsight(self):
        assert self.gelsight_device_id is not None, ValueError("Gelsight device id is required")
        return cv2.VideoCapture(self.gelsight_device_id)

    def _init_gelsight_sensor(self):
        for i in range(100):
            _ = self.get_gelsight_image()
            time.sleep(1 / self.fps)

    def connect_digit(self):
        # Connect to a Digit device with serial number with friendly name
        assert self.digit_serial is not None, ValueError("Digit serial number is required")
        digit_sensor = Digit(self.digit_serial, "Digit")
        digit_sensor.connect()
        digit_sensor.set_intensity(Digit.LIGHTING_MAX)
        # Change DIGIT resolution to QVGA
        qvga_res = Digit.STREAMS["QVGA"]
        digit_sensor.set_resolution(qvga_res)
        fps_30 = Digit.STREAMS["QVGA"]["fps"]["30fps"]
        digit_sensor.set_fps(fps_30)
        # Print device info
        print(digit_sensor.info())
        return digit_sensor

    def _init_digit_sensor(self):
        for i in range(100):
            _ = self.get_digit_image()
            time.sleep(1 / self.fps)

    def _process_image(self, tactile_image):
        tactile_image = cv2.cvtColor(tactile_image, cv2.COLOR_BGR2RGB)
        h, w, _ = tactile_image.shape
        if h < w:
            tactile_image = cv2.rotate(tactile_image, cv2.ROTATE_90_CLOCKWISE)

        h, w, _ = tactile_image.shape
        r = 4/3 # default aspect ratio
        if h/w != r:
            h2, w2 = int(h/r), w
            tactile_image = tactile_image[int((h-h2)/2):int((h+h2)/2), int((w-w2)/2):int((w+w2)/2)]
        return tactile_image

    def get_digit_image(self):
        tactile_image = self.touch_sensor.get_frame()
        tactile_image = cv2.flip(tactile_image, 1)
        tactile_image = self._process_image(tactile_image)
        self.tactile_window.append(tactile_image)
        return tactile_image


    def get_gelsight_image(self):
        ret, tactile_image = self.touch_sensor.read()
        tactile_image = cv2.flip(tactile_image, 0)
        tactile_image = cv2.resize(tactile_image, (320, 240))
        tactile_image = self._process_image(tactile_image)
        self.tactile_window.append(tactile_image)
        return tactile_image

    def _plot_tactile_clip(self, clip):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, self.num_frames, figsize=(20, 5))
        for i in range(self.num_frames):
            axs[i].imshow(clip[i].permute(1, 2, 0))
            axs[i].axis("off")
        plt.savefig("tactile_clip.png")
        plt.close()

    def get_model_inputs(self):
        if self.sensor == "digit":
            img = self.get_digit_image()
        elif "gelsight" in self.sensor:
            img = self.get_gelsight_image()
        else:
            raise ValueError("Sensor not supported")
        
        images, images_bg = self._get_tactile_inputs(add_bg=True)

        inputs = {}
        inputs["image"] = images
        inputs["image_bg"] = images_bg
        inputs['current_image_color'] = img
        return inputs

    def _get_tactile_inputs(self, add_bg=False):
        output, output_bg = None, None
        sample_images = []

        for i in self.frames_concat_idx[::-1]:
            img = self.tactile_window[i]
            img = self.loader(img, self.bg, self.enhance_diff_img)
            image = self.transform_resize(img)
            sample_images.append(image)
        
        output = torch.cat(sample_images, dim=0)

        if add_bg:
            bg = self.loader(self.bg, self.bg)
            bg = self.transform_resize(bg)
            output_bg = torch.cat([sample_images[0], bg], dim=0)

        return output, output_bg