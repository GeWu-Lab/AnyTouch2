# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pickle
import cv2
import io
import os
import ffmpeg

path_dataset = "/media/chiguera/GUM/datasets/gelsight/touch_go"
path_dataset_out = "/media/chiguera/GUM/datasets/gelsight/light/touch_go/"
max_files = 3500
w = 640
h = 480


def numpy_to_binary(arr):
    is_success, buffer = cv2.imencode(".jpg", arr)
    io_buf = io.BytesIO(buffer)
    return io_buf.read()


def show_video(object):
    video = read_video_ffmpeg(object)
    for frame in video:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(20)
        if key == ord("q"):
            break


def show_dataset(object):
    vid_capture = cv2.VideoCapture(f"{path_dataset}/{object}/gelsight.mp4")
    show_video(vid_capture)
    vid_capture.release()
    cv2.destroyAllWindows()


def compress_and_save_dataset(object):

    out_file = f"{path_dataset_out}/{object}.pkl"
    if os.path.exists(out_file):
        print(f"Dataset {object} already exists")
        return

    compress_dataset = []
    video = read_video_ffmpeg(object)
    for frame in video:
        compress_dataset.append(numpy_to_binary(frame))
        if len(compress_dataset) >= max_files:
            break

    if len(compress_dataset) >= 100:
        out_file = f"{path_dataset_out}/{object}.pkl"
        with open(out_file, "wb") as file:
            pickle.dump(compress_dataset, file)
        print(f"Saved binarized dataset to {out_file}")
    else:
        print(f"Dataset {object} is too small to be saved")


def check_saved_dataset(object):
    with open(f"{path_dataset_out}/{object}.pkl", "rb") as file:
        data = pickle.load(file)
    for frame in data:
        frame = cv2.imdecode(np.frombuffer(frame, np.uint8), -1)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(20)
        if key == ord("q"):
            break


def read_video_ffmpeg(object):
    video_file = f"{path_dataset}/{object}/gelsight.mp4"
    out, _ = (
        ffmpeg.input(video_file)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
    return video


def main():
    objects = os.listdir(path_dataset)
    for obj in objects:
        compress_and_save_dataset(obj)
    print("Done")


if __name__ == "__main__":
    main()
