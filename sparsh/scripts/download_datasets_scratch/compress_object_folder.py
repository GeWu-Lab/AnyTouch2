# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pickle
import cv2
import io
import os

path_dataset = "/media/chiguera/GUM/datasets/gelsight/object_folder/"
path_dataset_out = "/media/chiguera/GUM/datasets/gelsight/light/object_folder/"


def numpy_to_binary(arr):
    is_success, buffer = cv2.imencode(".jpg", arr)
    io_buf = io.BytesIO(buffer)
    return io_buf.read()


def compress_and_save_dataset(object):

    compress_dataset = []
    path_tactile_data = path_dataset + object + "/tactile_data/"
    reps = os.listdir(path_tactile_data)

    # remove from reps non-folder files
    reps = [r for r in reps if os.path.isdir(path_tactile_data + r)]
    for r in reps:
        print("Processing: ", r)
        path_tactile_data_rep = path_tactile_data + r + "/0/gelsight/"
        # check if the folder exists
        if os.path.exists(path_tactile_data_rep):
            files = os.listdir(path_tactile_data_rep)
            for f in files:
                try:
                    frame = cv2.imread(path_tactile_data_rep + f)
                    frame = numpy_to_binary(frame)
                    compress_dataset.append(frame)
                except:
                    print("Error with file: ", f)
                    continue

    with open(f"{path_dataset_out}/{int(object):03d}.pkl", "wb") as f:
        pickle.dump(compress_dataset, f)
    print("Saved dataset for object: ", object)


def check_saved_dataset(object):
    with open(f"{path_dataset_out}/{int(object):03d}.pkl", "rb") as f:
        data = pickle.load(f)
    for frame in data:
        frame = cv2.imdecode(np.frombuffer(frame, np.uint8), -1)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(20)
        if key == ord("q"):
            break


def main():
    objects = os.listdir(path_dataset)
    for obj in objects:
        compress_and_save_dataset(obj)
    print("Done")


if __name__ == "__main__":
    main()
