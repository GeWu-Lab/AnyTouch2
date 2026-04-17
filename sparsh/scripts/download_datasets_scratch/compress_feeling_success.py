# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pickle
import cv2
import io
import os
import deepdish as dd

path_dataset = "/media/chiguera/GUM/datasets/gelsight/feeling_success"
path_dataset_out = "/media/chiguera/GUM/datasets/gelsight/feeling_success/"
max_files = 3500


def numpy_to_binary(arr):
    is_success, buffer = cv2.imencode(".jpg", arr)
    io_buf = io.BytesIO(buffer)
    return io_buf.read()


def compress_and_save_dataset(object):
    print(f"Loading data for {object}")
    t = dd.io.load(f"{path_dataset}/{object}")
    compress_dataset = {
        "object_name": [],
        "is_gripping": [],
        "gelsightA_before": [],
        "gelsightA_during": [],
        "gelsightA_after": [],
        "gelsightB_before": [],
        "gelsightB_during": [],
        "gelsightB_after": [],
    }
    n_experiments = len(t)
    for idx in range(n_experiments):
        experiment = t[idx]
        compress_dataset["object_name"].append(experiment["object_name"])
        compress_dataset["is_gripping"].append(experiment["is_gripping"])

        for key in compress_dataset.keys():
            if "gelsight" in key:
                frame = numpy_to_binary(experiment[key])
                compress_dataset[key].append(frame)

    filename = object.split("_")[-1].split(".")[0]
    out_file = f"{path_dataset_out}/{filename}.pkl"
    with open(out_file, "wb") as file:
        pickle.dump(compress_dataset, file)
    print(f"Saved binarized dataset to {out_file}")


def check_saved_dataset(object):
    with open(f"{path_dataset_out}/{object}", "rb") as file:
        data = pickle.load(file)

    for frame in data["gelsightA_during"]:
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
