# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import glob


def load_data(data_dir: str, nominal_freq: int):
    with open(data_dir + "/data.pkl", "rb") as f:
        data = pickle.load(f)
    ee_pose = data["ee_pose"]
    force_data = data["force"]

    ee_timestamps = ee_pose[:, 0]
    duration = ee_timestamps[-1] - ee_timestamps[0]
    num_frames = int(duration * nominal_freq)
    start_timestamps = [ee_timestamps[0], force_data[0, 0]]
    end_timestamps = [ee_timestamps[-1], force_data[-1, 0]]

    start_timestamp = max(start_timestamps)
    end_timestamp = min(end_timestamps)

    interpolating_timestamps = np.linspace(start_timestamp, end_timestamp, num_frames)
    print(f"start_timestamp: {start_timestamp}, end_timestamp: {end_timestamp}")
    print(f"duration: {duration}, num_frames: {num_frames}")

    ee_pose = interp1d(ee_timestamps, ee_pose[:, 1:4], axis=0)(interpolating_timestamps)
    forces = interp1d(force_data[:, 0], force_data[:, 1:4], axis=0)(
        interpolating_timestamps
    )
    # Invert normal force to get positive forces
    forces[:, -1] = forces[:, -1] * -1
    return ee_pose, forces


def extract_strokes(in_contact: np.ndarray):
    trajectories = []
    trajectory = []
    for i in range(1, len(in_contact)):
        if in_contact[i] != in_contact[i - 1]:
            if len(trajectory) > 25:
                trajectories.append(trajectory)
            trajectory = []
        else:
            if in_contact[i] == 1:
                trajectory.append(i)
    return trajectories


def label_slip(frictional_coeff: float, forces: np.ndarray):
    shear_magnitude = np.linalg.norm(forces[:, :2], axis=-1)
    f_limit = frictional_coeff * forces[:, 2]
    slip_labels = np.ones(len(forces))
    slip_labels[shear_magnitude <= f_limit] = 0.0
    return slip_labels


def plot_slip_per_stroke(
    trajectories, slip_trajectories, ee_pose, shear_magnitude, forces, frictional_coeff
):
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=10,
            label="No Slip",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="Slip",
        ),
    ]

    for traj in range(1, len(trajectories), 10):
        _, ax = plt.subplots(2, 3, figsize=(10, 10))
        slip_labels_traj = slip_trajectories[traj]
        ax[0, 0].plot(slip_labels_traj, label="slip_label", color="blue")
        ax[0, 0].set_yticks([0, 1], labels=["No Slip", "Slip"])
        ax[0, 0].legend()

        ft = shear_magnitude[trajectories[traj]]
        fz = forces[:, 2][trajectories[traj]]
        x = np.linspace(0, ft.max(), 100)
        y = (1 / frictional_coeff) * x

        color = np.where(slip_labels_traj == 1, "red", "blue")
        pose = ee_pose[trajectories[traj], 1:3]
        disp = np.linalg.norm(pose[:] - pose[0], axis=-1)
        color_ = plt.get_cmap("viridis")(disp / disp.max())
        # color_ = np.where(disp > 0.0025, "red", "blue")
        ax[0, 1].scatter(ft, fz, c=color_, s=2)
        ax[0, 1].plot(x, y, "--", c="gray", label="Friction Boundary")
        ax[0, 1].set_ylabel("Normal Force (N)")
        ax[0, 1].set_xlabel("Shear Force Magnitude (N)")

        # ax[0, 1].set_aspect("equal")
        ax[0, 1].legend(handles=handles)

        # End effector displacement vs Frames for each stroke
        displacement = np.linalg.norm(pose[:] - pose[0], axis=-1)
        # displacement = np.linalg.norm(pose[1:] - pose[:-1], axis=-1)
        # displacement = np.concatenate([[0], displacement])
        ax[1, 0].scatter(
            np.arange(displacement.shape[0]),
            disp,
            color=color,
            label="displacement",
            s=2,
        )
        ax[1, 0].set_xlabel("Frames")
        ax[1, 0].set_ylabel("Displacement (cm)")
        ax[1, 0].legend(handles=handles)

        # Shear force vs displacement for each stroke
        ax[1, 1].scatter(displacement, ft, c=color, s=2)
        ax[1, 1].set_xlabel("Displacement (cm)")
        ax[1, 1].set_ylabel("Shear Force Magnitude (N)")
        ax[1, 1].legend(handles=handles)

        ax[1, 2].scatter(displacement, fz, c=color, s=2)
        ax[1, 2].set_xlabel("Displacement (cm)")
        ax[1, 2].set_ylabel("Normal force Magnitude (N)")
        ax[1, 2].legend(handles=handles)

        plt.title(f"Trajectory {traj}")
        plt.show()


def plot_data_summary(
    data_dir,
    trajectories,
    ee_pose,
    in_contact,
    forces,
    frictional_coeff,
    slip_percentage,
):
    _, axs = plt.subplots(2, 3, figsize=(15, 10))

    filtered_ee_pose = ee_pose[in_contact, :]
    shear_magnitude = np.linalg.norm(forces[:, :2], axis=-1)
    shear_direction = np.arctan2(forces[:, 1], forces[:, 0])
    normal_force = forces[:, 2]

    # Plot: End effector pose as line segments on a 2D plot
    axs[0, 0].plot(
        ee_pose[:, 0],
        ee_pose[:, 1],
        linestyle="-",
        color="gray",
        alpha=0.2,
        label=f"{len(trajectories)} trajectories",
    )

    for t in trajectories:
        axs[0, 0].plot(
            ee_pose[t, 0],
            ee_pose[t, 1],
            linestyle="-",
        )
    axs[0, 0].set_xlabel("X")
    axs[0, 0].set_ylabel("Y")
    axs[0, 0].set_title("Filtered End Effector Pose")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot: friction cone
    axs[0, 1].plot(
        shear_magnitude,
        normal_force,
        marker="o",
        markersize=2,
        linestyle="",
        label="Force Z vs Magnitude of Force X and Y",
        color="purple",
    )
    axs[0, 1].set_xlabel("Magnitude of Force X and Y")
    axs[0, 1].set_ylabel("Force Z")
    axs[0, 1].set_title("Force Z vs Magnitude of Force X and Y")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot: friction cone labels
    f_limit = frictional_coeff * normal_force
    slip_state = np.ones(len(normal_force))
    slip_state[shear_magnitude <= f_limit] = 0.0

    idx_no_slip = list(np.where(slip_state == 0.0)[0])
    idx_slip = list(np.where(slip_state == 1.0)[0])
    axs[0, 2].scatter(
        shear_magnitude[idx_no_slip],
        normal_force[idx_no_slip],
        c="blue",
        s=2,
        label=f"No Slip ({slip_percentage*100:.2f}%)",
    )
    axs[0, 2].scatter(
        shear_magnitude[idx_slip],
        normal_force[idx_slip],
        c="red",
        s=2,
        label=f"Slip ({(1 - slip_percentage)*100:.2f}%)",
    )
    axs[0, 2].set_xlabel("Magnitude of Force X and Y")
    axs[0, 2].set_ylabel("Force Z")
    axs[0, 2].set_title("Friction cone labels")
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # Plot: Normal force distribution (Z-axis data)
    # plot histogram for force z
    axs[1, 0].hist(
        normal_force,
        bins=30,
        alpha=0.9,
        color="green",
        edgecolor="gray",
        label="Normal Force (Z-axis)",
    )
    axs[1, 0].set_xlabel("Force (N)")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].set_title("Normal Force Distribution")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    axs[1, 0].set_yscale("log")

    # Plot: Shear magnitude and direction distribution (force X vs force Y data)
    axs[1, 1].hist(
        shear_magnitude,
        bins=30,
        alpha=0.9,
        color="orange",
        edgecolor="gray",
        label="Magnitude of Shear Force",
    )
    axs[1, 1].set_xlabel("Magnitude (N)")
    axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].set_title("Magnitude of Shear Force Distribution")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    axs[1, 1].set_yscale("log")

    axs[1, 2].hist(
        shear_direction,
        bins=30,
        alpha=0.9,
        color="orange",
        edgecolor="gray",
        label="Direction of Shear Force",
    )
    axs[1, 2].set_xlabel("Direction (rad)")
    axs[1, 2].set_ylabel("Frequency")
    axs[1, 2].set_title("Direction of Shear Force Distribution")
    axs[1, 2].legend()
    axs[1, 2].grid(True)
    axs[1, 2].set_yscale("log")

    # Adjust layout
    plt.tight_layout()

    # save the plot
    plt.savefig(data_dir + "/force_pose_data.png")

    # # Show the plot
    plt.show()


def main(args: argparse.Namespace):
    data_dir = args.data_dir
    ee_pose, forces = load_data(data_dir, args.data_nominal_freq)
    data_root = (
        "/home/akashsharma/workspace/datasets/reskin_data/downstream_slip_detection_/"
    )
    data_paths = glob.glob(data_root + "/*")
    ee_pose, forces = [], []
    for data_dir in data_paths:
        print(f"Processing {data_dir}")
        ee_pose_, forces_ = load_data(data_dir, args.data_nominal_freq)
        ee_pose.append(ee_pose_)
        forces.append(forces_)
    ee_pose = np.concatenate(ee_pose, axis=0)
    forces = np.concatenate(forces, axis=0)

    forces = savgol_filter(forces, 51, 3, axis=0)

    in_contact = np.zeros(forces.shape[0], dtype=int)
    in_contact[forces[:, 2] > args.normal_force_contact_threshold] = 1

    baseline_forces = forces[~in_contact]
    print(f"Baseline forces: {baseline_forces.shape}")
    plt.plot(baseline_forces[:, 0], color="r")
    plt.plot(baseline_forces[:, 1], color="g")
    plt.plot(baseline_forces[:, 2], color="b")
    plt.show()
    print(f"Baseline forces: {baseline_forces.mean(axis=0)}")
    forces -= baseline_forces.mean(axis=0)
    forces[forces[:, 2] < 0] = 0
    # forces[forces[:, 2] < 0] = 0

    plt.plot(forces[:, 2])
    plt.plot(in_contact)
    plt.show()

    trajectories = extract_strokes(in_contact)
    print(f"Number of strokes in rosbag: {len(trajectories)}")

    # label slip based on friction cone
    slip_labels = label_slip(args.frictional_coeff, forces)
    slip_percentage = np.sum(slip_labels) / len(slip_labels)
    # print(np.bincount(slip_labels.astype(int)))
    # print(np.sum(slip_labels), len(slip_labels) - np.sum(slip_labels))
    class_weights = len(slip_labels) / (2 * np.bincount(slip_labels.astype(int)))
    print(f"class weights: {class_weights}")
    no_slip_percentage = 1 - slip_percentage
    slip_trajectories = []
    for traj in trajectories:
        slip_trajectories.append(slip_labels[traj])

    print(f"Slip: {slip_percentage*100:.2f}%, No Slip: {no_slip_percentage*100:.2f}%")

    data_trajectories = {
        "trajectories": trajectories,
        "in_contact": in_contact,
        "slip_label": slip_labels,
        "slip_trajectories": slip_trajectories,
    }
    with open(data_dir + "/dataset_info_trajectories.pkl", "wb") as f:
        pickle.dump(data_trajectories, f)

    if args.plot_slip_per_stroke:
        shear_magnitude = np.linalg.norm(forces[:, :2], axis=-1)
        plot_slip_per_stroke(
            trajectories,
            slip_trajectories,
            ee_pose,
            shear_magnitude,
            forces,
            args.frictional_coeff,
        )

    plot_data_summary(
        data_dir=data_dir,
        trajectories=trajectories,
        ee_pose=ee_pose,
        in_contact=in_contact,
        forces=forces,
        frictional_coeff=args.frictional_coeff,
        slip_percentage=slip_percentage,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Slip Labelling")
    parser.add_argument(
        "--data-dir",
        default="/home/akashsharma/workspace/datasets/reskin_data/downstream_slip_detection_/reskin_palm_1cm_sphere-003.bag",
        type=str,
        help="Data directory",
    )
    parser.add_argument(
        "--data-nominal-freq",
        type=str,
        default=70,
        help="Nominal frequency to interpolate the data",
    )
    parser.add_argument(
        "--normal-force-contact-threshold",
        type=float,
        default=0.04,
        help="Threshold on measured normal force to determine if the object is in contact",
    )
    parser.add_argument(
        "--frictional-coeff",
        type=float,
        default=1.6,
        help="Frictional coefficient to determine slip based on friction cone",
    )
    parser.add_argument(
        "--plot-slip-per-stroke",
        action="store_true",
        default=False,
        help="Plot slip labels for each stroke",
    )
    args = parser.parse_args()
    main(args)
