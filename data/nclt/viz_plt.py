import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from typing import Optional
import pprint


def read_se3(folder: str, n: Optional[int] = None, skip: int = 1, abs: bool = False) -> np.ndarray:
    mode = "abs" if abs else ""
    paths = sorted(glob(os.path.join(folder, "*_gt_" + mode + "_pose.npy")))[:n:skip]
    poses = np.stack([np.load(p) for p in paths], axis=0)
    return poses


def plot_coordinate_frame(ax, pose, size=2):
    origin = pose[:3, 3]
    x_axis = pose[:3, 0] * size
    y_axis = pose[:3, 1] * size
    z_axis = pose[:3, 2] * size

    # Plot arrows: (x, y, z, u, v, w)
    ax.quiver(origin[0], origin[1], origin[2],
              x_axis[0], x_axis[1], x_axis[2],
              color='r', arrow_length_ratio=0.2)
    ax.quiver(origin[0], origin[1], origin[2],
              y_axis[0], y_axis[1], y_axis[2],
              color='g', arrow_length_ratio=0.2)
    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0], z_axis[1], z_axis[2],
              color='b', arrow_length_ratio=0.2)


def main():
    folder_path = "/export/dorers/nclt/keyframes/2013-01-10"
    print(f"Reading poses from {folder_path}.")
    poses = read_se3(folder_path, 100, 8)
    print("Creating coordinate frames.")
    pprint.pprint(poses)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for pose in poses:
        plot_coordinate_frame(ax, pose)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    # Get the center of each axis
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    # Get the max range needed
    max_range = max(
        (x_limits[1] - x_limits[0]) / 2,
        (y_limits[1] - y_limits[0]) / 2,
        (z_limits[1] - z_limits[0]) / 2
    )

    # Set equal limits around the middle point
    ax.set_xlim([x_middle - max_range, x_middle + max_range])
    ax.set_ylim([y_middle - max_range, y_middle + max_range])
    ax.set_zlim([z_middle - max_range, z_middle + max_range])

    plt.savefig("coordinate_frame.png")


if __name__ == "__main__":
    main()
