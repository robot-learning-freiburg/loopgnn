import os

import cv2
import matplotlib

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.size": 10,
    "text.usetex": True,
    "pgf.rcfonts": False,
})
import matplotlib.pyplot as plt
import numpy as np


def generate_video(data_folder,  output_video):
    """
    Generates a side-by-side video showing camera frames (left) and a bird's-eye view plot (right).

    :param image_folder: Path to folder containing images.
    :param position_data: List of (timestamp, x, y) positions.
    :param output_vi
      per second.
    :param video_size: Size of the output video.
    """
    # Load images sorted by timestamp
    video_size=(800, 400)
    fps = 30


    image_files = sorted([f for f in os.listdir(data_folder) if f.endswith(('.png'))])
    pose_files = sorted([f for f in os.listdir(data_folder) if f.endswith('_gt_abs_pose.npy')])
    
    gt_poses = [np.load(os.path.join(data_folder, pose_file)) for pose_file in pose_files]
    gt_pos = np.array([pose[:2, 3] for pose in gt_poses])
    
    if len(image_files) != len(pose_files):
        raise ValueError("Number of images and position data points must match.")

    # Initialize video writer
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_video, fourcc, fps, video_size)

    # Get trajectory bounds for plotting
    min_x, max_x = np.min(gt_pos[:, 0]), np.max(gt_pos[:, 0])
    min_y, max_y = np.min(gt_pos[:, 1]), np.max(gt_pos[:, 1])

    for i, pos in enumerate(gt_pos):
        # Read the image
        img_path = os.path.join(data_folder, image_files[i])
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Resize image to fit half of the video
        cam_width = video_size[0] // 2
        cam_height = video_size[1]
        frame = cv2.resize(frame, (video_size[0], cam_height))

        # Generate bird's eye view plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(min_x - 20, max_x + 20)
        ax.set_ylim(min_y - 20, max_y + 20)
        ax.scatter(gt_pos[:i+1, 0], gt_pos[:i+1, 1], color='grey', s=1)  # Past positions
        ax.scatter(pos[0], pos[1], color='red', marker='o', s=20)  # Current position

        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)

        # no x,y labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)


        # Save plot as image
        plot_path = "temp_plot.png"
        plt.savefig(plot_path)
        plt.close()
        plt.clf()

        # Load and resize plot image
        plot_img = cv2.imread(plot_path)
        plot_img = cv2.resize(plot_img, (cam_width, cam_height))

        # Concatenate images (side by side)
        combined_frame = np.hstack((frame, plot_img))

        # write combined frame to image to disk
        cv2.imwrite(os.path.join(output_video, f"{i}.png"), combined_frame)

        # Write frame to video
        # out.write(combined_frame)

    # Clean up
    # out.release()
    # os.remove(plot_path)
    # print(f"Video saved as {output_video}")

# Example Usage
data_folder = "/path/to/keyframes/figure_8_morning_2023-09-12-10-37-17/"
output_video = "/path/to/output-video/"

generate_video(data_folder, output_video)