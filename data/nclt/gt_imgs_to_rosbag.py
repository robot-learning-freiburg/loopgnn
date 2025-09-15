import rosbag
import rospy
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler

import os
import numpy as np
import cv2
from cv_bridge import CvBridge
from argparse import ArgumentParser

from nclt.undistort import Undistorter


def write_img(image_ts: int, image_folder: str, undistorter: Undistorter, bag: rosbag.Bag) -> None:
    """
    Writes the image to the bag
    Args:
        image_ts (int): Timestamp of the image in microseconds
        image_folder (str): Folder containing the images
        undistorter (Undistorter): Undistorter object
        bag (rosbag.Bag): Bag object to write to

    Returns:
        None
    """
    utime = image_ts
    timestamp = rospy.Time.from_sec(utime / 1e6)

    img_path = os.path.join(image_folder, f"{image_ts}.tiff")
    img = cv2.imread(img_path)
    img = undistorter.undistort(img)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    bridge = CvBridge()
    image_msg = bridge.cv2_to_imgmsg(img)
    image_msg.header.stamp = timestamp
    bag.write('/cam5', image_msg, t=timestamp)


def write_gt(gt: np.ndarray, i: int, bag: rosbag.Bag) -> None:
    """
    Write the ground truth pose to the bag

    Args:
        gt (np.ndarray): Ground truth data of shape `(n, 7)`
        i: Index of the ground truth data to write
        bag (rosbag.Bag): Bag object to write to

    Returns:
        None
    """
    odom = Odometry()
    utime, x, y, z, r, p, h = gt[i, :]
    q = quaternion_from_euler(r, p, h)
    odom.header.stamp = rospy.Time.from_sec(utime / 1e6)
    odom.header.frame_id = "odom"
    odom.pose.pose.position.x = x
    odom.pose.pose.position.y = y
    odom.pose.pose.position.z = z
    odom.pose.pose.orientation.x = q[0]
    odom.pose.pose.orientation.y = q[1]
    odom.pose.pose.orientation.z = q[2]
    odom.pose.pose.orientation.w = q[3]
    bag.write('/gt', odom, t=odom.header.stamp)


def main():
    parser = ArgumentParser(description="NCLT gt and image data to rosbag")
    parser.add_argument("--directory", type=str, help="Directory containing sensor data files")
    parser.add_argument("--date", type=str, help="Date of sensor data")
    args = parser.parse_args()

    bag_path = os.path.join(args.directory, args.date + ".bag")
    bag = rosbag.Bag(bag_path, 'w')
    print(f"Writing to {bag_path}")

    # GT
    gt_file = os.path.join(args.directory, "ground_truth", f"groundtruth_{args.date}.csv")
    gt = np.loadtxt(gt_file, delimiter=",")

    # Images
    image_folder = os.path.join(args.directory, "images", args.date, "lb3", "Cam5")
    image_ts = sorted([int(f.replace(".tiff", "")) for f in os.listdir(image_folder)])
    undistort = Undistorter(os.path.join(args.directory, "U2D_Cam5_1616X1232.txt"))

    i_gt = 0
    i_image = 0

    print('Loaded data, writing ROSbag...')
    total_packets = len(gt) + len(image_ts)
    while True:
        # Figure out next packet in time
        next_packet = "done"
        next_utime = -1

        if i_gt < len(gt) and (gt[i_gt, 0] < next_utime or next_utime < 0):
            next_packet = "gt"
            next_utime = gt[i_gt, 0]

        if i_image < len(image_ts) and (image_ts[i_image] < next_utime or next_utime < 0):
            next_packet = "image"
            next_utime = image_ts[i_image]

        # Now deal with the next packet
        # print(f"{next_packet=} at {next_utime=}")
        if next_packet == "done":
            break
        elif next_packet == "gt":
            write_gt(gt, i_gt, bag)
            i_gt += 1
        elif next_packet == "image":
            write_img(image_ts[i_image], image_folder, undistort, bag)
            i_image += 1
        else:
            print("Unknown packet type")
        packs_done = i_gt + i_image
        print(f"Progress:  {packs_done} / {total_packets}" + 10 * " ", end="\r")

    bag.close()

    return 0


if __name__ == '__main__':
    main()
