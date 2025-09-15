#!/usr/bin/env python3.8
import os
import cv2
import message_filters
import numpy as np
import ros_numpy
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image


def relative_se3(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """
    Computes the relative SE(3) transformation from T1 to T2.

    Args:
        T1 (np.ndarray): First pose as SE(3) transformation matrix of shape (4, 4)
        T2 (np.ndarray): Second pose as SE(3) transformation matrix of shape (4, 4)

    Returns:
        np.ndarray: Relative SE(3) transformation matrix from T1 to T2
    """
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]

    R_rel = R1.T @ R2
    t_rel = R1.T @ (t2 - t1)

    T_rel = np.eye(4)
    T_rel[:3, :3] = R_rel
    T_rel[:3, 3] = t_rel
    return T_rel


class KeyframeSaver:
    def __init__(self, save_prefix: str = "data/nclt/keyframes", sequence: str = "2013-01-10"):
        self.session_dir = os.path.join(save_prefix, sequence)
        os.makedirs(self.session_dir, exist_ok=True)

        # Register synchronized subscribers
        self.cam5_sub = message_filters.Subscriber("/cam5", Image)
        self.gt_sub = message_filters.Subscriber("/gt", Odometry)

        # ApproximateTimeSynchronizer allows synchronization within a given time tolerance
        self.ats = message_filters.ApproximateTimeSynchronizer(
            [
                self.cam5_sub,
                self.gt_sub,
            ],
            queue_size=100,
            slop=0.1,
        )
        self.ats.registerCallback(self.callback)
        print("--> synchronized callback registered")

        # First frame is always a keyframe
        self.keyframe_index = 0
        self.prev_pos = None
        self.current_pos = None
        self.ds = 0

    def callback(
        self,
        cam5_msg,
        gt_odom_msg,
    ):
        print("callback triggered")
        # Process the synchronized messages here
        rospy.loginfo("Synchronized image and odometry messages received.")
        rospy.loginfo(f"Cam5 timestamp: {cam5_msg.header.stamp}")
        rospy.loginfo(f"gt timestamp: {gt_odom_msg.header.stamp}")
        print("---")

        if self.keyframe_index == 0:
            # Save image as numpy file
            cam5_data = ros_numpy.numpify(cam5_msg)
            cv2.imwrite(
                f"{self.session_dir}/{self.keyframe_index:06d}_cam5.png",
                cam5_data,
            )

            # Save absolute gt odometry as numpy file
            gt_se3 = ros_numpy.numpify(gt_odom_msg.pose.pose)
            _file_name = self.session_dir + f"/{self.keyframe_index:06d}_gt_abs_pose.npy"
            np.save(_file_name, gt_se3)

            # Store first pose for within-sequence poses
            self.first_se3 = gt_se3

            # Save relative gt odometry as numpy file
            # First pose is coordinate system center
            _file_name = self.session_dir + f"/{self.keyframe_index:06d}_gt_pose.npy"
            np.save(_file_name, np.eye(4))

            self.prev_pos = ros_numpy.numpify(gt_odom_msg.pose.pose.position)
            self.keyframe_index += 1
        else:
            # Calculate distance and angle between current and previous keyframe
            self.current_pos = ros_numpy.numpify(gt_odom_msg.pose.pose.position)
            self.ds += np.linalg.norm(self.current_pos - self.prev_pos)
            if self.ds >= 0.5:
                cam5_data = ros_numpy.numpify(cam5_msg)
                cv2.imwrite(
                    f"{self.session_dir}/{self.keyframe_index:06d}_cam5.png",
                    cam5_data,
                )

                # Save absolute gt odometry as numpy file
                gt_se3 = ros_numpy.numpify(gt_odom_msg.pose.pose)
                _file_name = self.session_dir + f"/{self.keyframe_index:06d}_gt_abs_pose.npy"
                np.save(_file_name, gt_se3)

                # Save relative gt odometry as numpy file
                _file_name = self.session_dir + f"/{self.keyframe_index:06d}_gt_pose.npy"
                gt_rel_se3 = relative_se3(self.first_se3, gt_se3)
                np.save(_file_name, gt_rel_se3)

                self.prev_pos = self.current_pos
                self.keyframe_index += 1
                self.ds = 0


if __name__ == "__main__":
    rospy.init_node("keyframe_saver")
    saver = KeyframeSaver()
    rospy.spin()
