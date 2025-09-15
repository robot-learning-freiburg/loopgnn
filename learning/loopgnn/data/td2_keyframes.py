import glob
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as tvt
import tqdm
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset, Subset


def angle_between_rotations_quaternion(R1, R2):
    # Convert rotation matrices to quaternions
    q1 = R.from_matrix(R1).as_quat()  # [x, y, z, w]
    q2 = R.from_matrix(R2).as_quat()
    
    # Compute relative quaternion
    q_rel = R.from_quat(q2).inv() * R.from_quat(q1)
    
    # Extract the scalar part
    q_w = np.abs(q_rel.as_quat()[-1])
    
    # Compute angle
    angle = 2 * np.arccos(q_w)
    return angle


class TD2Keyframes(Dataset):
    """
    Wrapper class for a TartanDrive2 keyframes.

    :left_image_files: list of paths to the left image files
    :pred_pose_files: list of paths to the predicted pose files
    :gt_pose_files: list of paths to the ground truth pose files

    :keyframes: list of tuples containing the paths to the left image, predicted pose, and ground truth pose files
    """

    def __init__(self, folder: str, mode: str = "multi", kp_method: str = None, split: str = "train", dist_thresh: float = 4.0, angle_thresh: float = 30*np.pi/180, seqs: List[str] = [], cross_seq_loops: bool = False, scene_wise: bool = True) -> None:
        """
        Args:
            folder (str): The folder containing the dataset.
            name (str): The name of the track.
            dist_thresh (float): The distance threshold for loop closure. Default is `4.0`.
            angle_thresh (float): The angle threshold for loop closure. Default is `30*np.pi/180`.
        """
        self.dataset = "td2"
        self.folder: str = folder
        self.mode: str = mode
        self.sequences: List[str] = seqs
        self.scene_wise: bool = scene_wise
        self.cross_seq_loops: bool = cross_seq_loops
        self.split: str = split
        self._img_width = 1024 
        self._img_height = 544 
        self.load_pred_poses = False
        self.kp_method = kp_method

        assert self.mode in ["single", "multi"], "Mode must be either 'single' or 'multi'."
        assert self.split in ["train", "test"], "Split must be either 'train' or 'test'."

        self.scene2id = {scene: i for i, scene in zip(range(len(self.sequences)), self.sequences)}
        self.id2scene = {v: k for k, v in self.scene2id.items()}

        self.dist_thresh = dist_thresh
        self.angle_thresh = angle_thresh

        if self.scene_wise:
            _ = self._gather_all_data()
            self.keyframes = dict()
            for scene_idx, scene in enumerate(self.sequences):
                self.keyframes[scene_idx] = self._gather_single_scene([scene])
        else:
            self.keyframes = self._gather_all_data()

        self.num_keyframes = len(self.keyframes) if not self.scene_wise else np.sum([len(self.keyframes[k]) for k in range(len(self.keyframes))])
        print("---------------------------------------------")
        print(f"{split.capitalize()} sequence data loaded with {self.num_keyframes} keyframes:")
        self.global_desc = None

        if self.scene_wise:
            self.gt_loops = dict()
            self.gt_loop_matrix = dict()
            self.gt_loop_transforms = dict()    
            for scene_idx, scene in enumerate(self.sequences):                
                self.gt_loop_matrix[scene_idx] = np.zeros((len(self.keyframes[scene_idx]), len(self.keyframes[scene_idx])))
                self.gt_loop_transforms[scene_idx] = list()
                self.gt_loops[scene_idx] = list()
        else:
            self.gt_loops = list()
            self.gt_loop_transforms = list()
            self.gt_loop_matrix = np.zeros((len(self.keyframes), len(self.keyframes)))

        self._index_loops_cross() if self.cross_seq_loops else self._index_loops_within()
        self._init_pose_matrix()

        
        print("\n".join([f"  {seq}" for seq in self.sequences]))
        # print(f"Identified {len(self.gt_loops)} ground truth loop closures.")
        

    def _gather_all_data(self) -> List[Dict[str, Any]]:
        """
        Gather all the data from the given folder.
        """
        if self.mode == "single":
            self.left_image_files = glob.glob(os.path.join(self.folder, "*_image_left_color.png"))
            self.pred_pose_files = glob.glob(os.path.join(self.folder, "*_ov_msckf_pose.npy")) if self.load_pred_poses else None
            self.gt_pose_files = glob.glob(os.path.join(self.folder, "*_gt_pose.npy"))
            self.abs_gt_pose_files = glob.glob(os.path.join(self.folder, "*_gt_abs_pose.npy"))
        
        elif self.mode == "multi":
            self.left_image_files = glob.glob(os.path.join(self.folder, "*/*_image_left_color.png"))
            self.pred_pose_files = glob.glob(os.path.join(self.folder, "*/*_ov_msckf_pose.npy")) if self.load_pred_poses else None
            self.gt_pose_files = glob.glob(os.path.join(self.folder, "*/*_gt_pose.npy"))
            self.abs_gt_pose_files = glob.glob(os.path.join(self.folder, "*/*_gt_abs_pose.npy"))

        # remove all entries that do not contain any specified sequence name
        self.left_image_files = [f for f in self.left_image_files if any([seq in f for seq in self.sequences])]
        self.pred_pose_files = [f for f in self.pred_pose_files if any([seq in f for seq in self.sequences])] if self.load_pred_poses else None
        self.gt_pose_files = [f for f in self.gt_pose_files if any([seq in f for seq in self.sequences])]
        self.abs_gt_pose_files = [f for f in self.abs_gt_pose_files if any([seq in f for seq in self.sequences])]

        assert len(self.left_image_files) == len(self.gt_pose_files) == len(self.abs_gt_pose_files) and len(self.left_image_files) > 0, f"Number of files is either zero or uneven"

        # sort the data list
        self.left_image_files.sort()
        self.pred_pose_files.sort() if self.load_pred_poses else None
        self.gt_pose_files.sort()
        self.abs_gt_pose_files.sort()

        self.sample2scene = np.array([self.scene2id[f.split("/")[-2]] for f in self.left_image_files])
        self.scene_idx_offsets = [0] + [x + 1 for x in np.diff(self.sample2scene).nonzero()[0].tolist()] # diff takes the index "before" the diff

        if self.load_pred_poses:
            return [{"left_image": im, "pred_pose": pred, "gt_pose": gt, "abs_gt_pose": abs_gt} for im, pred, gt, abs_gt in zip(self.left_image_files, self.pred_pose_files, self.gt_pose_files, self.abs_gt_pose_files)]
        else:
            return [{"left_image": im, "gt_pose": gt, "abs_gt_pose": abs_gt} for im, gt, abs_gt in zip(self.left_image_files, self.gt_pose_files, self.abs_gt_pose_files)]
        

    def _gather_single_scene(self, seq: str) -> List[Dict[str, Any]]:
        """
        Gather all the data of just a single scene.
        """
        left_image_files = [f for f in self.left_image_files if any([seq in f for seq in seq])]
        pred_pose_files = [f for f in self.pred_pose_files if any([seq in f for seq in seq])] if self.load_pred_poses else None
        gt_pose_files = [f for f in self.gt_pose_files if any([seq in f for seq in seq])]
        abs_gt_pose_files = [f for f in self.abs_gt_pose_files if any([seq in f for seq in seq])]

        assert len(left_image_files) == len(gt_pose_files) == len(abs_gt_pose_files) and len(left_image_files) > 0, f"Number of files is either zero or uneven"

        # sort the data list
        left_image_files.sort()
        pred_pose_files.sort() if self.load_pred_poses else None
        gt_pose_files.sort()
        abs_gt_pose_files.sort()

        if self.load_pred_poses:
            return [{"left_image": im, "pred_pose": pred, "gt_pose": gt, "abs_gt_pose": abs_gt} for im, pred, gt, abs_gt in zip(left_image_files, pred_pose_files, gt_pose_files, abs_gt_pose_files)]
        else:
            return [{"left_image": im, "gt_pose": gt, "abs_gt_pose": abs_gt} for im, gt, abs_gt in zip(left_image_files, gt_pose_files, abs_gt_pose_files)]
        

    def _index_loops_within(self) -> None:
        """
        Extract all GT poses and find all poses that are pair-wise similar ones constituting loops.
        """

        # in case of test set we only consider poses within the same scene
        assert not self.cross_seq_loops, "Only considering loop closures within the same sequence."

        if self.scene_wise:
            for scene_idx, scene in tqdm.tqdm(enumerate(self.sequences), total=len(self.sequences), desc="Evaluating loops"):
                gt_poses = [np.load(pose) for pose in self.abs_gt_pose_files if scene in pose]

                gt_pos = np.array([pose[:3, 3] for pose in gt_poses])
                gt_ori = np.array([pose[:3, :3] for pose in gt_poses])

                pos_dist = cdist(gt_pos, gt_pos, 'euclidean')
                pos_pairs = np.argwhere(pos_dist < self.dist_thresh)

                for _, row in enumerate(pos_pairs):
                    i, j = row
                    if i != j:
                        angle = angle_between_rotations_quaternion(gt_ori[i], gt_ori[j])
                        if angle < self.angle_thresh:
                            # print(f"Angle between {i} and {j} is {angle}")
                            self.gt_loops[scene_idx].append((i, j))
                            self.gt_loop_transforms[scene_idx].append(np.linalg.inv(gt_poses[i]) @ gt_poses[j])
                            self.gt_loop_matrix[scene_idx][i, j] = 1
        
        else:
            for scene, idx_offset in tqdm.tqdm(zip(self.sequences, self.scene_idx_offsets), total=len(self.sequences), desc="Evaluating loops"):

                gt_poses = [np.load(pose) for pose in self.abs_gt_pose_files if scene in pose]

                gt_pos = np.array([pose[:3, 3] for pose in gt_poses])
                gt_ori = np.array([pose[:3, :3] for pose in gt_poses])

                pos_dist = cdist(gt_pos, gt_pos, 'euclidean')
                pos_pairs = np.argwhere(pos_dist < self.dist_thresh)

                for _, row in enumerate(pos_pairs):
                    i, j = row
                    if i != j:
                        angle = angle_between_rotations_quaternion(gt_ori[i], gt_ori[j])
                        if angle < self.angle_thresh:
                            # print(f"Angle between {i} and {j} is {angle}")
                            self.gt_loops.append((i + idx_offset, j + idx_offset))
                            self.gt_loop_transforms.append(np.linalg.inv(gt_poses[i]) @ gt_poses[j])
                            self.gt_loop_matrix[i + idx_offset, j + idx_offset] = 1


    def _index_loops_cross(self) -> None:
        """
        Extract all GT poses and find all poses that are pair-wise similar ones constituting loops.
        """
        assert self.cross_seq_loops and not self.scene_wise, "Cross sequence loops can only be computed for non-scene-wise data."
        gt_poses = [np.load(pose) for pose in self.abs_gt_pose_files]

        gt_pos = np.array([pose[:3, 3] for pose in gt_poses])
        gt_ori = np.array([pose[:3, :3] for pose in gt_poses])

        pos_dist = cdist(gt_pos, gt_pos, 'euclidean')
        pos_pairs = np.argwhere(pos_dist < self.dist_thresh)

        for _, row in tqdm.tqdm(enumerate(pos_pairs), total=len(pos_pairs), desc="Evaluating loops"):
            i, j = row
            if i != j:
                angle = angle_between_rotations_quaternion(gt_ori[i], gt_ori[j])
                if angle < self.angle_thresh:
                    # print(f"Angle between {i} and {j} is {angle}")
                    self.gt_loops.append((i, j))
                    self.gt_loop_transforms.append(np.linalg.inv(gt_poses[i]) @ gt_poses[j])
                    self.gt_loop_matrix[i, j] = 1


    def __len__(self) -> int:
        """
        Returns:
            _ (int): The number of keyframes in the sequence.
        """
        return len(self.keyframes)
    
    def plot_all_trajectories(self) -> None:
        """
        Plot all trajectories in the sequence.
        """
        gt_poses = [np.load(pose) for pose in self.abs_gt_pose_files]

        gt_pos = np.array([pose[:3, 3] for pose in gt_poses])

        # create figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # scatter in 2d
        ax.scatter(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2])
        # set xlim, ylim
        ax.set_xlim(np.min(gt_pos[:, 0]), np.max(gt_pos[:, 0]))
        ax.set_ylim(np.min(gt_pos[:, 1]), np.max(gt_pos[:, 1]))
        plt.savefig("trajectories.png")

    def generate_descriptors(self, extractor: None, num_kp: int = 2048, img_size: int = 512, device: str = "cuda", recompute: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the keypoint descriptors for the given keyframes.

        Args:
            extractor (Any): The extractor to use for the keypoint descriptors.
            img_size (int): The size to resize the images to. Default is `512`.
            device (str): The device to use for the extraction. Default is `cuda`.
            recompute (bool): Whether to recompute the extracted features. Default is `False`.

        Returns:
            _ (Tuple[torch.Tensor, torch.Tensor]): The keypoint matrix and the descriptor matrix.
        """
        assert self.kp_method is not None, "Keypoint type must be specified."

        save_path = f"data/{self.dataset}/{self.kp_method}/{self.split}/"
        os.makedirs(save_path, exist_ok=True)

        if os.path.exists(save_path) and not recompute:
            print(f"Loading {self.split} descriptors from disk")
            self.desc_matrices, self.kp_matrices = list(), list()
            for scene in self.sequences:
                self.desc_matrices.append(torch.load(f"{save_path}/{scene}_descriptors.pt", weights_only=True))
                self.kp_matrices.append(torch.load(f"{save_path}/{scene}_kp.pt", weights_only=True))
            self.kp_descriptor_matrix = torch.cat(self.desc_matrices).to(device)
            self.kp_matrix = torch.cat(self.kp_matrices).to(device)
        else:    
            for scene in self.sequences:
                results = []
                num_keyframes = len(self.keyframes) if not self.scene_wise else np.sum([len(self.keyframes[k]) for k in range(len(self.keyframes))])
                for i in tqdm.tqdm(range(num_keyframes), total=num_keyframes, desc=f"Extracting {self.split}/{self.kp_method}/{scene} descriptors"):
                    if self.left_image_files[i].split("/")[-2] == scene:
                        img = extractor.load_image(self.left_image_files[i], cut_below=464, resize=img_size)
                        result = list(extractor.forward(img))
                        # fill up the num_kp rows with nans if less than num_kp keypoints are detected, or delete more than 2048 rows
                        if result[1].shape[0] < num_kp:
                            result[0] = torch.cat([result[0].to(device), torch.full((num_kp - result[0].shape[0], 2), float('nan')).to(device)]) # adding nans to kps
                            result[1] = torch.cat([result[1].to(device), torch.full((num_kp - result[1].shape[0], result[1].shape[1]), float('nan')).to(device)]) # adding nans to descs
                            assert result[0].shape[0] == num_kp and result[1].shape[0] == num_kp, f"Keypoints and descriptors do not match num_kp."
                        elif result[1].shape[0] > num_kp:
                            result[0] = result[0][:num_kp].to(device)
                            result[1] = result[1][:num_kp].to(device)
                        else:
                            result[0] = result[0].to(device)
                            result[1] = result[1].to(device)
                        results.append(result)
                    else:
                        continue
                self.scene_descriptor_matrix = torch.stack([results[i][1].cpu() for i in range(len(results))])
                self.scene_kp_matrix = torch.stack([results[i][0].cpu() for i in range(len(results))])
                torch.save(self.scene_descriptor_matrix.cpu(), f"{save_path}/{scene}_descriptors.pt")
                torch.save(self.scene_kp_matrix.cpu(), f"{save_path}/{scene}_kp.pt")
                del self.scene_descriptor_matrix, self.scene_kp_matrix
                del results

            # Directly load from disk again to check for usability
            self.desc_matrices, self.kp_matrices = list(), list()
            for scene in self.sequences:
                self.desc_matrices.append(torch.load(f"{save_path}/{scene}_descriptors.pt", weights_only=True))
                self.kp_matrices.append(torch.load(f"{save_path}/{scene}_kp.pt", weights_only=True))
            self.kp_descriptor_matrix = torch.cat(self.desc_matrices).to(device)
            self.kp_matrix = torch.cat(self.kp_matrices).to(device)
                
        return self.kp_descriptor_matrix, self.kp_matrix
        
    
    def load_kp_descriptors(self, path: str = None) -> None:
        """
        Load the descriptors with shape (num_images, num_descriptors, desc_dim) from disk.
        """
        if os.path.exists(os.path.join(path, self.dataset)):
            if self.scene_wise:
                self.kp_descriptor_matrix, self.kp_matrix = dict(), dict()
                for scene_idx, scene in enumerate(self.sequences):
                    self.kp_descriptor_matrix[scene_idx] = torch.load(f"data/{self.dataset}/{self.kp_method}/{self.split}/{scene}_descriptors.pt", weights_only=True)
                    self.kp_matrix[scene_idx] = torch.load(f"data/{self.dataset}/{self.kp_method}/{self.split}/{scene}_kp.pt", weights_only=True)
            else:
                self.desc_matrices, self.kp_matrices = list(), list()
                for scene in self.sequences:
                    self.desc_matrices.append(torch.load(f"data/{self.dataset}/{self.kp_method}/{self.split}/{scene}_descriptors.pt", weights_only=True))
                    self.kp_matrices.append(torch.load(f"data/{self.dataset}/{self.kp_method}/{self.split}/{scene}_kp.pt", weights_only=True))
                self.kp_descriptor_matrix = torch.cat(self.desc_matrices)
                self.kp_matrix = torch.cat(self.kp_matrices)

                # print(f"Loading {self.split} descriptors from disk")
                # self.kp_descriptor_matrix = torch.load(os.path.join(path, self.dataset, self.split, f"{self.dataset}_{self.split}_descriptors.pt"), weights_only=True)
                # self.kp_matrix = torch.load(os.path.join(path, self.dataset, f"{self.dataset}_{self.split}_kp.pt"), weights_only=True)
        else:
            raise ValueError("Path to descriptors does not exist.")

    def __getitem__(self, idx: int) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Access the data for the given index in the time-series.

        Args:
            idx (int): The index of the data to access.

        Returns:
            _ (Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): The data for the given index. `timestamp, image, gps, imu, tartanvo`

        Note:
        The image is cropped and normalized for further processing. Use `image_left_color` for the original images.
        """

        left_image = np.load(self.keyframes[idx]["left_image"])
        pred_pose = np.load(self.keyframes[idx]["pred_pose"])
        gt_pose = np.load(self.keyframes[idx]["gt_pose"])

        return left_image, pred_pose, gt_pose
    
    def filter_keyframes(self, idcs: List[int]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Mask the dataset with the given keyframe indices.

        Args:
            idcs (List[int]): The indices to keep in the dataset.
        """
        self.mask_idcs = idcs
        self.masked_keyframes = [self.keyframes[i] for i in idcs]
        self.mask_to_orig = {new_i: i for new_i, i in enumerate(idcs)}
        self.orig_to_mask = {i: new_i for new_i, i in enumerate(idcs)}

        # mask ground truth matrix
        self.masked_gt_loop_matrix = self.gt_loop_matrix[idcs][:, idcs]
        
        loop_idcs = self.masked_gt_loop_matrix.nonzero()
        self.masked_gt_loops = [(int(loop_idcs[0][i]), int(loop_idcs[1][i])) for i in range(len(loop_idcs[0]))]
        return self.masked_gt_loop_matrix, self.masked_gt_loops
    
    def _init_pose_matrix(self) -> None:
        """
        Get the pose matrix over all keyframes.
        """
        if self.scene_wise:
            self.pose_files, self.pose_matrix = dict(), dict()
            for scene_idx, scene in enumerate(self.sequences):
                self.pose_files[scene_idx] = [self.keyframes[scene_idx][i]["gt_pose"] for i in range(len(self.keyframes[scene_idx]))]
                self.pose_matrix[scene_idx] = np.array([np.load(pose_file) for pose_file in self.pose_files[scene_idx]]).reshape(-1, 4, 4)
        else:
            self.pose_files = [self.keyframes[i]["gt_pose"] for i in range(len(self.keyframes))]
            self.pose_matrix = np.array([np.load(pose_file) for pose_file in self.pose_files]).reshape(-1, 4, 4)

    def compute_rel_trafo(self, i: int, j: int, ) -> np.ndarray:
        """
        Get the relative pose between two keyframes.
        """
        return np.linalg.inv(self.pose_matrix[i]) @ self.pose_matrix[j]
    
    def compute_rel_trafo_scene(self, i: int, j: int, scene_idx: int) -> np.ndarray:
        """
        Get the relative pose between two keyframes.
        """
        return np.linalg.inv(self.pose_matrix[scene_idx][i] @ self.pose_matrix[scene_idx][j])


    def mask2orig(self, masked_idcs: Tuple[int, int] | int) -> Tuple[int, int] | int:
        """
        Convert masked indices to original indices.
        
        Args:
            masked_idcs (Tuple[int, int] | int): The masked indices to convert.
        Returns:
            _ (Tuple[int, int] | int): The original indices.
        """
        if isinstance(masked_idcs, int):
            return self.mask_to_orig[masked_idcs]
        elif isinstance(masked_idcs, tuple):
            return self.mask_to_orig[masked_idcs[0]], self.mask_to_orig[masked_idcs[1]]
        else:
            raise ValueError("Input must be either an integer or a tuple of integers.")
        
    def add_global_descriptors(self, global_descriptors: torch.Tensor | None, save_path: str = None) -> None:
        assert global_descriptors.shape[0] == len(self.keyframes), "Number of global descriptors does not match number of keyframes."
        self.global_desc = global_descriptors
        if save_path is not None:
            # np.save(save_path, global_descriptors.cpu().numpy())
            torch.save(global_descriptors.cpu(), save_path)

    def load_global_descriptors(self, path: str) -> None:
        """
        Load the global descriptors from disk.
        """
        assert path is not None, "Path to global descriptors must be given."
        self.global_desc = torch.load(path, weights_only=True)


if __name__ == "__main__":
    folder = "/data/td2/keyframes/"
    name = "2023-11-14-14-34-53_gupta"
    track = TD2Keyframes(folder, name)
