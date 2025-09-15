import glob
import os
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as tfm
import tqdm
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


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


class NCLTKeyframes(Dataset):
    """
    Wrapper class for a NCLT keyframes.

    :cam5_files: list of paths to the left camera images
    :gt_pose_files: list of paths to the ground truth pose files
    :gt_pose_abs_files: list of paths to the ground truth pose files

    :keyframes: list of tuples containing the paths to the left cam1 image and ground truth pose file
    """

    def __init__(self, 
                 folder: str, 
                 kp_method: str = None, 
                 mode: str = "multi", 
                 split: str = "train", 
                 dist_thresh: float = 4.0, 
                 angle_thresh: float = 30 * np.pi / 180, 
                 cross_seq_loops: bool = False,
                 scene_wise: bool = True,
                 seqs: List[str] = [],
                 crop_params: Dict = {}) -> None:
        """
        Args:
            folder (str): The folder containing the dataset.
            name (str): The name of the track.
            dist_thresh (float): The distance threshold for loop closure. Default is `4.0`.
            angle_thresh (float): The angle threshold for loop closure. Default is `30*np.pi/180`.
        """
        self.dataset = "nclt"
        self.folder: str = folder
        self.mode: str = mode
        self.sequences: List[str] = seqs
        self.scene_wise: bool = scene_wise
        self.cross_seq_loops: bool = cross_seq_loops
        self.split: str = split
        self._img_width = 800 # after initial cropping
        self._img_height = 800 # after initial cropping
        self.kp_method = kp_method
        self.crop_params = crop_params


        assert self.mode in ["single", "multi"], "Mode must be either 'single' or 'multi'."
        assert self.split in ["train", "test"], "Split must be either 'train' or 'test'."

        self.scene2id = {scene: i for i, scene in zip(range(len(self.sequences)), self.sequences)}
        self.id2scene = {v: k for k, v in self.scene2id.items()}

        self.dist_thresh = dist_thresh
        self.angle_thresh = angle_thresh

        if self.scene_wise:
            _ = self._gather_all_data()  # Loads sequence file paths
            self.keyframes = dict()
            for scene_idx, scene in enumerate(self.sequences):
                self.keyframes[scene_idx] = self._gather_single_scene([scene])
        else:
            self.keyframes = self._gather_all_data()
            print(f"{split.capitalize()} sequence loaded with {len(self.keyframes)} keyframes")

        self.global_desc = None

        if self.scene_wise:
            self.gt_loops = dict()
            self.gt_loop_matrix = dict()
            self.gt_loop_transforms = dict()
            for scene_idx, scene in enumerate(self.sequences):
                self.gt_loop_matrix[scene_idx] = np.zeros(
                    (len(self.keyframes[scene_idx]), len(self.keyframes[scene_idx])))
                self.gt_loop_transforms[scene_idx] = list()
                self.gt_loops[scene_idx] = list()
        else:
            self.gt_loops = list()
            self.gt_loop_transforms = list()
            self.gt_loop_matrix = np.zeros((len(self.keyframes), len(self.keyframes)))

        self._index_loops_cross() if self.cross_seq_loops else self._index_loops_within()
        self._init_pose_matrix()

        # if self.scene_wise:
        #     print(f"Loaded {len(self.sequences)} sequences:")
        #     for idx, self.keyframes in self.keyframes.items():
        #         print(f"  {self.sequences[idx]}: {len(self.keyframes)} keyframes and {len(self.gt_loops[idx])} loops.")

    def _gather_all_data(self) -> List[Dict[str, Any]]:
        """
        Gather all the data from the given folder.
        """
        if self.mode == "single":
            self.cam5_files = glob.glob(os.path.join(self.folder, "*_cam5.png"))
            self.gt_pose_files = glob.glob(os.path.join(self.folder, "*_gt_pose.npy"))
            self.abs_gt_pose_files = glob.glob(os.path.join(self.folder, "*_gt_abs_pose.npy"))

        elif self.mode == "multi":
            self.cam5_files = glob.glob(os.path.join(self.folder, "*/*/*_cam5.png"))
            self.gt_pose_files = glob.glob(os.path.join(self.folder, "*/*/*_gt_pose.npy"))
            self.abs_gt_pose_files = glob.glob(os.path.join(self.folder, "*/*/*_gt_abs_pose.npy"))

        # remove all entries that do not contain any specified sequence name
        self.cam5_files = [f for f in self.cam5_files if any([seq in f for seq in self.sequences])]
        self.gt_pose_files = [f for f in self.gt_pose_files if any([seq in f for seq in self.sequences])]
        self.abs_gt_pose_files = [f for f in self.abs_gt_pose_files if any([seq in f for seq in self.sequences])]

        assert len(self.cam5_files) == len(self.gt_pose_files) == len(
            self.abs_gt_pose_files), "Number of files is uneven"
        assert len(self.cam5_files) > 0, "Number of files is zero"

        # sort the data list
        self.cam5_files.sort()
        self.gt_pose_files.sort()
        self.abs_gt_pose_files.sort()

        self.sample2scene = np.array([self.scene2id[f.split("/")[-2]] for f in self.cam5_files])
        # diff takes the index "before" the diff
        self.scene_idx_offsets = [0] + [x + 1 for x in np.diff(self.sample2scene).nonzero()[0].tolist()]

        return [
            {"cam5": im, "gt_pose": gt, "abs_gt_pose": abs_gt}
            for im, gt, abs_gt in
            zip(self.cam5_files, self.gt_pose_files, self.abs_gt_pose_files)]

    def _gather_single_scene(self, seqs: List[str]) -> List[Dict[str, Any]]:
        """
        Gather all the data of just a single scene.
        """
        cam5_files = [f for f in self.cam5_files if any([seq in f for seq in seqs])]
        gt_pose_files = [f for f in self.gt_pose_files if any([seq in f for seq in seqs])]
        abs_gt_pose_files = [f for f in self.abs_gt_pose_files if any([seq in f for seq in seqs])]

        assert len(cam5_files) == len(gt_pose_files) == len(abs_gt_pose_files), "Number of files is uneven"
        assert len(cam5_files) > 0, "Number of files is zero"

        # sort the data list
        cam5_files.sort()
        gt_pose_files.sort()
        abs_gt_pose_files.sort()

        return [{"cam5": im, "gt_pose": gt, "abs_gt_pose": abs_gt}
                for im, gt, abs_gt in zip(cam5_files, gt_pose_files, abs_gt_pose_files)]

    def _index_loops_within(self) -> None:
        """
        Extract all GT poses and find all poses that are pair-wise similar ones constituting loops.
        """

        # in case of test set we only consider poses within the same scene
        assert not self.cross_seq_loops, "Only considering loop closures within the same sequence."

        if self.scene_wise:
            for scene_idx, scene in enumerate(self.sequences):
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

    def plot_all_trajectories_2D(self) -> None:
        """
        Plot all trajectories in the sequence.
        """
        idx = {'2012-01-22': (5297, 10617), '2012-02-02': (2472, 8403),
               '2012-02-18': (4410, 9792), '2012-05-11': (4303, 9938)}
        sequences = {seq: [] for seq in self.sequences}
        for pose_file in self.abs_gt_pose_files:
            for seq in self.sequences:
                if seq not in pose_file:
                    continue
                kf_index = int(pose_file.split("/")[-1].split("_")[0])
                if kf_index < idx[seq][0] or kf_index > idx[seq][1]:
                    continue
                sequences[seq].append(np.load(pose_file)[:2, 3])

        # create figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # scatter in 2d
        scatter = None
        for seq, poses in sequences.items():
            poses = np.array(poses)
            if len(poses) > 0:
                # Use a colormap to represent progress
                scatter = ax.scatter(poses[:, 0], poses[:, 1], c=np.arange(len(poses)), cmap="viridis", label=seq, s=1)
            else:
                ax.plot(poses[:, 0], poses[:, 1], label=seq)

        # Add color bar
        if scatter is not None:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Keyframe Index')

        plt.legend()
        plt.savefig(f"trajectories2D_{'_'.join(self.sequences)}.png")

    def get_indices_in_area(self, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> Dict[str, Tuple[int, int]]:
        """
        Get the indices of the keyframes that are in the specified area.
        """
        idx = {seq: (np.inf, -np.inf) for seq in self.sequences}
        for pose_file in self.abs_gt_pose_files:
            for seq in self.sequences:
                if seq in pose_file:
                    kf_index = pose_file.split("/")[-1].split("_")[0]
                    x, y = np.load(pose_file)[:2, 3]
                    if xlim[0] < x < xlim[1] and ylim[0] < y < ylim[1]:
                        idx[seq] = (min(idx[seq][0], int(kf_index)), max(idx[seq][1], int(kf_index)))
        return idx


    def set_crop_params_intrinsics(self, crop_params, img_size, K) -> None:
        """
        Crop the original images to the given size.
        """
        self.x_left = (crop_params["width"] - crop_params["width_new"]) // 2
        self.x_right = crop_params["width"] - self.x_left
        self.y_top = (crop_params["height"] - crop_params["height_new"]) // 2
        self.y_bottom = crop_params["height"] - self.y_top

        # recompute principal point intrinsics after initial cropping
        self.K = K.copy()
        c_x_new = K[0, 2] - self.x_left
        c_y_new = K[1, 2] - self.y_top
        self.K[0, 2] = c_x_new
        self.K[1, 2] = c_y_new

        # recompute focal lengths after cropping
        scaler = img_size  / crop_params["width_new"]
        self.K[0, 0] = K[0, 0] * scaler
        self.K[1, 1] = K[1, 1] * scaler

        # recompute the principal points after scaling
        self.K[0, 2] = self.K[0, 2] * scaler
        self.K[1, 2] = self.K[1, 2] * scaler

        self._img_height = crop_params["height_new"]
        self._img_width = crop_params["width_new"]

        print("Cropping parameters set and intrinsics updated")
        return


    def generate_descriptors(self, extractor: None, num_kp: int = 512, img_size: int = 512, device: str = "cuda", recompute: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
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
        if os.path.exists(f"data/{self.dataset}/{self.kp_method}/{self.split}") and not recompute:
            print(f"Loading {self.split} descriptors from disk")
            self.desc_matrices, self.kp_matrices = list(), list()
            for scene in self.sequences:
                self.desc_matrices.append(torch.load(
                    f"data/{self.dataset}/{self.kp_method}/{self.split}/{scene}_descriptors.pt", weights_only=True))
                self.kp_matrices.append(torch.load(
                    f"data/{self.dataset}/{self.kp_method}/{self.split}/{scene}_kp.pt", weights_only=True))
            self.kp_descriptor_matrix = torch.cat(self.desc_matrices).to(device)
            self.kp_matrix = torch.cat(self.kp_matrices).to(device)
        else:
            for scene in self.sequences:
                results = []
                num_keyframes = len(self.keyframes) if not self.scene_wise else np.sum([len(self.keyframes[k]) for k in range(len(self.keyframes))])
                for i in tqdm.tqdm(range(num_keyframes), total=num_keyframes, desc=f"Extracting {self.split}/{scene} descriptors"):
                    if self.cam5_files[i].split("/")[-2] == scene:
                        # crop the image 
                        img_orig = extractor.load_image(self.cam5_files[i], cut_below=None, resize=None)
                        img_cropped = tfm.functional.crop(img_orig, self.y_top, self.x_left, self._img_height, self._img_width)
                        img = tfm.functional.resize(img_cropped, (img_size, img_size), antialias=True)
                        # tfm.ToPILImage()(img).save(f"img_{i}.png")
                        result = list(extractor.forward(img))
                        # fill up the num_kp rows with nans if less than num_kp keypoints are detected
                        if result[1].shape[0] < num_kp:
                            result[0] = torch.cat([result[0], torch.full((num_kp - result[0].shape[0], 2),
                                                                         float('nan')).to(device)]).to(device)  # adding nans to kps
                            result[1] = torch.cat([result[1], torch.full(
                                (num_kp - result[1].shape[0], result[1].shape[1]), float('nan')).to(device)]).to(device)  # adding nans to descs
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
                torch.save(self.scene_descriptor_matrix.cpu(),
                           f"data/{self.dataset}/{self.kp_method}/{self.split}/{scene}_descriptors.pt")
                torch.save(self.scene_kp_matrix.cpu(), f"data/{self.dataset}/{self.kp_method}/{self.split}/{scene}_kp.pt")
                del self.scene_descriptor_matrix, self.scene_kp_matrix
                del results

            # load generated data again as a sanity check
            self.desc_matrices, self.kp_matrices = list(), list()
            for scene in self.sequences:
                self.desc_matrices.append(torch.load(
                    f"data/{self.dataset}/{self.kp_method}/{self.split}/{scene}_descriptors.pt", weights_only=True))
                self.kp_matrices.append(torch.load(
                    f"data/{self.dataset}/{self.kp_method}/{self.split}/{scene}_kp.pt", weights_only=True))
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
                    self.kp_descriptor_matrix[scene_idx] = torch.load(
                        f"data/{self.dataset}/{self.kp_method}/{self.split}/{scene}_descriptors.pt", weights_only=True)
                    self.kp_matrix[scene_idx] = torch.load(
                        f"data/{self.dataset}/{self.kp_method}/{self.split}/{scene}_kp.pt", weights_only=True)
            else:
                self.desc_matrices, self.kp_matrices = list(), list()
                for scene in self.sequences:
                    self.desc_matrices.append(torch.load(
                        f"data/{self.dataset}/{self.kp_method}/{self.split}/{scene}_descriptors.pt", weights_only=True))
                    self.kp_matrices.append(torch.load(
                        f"data/{self.dataset}/{self.kp_method}/{self.split}/{scene}_kp.pt", weights_only=True))
                self.kp_descriptor_matrix = torch.cat(self.desc_matrices)
                self.kp_matrix = torch.cat(self.kp_matrices)

                # print(f"Loading {self.split} descriptors from disk")
                # self.kp_descriptor_matrix = torch.load(os.path.join(path, self.dataset, self.split, f"{self.dataset}_{self.split}_descriptors.pt"), weights_only=True)
                # self.kp_matrix = torch.load(os.path.join(path, self.dataset, f"{self.dataset}_{self.split}_kp.pt"), weights_only=True)
        else:
            raise ValueError("Path to descriptors does not exist.")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Access the data for the given index in the time-series.

        Args:
            idx (int): The index of the data to access.

        Returns:
            _ (Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): The data for the given index. `timestamp, image, gps, imu, tartanvo`

        Note:
        The image is cropped and normalized for further processing. Use `image_left_color` for the original images.
        """

        cam5 = cv2.imread(self.keyframes[idx]["cam5"])
        gt_abs_pose = np.load(self.keyframes[idx]["abs_gt_pose"])

        return cam5, gt_abs_pose

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
                self.pose_files[scene_idx] = [self.keyframes[scene_idx][i]["gt_pose"]
                                              for i in range(len(self.keyframes[scene_idx]))]
                self.pose_matrix[scene_idx] = np.array([np.load(pose_file)
                                                        for pose_file in self.pose_files[scene_idx]]).reshape(-1, 4, 4)
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
        assert global_descriptors.shape[0] == self.kp_descriptor_matrix.shape[0], "Number of global descriptors does not match number of keyframes."
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
    folder = "/data/nclt"
    # seqs = ["2012-01-22", "2012-02-02", "2012-02-18", "2012-05-11"]
    # seqs=["2012-02-02"]
    # seqs=["2012-02-18"]
    # seqs = ["2012-05-11"]
    seqs = ["2012-01-22"]
    track = NCLTKeyframes(folder, seqs=seqs)
    track.plot_all_trajectories_2D()
    # idx = track.get_indices_in_area((-1000, 0), (-1000, -420))
