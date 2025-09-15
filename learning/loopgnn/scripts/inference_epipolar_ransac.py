import os
from collections import defaultdict
from typing import Dict, List

import cv2
import hydra
import numpy as np
import scipy
import torch
import torch_geometric.data
import tqdm
from loopgnn.data.graph_data import GraphDataset
from loopgnn.data.td2_keyframes import TD2Keyframes
from loopgnn.models.network import LoopGNN
from loopgnn.utils.vlad import VLAD, get_top_k_recall
from omegaconf import DictConfig, OmegaConf
from torcheval.metrics import (BinaryAUPRC, BinaryF1Score,
                               BinaryRecallAtFixedPrecision, Mean)

# set seed
np.random.seed(42)
torch.manual_seed(42)

DEFAULT_RANSAC_ITERS = 2000
DEFAULT_RANSAC_CONF = 0.95
DEFAULT_REPROJ_THRESH = 3



class EpipolarRansacInference(torch_geometric.data.Dataset):
    """Characterizes a graph dataset to torch geometric."""

    def __init__(self, params: DictConfig, scenes: List[str], split: str):
        super(EpipolarRansacInference, self).__init__()
        """
        Initializes a graph dataset.
        Args:
        """
        self.params = params
        self.scenes = scenes
        self.split = split
        self.kp_method = self.params.main.kp_method
        self.desc_dim = self.params.preprocessing.kp[self.kp_method].desc_dim
        self.num_kp = self.params.preprocessing.kp[self.kp_method].num_kp

        self.K = np.array(self.params.data[self.params.main.dataset].intrinsics).reshape(3,3)

        self.test_scenes = params.data[params.main.dataset].test

        self.test_graphdata = GraphDataset(self.params, self.test_scenes, split='test')
        self.test_graph_loader = torch_geometric.loader.DataLoader(self.test_graphdata, batch_size=1, num_workers=4, shuffle=False)

        self.pred_score_arrays = dict()
        self.ransac_score_arrays = dict()

        self.match_min_cossim = 0.82


    def find_epipolor_geometry(
        self,
        points1: np.ndarray | torch.Tensor,
        points2: np.ndarray | torch.Tensor, 
        reproj_thresh: int = DEFAULT_REPROJ_THRESH,
        num_iters: int = DEFAULT_RANSAC_ITERS,
        ransac_conf: float = DEFAULT_RANSAC_CONF,
    ):

        assert points1.shape == points2.shape
        assert points1.shape[1] == 2
        # check if points1 is a torch tensor
        if isinstance(points1, torch.Tensor):
            points1, points2 = points1.cpu().numpy(), points2.cpu().numpy()

        F, inliers_mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, reproj_thresh, ransac_conf, num_iters)
        assert inliers_mask.shape[1] == 1
        inliers_mask = inliers_mask[:, 0]
        return F, inliers_mask.astype(bool)
        

    def evaluate(self):
        metrics = defaultdict(list)
        
        average_precision = BinaryAUPRC()
        maximum_recall = BinaryRecallAtFixedPrecision(min_precision=1.0)
        f1_score = BinaryF1Score(device=self.params.main.device, threshold=0.8)
        average_relative_pose_error = Mean()
        average_translation_error = Mean()
                
        # turn pred_scores into numpy array
        for scene_idx in list(self.test_graphdata.scene_vlad_sim.keys()):
            self.ransac_score_arrays[scene_idx] = np.zeros_like(self.test_graphdata.scene_vlad_sim[scene_idx])

        overall_avg_prec, overall_max_rec, overall_f1 = dict(), dict(), dict()
        overall_ape, overall_ate = dict(), dict()
        for scene_idx in list(self.test_graphdata.scene_vlad_sim.keys()):
            print(f"Scene {scene_idx}")
            scene_len = len(self.test_graphdata.kf_data.keyframes[scene_idx])
            scene_idx_offset = self.test_graphdata.kf_data.scene_idx_offsets[scene_idx]
            scene_global_desc = self.test_graphdata.kf_data.global_desc[0+scene_idx_offset:scene_len+scene_idx_offset]
            # keyframes_idcs = np.linspace(0, scene_len-1, scene_len).astype(np.int16)


            # Given the idx of the query frame, extract the top-k similar frames
            topk = min(int(0.01 * len(self.test_graphdata.kf_data.keyframes[scene_idx])), 100)
            
            # gt labels
            labels = torch.from_numpy(self.test_graphdata.kf_data.gt_loop_matrix[scene_idx]).flatten().long()

            # keypoint data
            kp_desc = self.test_graphdata.kf_data.kp_descriptor_matrix[scene_idx] # .reshape(-1, self.num_kp*self.desc_dim)
            kp = self.test_graphdata.kf_data.kp_matrix[scene_idx] # [keyframes_idcs, :, :].reshape(-1, self.num_kp*2)
            node_poses = self.test_graphdata.kf_data.pose_matrix[scene_idx] # [keyframes_idcs, :, :].reshape(-1, 4*4)

            for query_idx in tqdm.tqdm(list(range(scene_len)), total=scene_len, desc="Evaluating model"):
                query_neighborhood = np.argpartition(self.test_graphdata.scene_vlad_sim[scene_idx][query_idx, :], topk)[-topk:]
                for target_idx in query_neighborhood:

                    query_desc = kp_desc[query_idx]
                    query_kp = kp[query_idx]
                    query_pose = node_poses[query_idx]

                    target_desc = kp_desc[target_idx]
                    target_kp = kp[target_idx]
                    target_pose = node_poses[target_idx]

                    cossim = query_desc @ target_desc.t()
                    cossim_t = target_desc @ query_desc.t()
                    
                    _, match12 = cossim.max(dim=1)
                    _, match21 = cossim_t.max(dim=1)

                    idx0 = torch.arange(len(match12), device=match12.device)
                    mutual = match21[match12] == idx0

                    if self.match_min_cossim > 0:
                        cossim, _ = cossim.max(dim=1)
                        good = cossim > self.match_min_cossim
                        idx0 = idx0[mutual & good]
                        idx1 = match12[mutual & good]
                    else:
                        idx0 = idx0[mutual]
                        idx1 = match12[mutual]

                    mkpts0, mkpts1 = query_kp[idx0], target_kp[idx1]
                    try:
                        F, inlier_mask = self.find_epipolor_geometry(mkpts0, mkpts1)

                        inlier_kpts0 = mkpts0[inlier_mask]
                        inlier_kpts1 = mkpts1[inlier_mask]

                        if F is not None:
                            pred_score = len(inlier_kpts0)/len(mkpts1)
                            self.ransac_score_arrays[scene_idx][query_idx, target_idx] = pred_score

                            if pred_score > 0.5:
                                E, mask = cv2.findEssentialMat(mkpts0.numpy(), mkpts1.numpy(), self.K, cv2.RANSAC, 0.999, 1.0)
                                _, R, t, mask = cv2.recoverPose(E, mkpts0.numpy(), mkpts1.numpy(), self.K)

                                # relative ground truth pose 
                                rel_gt_pose_matrix = scipy.linalg.inv(query_pose) @ target_pose
                                rel_gt_rotaiton = rel_gt_pose_matrix[:3, :3]
                                rel_gt_translation = rel_gt_pose_matrix[:3, 3] / (np.linalg.norm(rel_gt_pose_matrix[:3, 3])+1e-6)
                                
                                transl_error = torch.norm(torch.from_numpy(t) - torch.from_numpy(rel_gt_translation))
                                if torch.isnan(transl_error):
                                    print("Translation error is nan")
                                else:
                                    average_translation_error.update(transl_error)

                                relative_rotation_error = rel_gt_rotaiton.T @ R 
                                trace = np.trace(relative_rotation_error)
                                theta = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
                                theta_deg_error = np.rad2deg(theta)
                                average_relative_pose_error.update(torch.tensor(theta_deg_error))
                    except Exception as e:
                        print(f"Error: {e}")

            labels = torch.from_numpy(self.test_graphdata.kf_data.gt_loop_matrix[scene_idx]).flatten().long()
            preds = torch.from_numpy(self.ransac_score_arrays[scene_idx]).flatten()
            if not os.path.exists(f"data/{self.params.main.dataset}/{self.params.main.kp_method}/eval-ransac/"):
                os.makedirs(f"data/{self.params.main.dataset}/{self.params.main.kp_method}/eval-ransac/")
            torch.save(preds, f"data/{self.params.main.dataset}/{self.params.main.kp_method}/eval-ransac/{self.split}_ransac_scene_idx_{scene_idx}.pt")
            print(labels.shape, preds.shape)
            print(labels.sum(), preds.sum())
            average_precision.reset()
            maximum_recall.reset()
            average_precision.update(preds[preds.nonzero()].squeeze(-1), labels[preds.nonzero()].squeeze(-1))
            maximum_recall.update(preds[preds.nonzero()].squeeze(-1), labels[preds.nonzero()].squeeze(-1))
            f1_score.update(preds[preds.nonzero()].squeeze(-1), labels[preds.nonzero()].squeeze(-1))
            overall_avg_prec[scene_idx] = average_precision.compute().item()
            overall_max_rec[scene_idx] = maximum_recall.compute()[0].item()
            overall_f1[scene_idx] = f1_score.compute().item()
            overall_ape[scene_idx] = average_relative_pose_error.compute().item()
            overall_ate[scene_idx] = average_translation_error.compute().item()
            print(f"Scene {scene_idx} AP: {overall_avg_prec[scene_idx]}, MR: {overall_max_rec[scene_idx]}, F1: {overall_f1[scene_idx]}, APE: {overall_ape[scene_idx]}, ATE: {overall_ate[scene_idx]}")

        metrics['test/avgprec'] = np.nanmean(list(overall_avg_prec.values()))
        metrics['test/max_rec'] = np.nanmean(list(overall_max_rec.values()))
        metrics['test/f1'] = np.nanmean(list(overall_f1.values()))
        metrics['test/ape'] = np.nanmean(list(overall_ape.values()))
        metrics['test/ate'] = np.nanmean(list(overall_ate.values()))
                                      
        return metrics
                    


@hydra.main(version_base=None, config_path="../../config", config_name="config_td2")
def main(params: DictConfig):

    inference_data = EpipolarRansacInference(params=params, scenes=params.data[params.main.dataset].test, split="test")
    result = inference_data.evaluate()
    print(result)

if __name__ == "__main__":
    main()
