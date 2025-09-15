import os
from collections import defaultdict
from typing import Dict, List

import hydra
import numpy as np
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



class VLADInference(torch_geometric.data.Dataset):
    """Characterizes a graph dataset to torch geometric."""

    def __init__(self, params: DictConfig, scenes: List[str], split: str):
        super(VLADInference, self).__init__()
        """
        Initializes a graph dataset.
        Args:
        """
        self.params = params
        self.scenes = scenes
        self.split = split

        self.test_scenes = params.data[params.main.dataset].test

        self.test_graphdata = GraphDataset(self.params, self.test_scenes, split='test')
        self.test_graph_loader = torch_geometric.loader.DataLoader(self.test_graphdata, batch_size=1, num_workers=4, shuffle=False)
        

    def evaluate(self):
        metrics = defaultdict(list)
        
        average_precision = BinaryAUPRC()
        maximum_recall = BinaryRecallAtFixedPrecision(min_precision=1.0)
        f1_score = BinaryF1Score(device=self.params.main.device)

        # construct multiple dicts holding the predicted scores per scene
        self.pred_scores_dict = dict()
        self.pred_score_arrays = dict()
        self.pred_score_arrays_median = dict()
        for scene_idx in list(self.test_graphdata.scene_vlad_sim.keys()):
            self.pred_scores_dict[scene_idx] = defaultdict(list)
            self.pred_score_arrays[scene_idx] = np.zeros((len(self.test_graphdata.kf_data.keyframes[scene_idx]), len(self.test_graphdata.kf_data.keyframes[scene_idx])))
            self.pred_score_arrays_median[scene_idx] = np.zeros((len(self.test_graphdata.kf_data.keyframes[scene_idx]), len(self.test_graphdata.kf_data.keyframes[scene_idx])))
        with torch.no_grad():
            for idx in tqdm.tqdm(range(len(self.test_graphdata)), total=len(self.test_graphdata), desc="Evaluating model"):

                edge_list, edge_feat_list, global_edge_list, edge_gt_list, edge_rel_poses = list(), list(), list(), list(), list()
                pot_loop_edges, query_edges = list(), list()
        
                scene_idx = self.test_graphdata.kf_data.sample2scene[idx]
                local_idx = idx - self.test_graphdata.kf_data.scene_idx_offsets[scene_idx]

                # Given the idx of the query frame, extract the top-k similar frames
                topk = min(int(self.params.preprocessing.retrieval_ratio * len(self.test_graphdata.kf_data.keyframes[scene_idx])), 100)
                neighborhood_ind = np.argpartition(self.test_graphdata.scene_vlad_sim[scene_idx][local_idx, :], topk)[-topk:]
                pred_scores = self.test_graphdata.scene_vlad_sim[scene_idx][local_idx, neighborhood_ind]

                # append predictions from current forward pass
                for local_n_idx, neighborhood_idx in enumerate(neighborhood_ind.T):
                    self.pred_scores_dict[scene_idx][(local_idx, neighborhood_idx)].append(pred_scores[local_n_idx])

        
        # turn pred_scores into numpy array
        for scene_idx in list(self.test_graphdata.scene_vlad_sim.keys()):
            for edge, score_list in self.pred_scores_dict[scene_idx].items():
                self.pred_score_arrays[scene_idx][edge] = np.mean(score_list) # catch undirected edge duplicates
                self.pred_score_arrays_median[scene_idx][edge] = np.median(score_list) # catch undirected edge duplicates
        # torch.save(torch.from_numpy(self.pred_scores_array), "data/td2/td2_test_gnn_sim.pt")

        overall_avg_prec = dict()
        overall_max_rec = dict()
        overall_f1 = dict()
        scene_labels = dict()
        scene_preds = dict()
        for scene_idx in list(self.test_graphdata.scene_vlad_sim.keys()):
            print(f"Scene {scene_idx}")
            labels = torch.from_numpy(self.test_graphdata.kf_data.gt_loop_matrix[scene_idx]).flatten().long()
            preds = torch.from_numpy(self.pred_score_arrays[scene_idx]).flatten()
            print(labels.shape, preds.shape)
            print(labels.sum(), preds.sum())
            scene_labels[scene_idx], scene_preds[scene_idx] = labels, preds
            average_precision.reset()
            maximum_recall.reset()
            average_precision.update(preds[preds.nonzero()].squeeze(-1), labels[preds.nonzero()].squeeze(-1))
            maximum_recall.update(preds[preds.nonzero()].squeeze(-1), labels[preds.nonzero()].squeeze(-1))
            f1_score.update(preds[preds.nonzero()].squeeze(-1), labels[preds.nonzero()].squeeze(-1))
            overall_avg_prec[scene_idx] = average_precision.compute().item()
            overall_max_rec[scene_idx] = maximum_recall.compute()[0].item()
            overall_f1[scene_idx] = f1_score.compute().item()
            print(f"Scene {scene_idx} AP: {overall_avg_prec[scene_idx]}, MaxRec: {overall_max_rec[scene_idx]}, F1: {overall_f1[scene_idx]}")

        metrics['test/avgprec'] = np.mean(list(overall_avg_prec.values()))
        metrics['test/max_rec'] = np.mean(list(overall_max_rec.values()))
        metrics['test/f1'] = np.mean(list(overall_f1.values()))

        return metrics
                    


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(params: DictConfig):

    inference_data = VLADInference(params=params, scenes=params.data[params.main.dataset].test, split="test")
    result = inference_data.evaluate()
    print(result)

if __name__ == "__main__":
    main()
