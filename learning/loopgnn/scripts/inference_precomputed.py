import os
from collections import defaultdict
from typing import Dict, List

import hydra
import numpy as np
import torch
import torch_geometric.data
# import torchvision.transforms as transforms
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



class LoopGNNInference(torch_geometric.data.Dataset):
    """Characterizes a graph dataset to torch geometric."""

    def __init__(self, params: DictConfig, scenes: List[str], split: str):
        super(LoopGNNInference, self).__init__()
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

        self.pred_score_arrays = dict()
        

    def evaluate_trained_model(self, predictions_path):
        metrics = defaultdict(list)
        
        average_precision = BinaryAUPRC()
        maximum_recall = BinaryRecallAtFixedPrecision(min_precision=1.0)
        f1_score = BinaryF1Score(device=self.params.main.device, threshold=0.8)

                
        # turn pred_scores into numpy array
        for scene_idx in list(self.test_graphdata.scene_vlad_sim.keys()):
            self.pred_score_arrays[scene_idx] = torch.load(f"{predictions_path}/test_sim_scene_idx_{scene_idx}.pt", weights_only=True).numpy()

        overall_avg_prec, overall_max_rec, overall_f1 = dict(), dict(), dict()
        scene_labels = dict()
        scene_preds = dict()
        for scene_idx in list(self.test_graphdata.scene_vlad_sim.keys()):
            print(f"Scene {scene_idx}")
            labels = torch.from_numpy(self.test_graphdata.kf_data.gt_loop_matrix[scene_idx]).flatten().long()
            preds = torch.from_numpy(self.pred_score_arrays_median[scene_idx]).flatten()
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

    inference_data = LoopGNNInference(params=params, scenes=params.data[params.main.dataset].test, split="test")
    result = inference_data.evaluate_trained_model(predictions_path="data/td2/outputs/youthful-flower-155")
    print(result)

if __name__ == "__main__":
    main()
