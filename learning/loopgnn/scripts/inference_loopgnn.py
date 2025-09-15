# torch.cuda.empty_cache()
# torch.cuda.memory_summary(device=None, abbreviated=False)
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
from torcheval.metrics import BinaryAUPRC, BinaryRecallAtFixedPrecision, Mean, BinaryF1Score

# from matching import get_matcher
# from matching.viz import plot_matches

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

        self.num_kp = params.preprocessing.kp[params.main.kp_method].num_kp
        self.desc_dim = params.preprocessing.kp[params.main.kp_method].desc_dim

        self.test_scenes = params.data[params.main.dataset].test

        self.test_graphdata = GraphDataset(self.params, self.test_scenes, split='test')
        self.test_graph_loader = torch_geometric.loader.DataLoader(self.test_graphdata, batch_size=1, num_workers=4, shuffle=False)
        

    def evaluate_trained_model(self, model):
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
            model.eval()
            for idx in tqdm.tqdm(range(len(self.test_graphdata)), total=len(self.test_graphdata), desc="Evaluating model"):

                edge_list, edge_feat_list, global_edge_list, edge_gt_list, edge_rel_poses = list(), list(), list(), list(), list()
                pot_loop_edges, query_edges = list(), list()
        
                scene_idx = self.test_graphdata.kf_data.sample2scene[idx]
                local_idx = idx - self.test_graphdata.kf_data.scene_idx_offsets[scene_idx]

                # Given the idx of the query frame, extract the top-k similar frames
                topk = min(int(self.params.preprocessing.retrieval_ratio * len(self.test_graphdata.kf_data.keyframes[scene_idx])), 100)
                neighborhood_ind = np.argpartition(self.test_graphdata.scene_vlad_sim[scene_idx][local_idx, :], topk)[-topk:]
                neighborhood_ind = np.append(neighborhood_ind, local_idx)

                sorted_indices = np.sort(neighborhood_ind)
                diffs = np.diff(sorted_indices)
                cluster_boundaries = np.where(diffs > 1)[0]
                clusters = np.split(sorted_indices, cluster_boundaries + 1)
                
                # assign nodes to clusters
                cluster_dict = {}
                for i, cluster in enumerate(clusters):
                    for node in cluster.tolist():
                        cluster_dict[node] = i

                # create interconnected graph among query_idx and its closest neighbors
                # do this per scene
                for i_rel, i in enumerate(neighborhood_ind):
                    for j_rel, j in enumerate(neighborhood_ind):
                        if i != j:
                            global_edge_list.append([i, j])
                            edge_list.append([i_rel, j_rel])
                            edge_feat_list.append(float(self.test_graphdata.scene_vlad_sim[scene_idx][i, j]))
                            edge_gt_list.append(self.test_graphdata.kf_data.gt_loop_matrix[scene_idx][i, j])
                            edge_rel_poses.append(torch.from_numpy(self.test_graphdata.kf_data.compute_rel_trafo_scene(i, j, scene_idx)))
                            if cluster_dict[i] == cluster_dict[j]:
                                pot_loop_edges.append(0)
                            else:    
                                pot_loop_edges.append(1)

                            if i == local_idx or j == local_idx:
                                query_edges.append(1)
                            else:
                                query_edges.append(0)

                node_feats = self.test_graphdata.kf_data.kp_descriptor_matrix[scene_idx][neighborhood_ind, :, :].reshape(-1, self.num_kp*self.desc_dim)
                node_poses = self.test_graphdata.kf_data.pose_matrix[scene_idx][neighborhood_ind, :, :].reshape(-1, 4*4)
                node_kps = self.test_graphdata.kf_data.kp_matrix[scene_idx][neighborhood_ind, :, :].reshape(-1, self.num_kp*2)
                global_edges = torch.tensor(global_edge_list).T.to(self.params.main.device)

                edge_feats = torch.tensor(edge_feat_list, dtype=torch.float32).unsqueeze(1)
                edge_gt = torch.tensor(edge_gt_list, dtype=torch.int32).contiguous()
                edges = torch.tensor(edge_list, dtype=torch.long).contiguous().T
                edge_rel_poses = torch.cat(edge_rel_poses).reshape(-1, 4*4).to(torch.float32)
                pot_loop_edges = torch.tensor(pot_loop_edges, dtype=torch.int32).contiguous()
                query_edges = torch.tensor(query_edges, dtype=torch.int32).contiguous()

                data = torch_geometric.data.Data(node_feats=node_feats,
                                                node_kps=node_kps,
                                                node_poses=node_poses,
                                                edge_index=edges,
                                                edge_attr=edge_feats,
                                                rel_poses=edge_rel_poses,
                                                pot_loop_edges=pot_loop_edges,
                                                query_edges=query_edges,
                                                y=edge_gt,
                                                ).to(self.params.main.device)

                pred_scores = model.forward(data)[0].squeeze(1)

                query_scores = pred_scores[data.query_edges == 1]
                query_gt = data.y[data.query_edges == 1]
                query_global_edges = global_edges[:, data.query_edges == 1]

                global_edges = torch.tensor(global_edge_list).T
                assert global_edges.shape[1] == data.edge_index.shape[1]
                # append predictions from current forward pass
                for local_edge_idx, edge in enumerate(query_global_edges.T):
                    self.pred_scores_dict[scene_idx][(edge[0].item(), edge[1].item())].append(query_scores[local_edge_idx].cpu().item())

        
        # turn pred_scores into numpy array
        for scene_idx in list(self.test_graphdata.scene_vlad_sim.keys()):
            for edge, score_list in self.pred_scores_dict[scene_idx].items():
                self.pred_score_arrays[scene_idx][edge] = np.mean(score_list) # catch undirected edge duplicates
                self.pred_score_arrays_median[scene_idx][edge] = np.median(score_list) # catch undirected edge duplicates
            save_path = f"data/{self.params.main.dataset}/{self.params.main.kp_method}/{self.params.inference.checkpoint.split('/')[-2]}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(torch.from_numpy(self.pred_score_arrays[scene_idx]), f"{save_path}/td2_test_gnn_sim_scene_idx_{scene_idx}.pt")

        overall_avg_prec, overall_max_rec, overall_f1 = dict(), dict(), dict()
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

        metrics['test/avgprec'] = np.mean(list(overall_avg_prec.values()))
        metrics['test/max_rec'] = np.mean(list(overall_max_rec.values()))
        metrics['test/f1'] = np.mean(list(overall_f1.values()))

        return metrics
                    


@hydra.main(version_base=None, config_path="../../config", config_name="config_td2")
def main(params: DictConfig):

    inference_data = LoopGNNInference(params=params, scenes=params.data[params.main.dataset].test, split="test")

    model = LoopGNN(params)
    # Define model checkpoint to continue training
    chckpt = torch.load(os.path.join(params.paths.project, params.inference.checkpoint), map_location=params.main.device)
    missing_keys, unexpected_keys = model.load_state_dict(chckpt, strict=False)
    model = model.to(params.main.device)

    result = inference_data.evaluate_trained_model(model)
    print(result)

if __name__ == "__main__":
    main()
