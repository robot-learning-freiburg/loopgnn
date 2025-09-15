# torch.cuda.empty_cache()
# torch.cuda.memory_summary(device=None, abbreviated=False)
import os
from typing import Dict, List

import numpy as np
import torch
import torch_geometric.data
# import torchvision.transforms as transforms
import tqdm
from loopgnn.data.nclt_keyframes import NCLTKeyframes
from loopgnn.data.td2_keyframes import TD2Keyframes
from loopgnn.utils.vlad import VLAD, compute_cosine_sim, get_top_k_recall
from omegaconf import DictConfig, OmegaConf


class GraphDataset(torch_geometric.data.Dataset):
    """Characterizes a graph dataset to torch geometric."""

    def __init__(self, params: DictConfig, scenes: List[str], split: str):
        super(GraphDataset, self).__init__()
        """
        Initializes a graph dataset.
        Args:
        """
        self.params = params
        self.scenes = scenes
        self.split = split
        self.dataset_path = self.params.paths.data[params.main.dataset]
        self.cross_seq_loops = self.params.preprocessing.cross_seq_loops[self.split]
        self.scene_wise = self.params.preprocessing.scene_wise[self.split]
        self.kp_method = self.params.main.kp_method
        self.desc_dim = self.params.preprocessing.kp[self.kp_method].desc_dim

        if params.main.dataset == "td2":
            self.kf_data = TD2Keyframes(self.dataset_path, 
                                        kp_method=self.kp_method,
                                        split=self.split, 
                                        seqs=self.scenes, 
                                        cross_seq_loops=self.cross_seq_loops, 
                                        scene_wise=self.scene_wise)
        elif params.main.dataset == "nclt":
            self.kf_data = NCLTKeyframes(self.dataset_path, 
                                         kp_method=self.kp_method,
                                         split=self.split, 
                                         seqs=self.scenes, 
                                         scene_wise=self.scene_wise, 
                                         cross_seq_loops=self.cross_seq_loops,
                                         crop_params=params.data[params.main.dataset].crop_params)
        self.kf_data.load_kp_descriptors(os.path.join(self.params.paths.project, "data")) 

        self.vlad = VLAD(num_clusters=64, 
                         desc_dim=self.desc_dim, 
                         dist_mode=self.params.preprocessing.kp[self.kp_method].vlad_dist_mode, 
                         vlad_mode=self.params.preprocessing.kp[self.kp_method].vlad_mode, 
                         cache_dir=os.path.join(params.paths.project, "data", params.main.dataset, self.kp_method))
        self.vlad.fit(train_descs=None)
        print("Loaded VLAD weights")
        
        self.global_descriptor_path = os.path.join(params.paths.project, "data", params.main.dataset, self.kp_method, f"{params.main.dataset}_{self.split}_global_descriptors.pt")
        if os.path.exists(self.global_descriptor_path):
            print(f"Loading global descriptors for {self.split} set")
            self.kf_data.load_global_descriptors(self.global_descriptor_path)
        else:
            print(f"Generating global descriptors for {self.split} set")
            self.kf_data.add_global_descriptors(self.vlad.generate_multi(self.kf_data.kp_descriptor_matrix.to(params.main.device)), save_path=self.global_descriptor_path)
        
        self.global_edge_registry = {}
        
        # go through all scenes and create highly connected cliques, through VLAD
        self.evaluated_kf_idcs = []

        # sample neighborhoods through VLAD retrieval
        vlad_sim_path = f"data/{self.params.main.dataset}/{self.kp_method}/{self.params.main.dataset}_{self.split}_dense_vlad_sim.pt"
        if os.path.exists(vlad_sim_path) and not self.params.main.recompute_vlad_sim:
            print(f"Loading dense VLAD similarities for {self.split} set")
            self.dense_vlad_sim = torch.load(vlad_sim_path, weights_only=True).numpy()
            if self.scene_wise:
                self.evaluated_kf_idcs = list(range(sum([len(kfs) for kfs in self.kf_data.keyframes.values()])))    
            else:
                self.evaluated_kf_idcs = list(range(len(self.kf_data.keyframes)))
        else:
            if not self.scene_wise:
                print("Extracting dense VLAD similarities per scene. This might take a little, but gets cached for later use.")
                self.dense_vlad_sim = np.zeros((len(self.kf_data.keyframes), len(self.kf_data.keyframes)))
                for i in tqdm.tqdm(range(len(self.kf_data.keyframes)), total=len(self.kf_data.keyframes), desc="Sampling neighborhoods"):
                    dist, idcs = compute_cosine_sim(qu=self.kf_data.global_desc[i],
                                            db=self.kf_data.global_desc,
                                            norm_descs=True)
                    self.dense_vlad_sim[i, idcs] = dist
                    # self.dense_vlad_sim[idcs, i] = dist
                    
                    # remove immediate neighbors to focus on more diverse correspondences
                    if self.split == "train" and self.params.preprocessing.remove_vicinal_kf:
                        self.dense_vlad_sim[i, i-10:i+10] = 0.0
                    self.evaluated_kf_idcs.append(i)
            
                torch.save(torch.from_numpy(self.dense_vlad_sim), vlad_sim_path)
            else:
                print("Extracting dense VLAD similarities across all considered scenes per split. This will take a while but gets cached for later use.")
                overall_num_keyframes = sum([len(kfs) for kfs in self.kf_data.keyframes.values()])
                self.dense_vlad_sim = np.zeros((overall_num_keyframes, overall_num_keyframes))
                # TODO: needs adaptation to scene-wise / non-scene-wise case
                for scene_idx, scene in enumerate(self.kf_data.sequences):
                    scene_len = len(self.kf_data.keyframes[scene_idx])
                    scene_idx_offset = self.kf_data.scene_idx_offsets[scene_idx]
                    scene_global_desc = self.kf_data.global_desc[0+scene_idx_offset:scene_len+scene_idx_offset]
                    assert scene_global_desc.shape[0] == scene_len, "Scene global desc shape mismatch length of scene"

                    for i in tqdm.tqdm(range(scene_len), total=scene_len, desc="Sampling neighborhoods"):
                        dist, idcs = compute_cosine_sim(qu=scene_global_desc[i],
                                                db=scene_global_desc,
                                                norm_descs=True)
                        self.dense_vlad_sim[i+scene_idx_offset, idcs+scene_idx_offset] = dist
                        
                        # remove immediate neighbors to focus on more diverse correspondences
                        if self.split == "train" and self.params.preprocessing.remove_vicinal_kf:
                            self.dense_vlad_sim[i+scene_idx_offset, i-10+scene_idx_offset:i+10+scene_idx_offset] = 0.0
                        self.evaluated_kf_idcs.append(i+scene_idx_offset)
                
                torch.save(torch.from_numpy(self.dense_vlad_sim), vlad_sim_path)
        print("Done with extracting neighborhoods")

        if self.scene_wise and not self.cross_seq_loops:
            self.scene_vlad_sim = dict()
            for scene_idx in np.unique(self.kf_data.sample2scene):
                scene_filter = self.kf_data.sample2scene == scene_idx
                self.scene_vlad_sim[scene_idx] = self.dense_vlad_sim[scene_filter, :][:,scene_filter]


    def __len__(self):
        return len(self.evaluated_kf_idcs)

    def __getitem__(self, idx):
        """
        Source a batch by index
        Args:
            idx: some index used by dataloader to identify a batch

        Returns:
            data: torch_geometric.data.Data (one batch of multiple frames, not a batch of batches in the torch sense)
        """

        edge_list, edge_feat_list, global_edge_list, edge_gt_list, edge_rel_poses = list(), list(), list(), list(), list()
        pot_loop_edges, query_edges = list(), list()

        if self.scene_wise and not self.cross_seq_loops:
            scene_idx = self.kf_data.sample2scene[idx]
            local_idx = idx - self.kf_data.scene_idx_offsets[scene_idx]

            # Given the idx of the query frame, extract the top-k similar frames
            topk = min(int(self.params.preprocessing.retrieval_ratio * len(self.kf_data.keyframes[scene_idx])), 1000)
            neighborhood_ind = np.argpartition(self.scene_vlad_sim[scene_idx][local_idx, :], topk)[-topk:]
            neighborhood_ind = np.append(neighborhood_ind, local_idx)
        else:
            topk = min(int(self.params.preprocessing.retrieval_ratio * len(self.kf_data.keyframes)), 1000)
            scene_idx = self.kf_data.sample2scene[idx]
            scene_filter = self.kf_data.sample2scene == scene_idx if not self.cross_seq_loops else np.ones(len(self.kf_data.keyframes), dtype=bool)
            neighborhood_ind = np.argpartition(self.dense_vlad_sim[idx, scene_filter], topk)[-topk:]

            # if not considering cross-seq edges, translate indices based on scene-wise offsets
            if not self.cross_seq_loops:
                neighborhood_ind = neighborhood_ind + self.kf_data.scene_idx_offsets[scene_idx]
            neighborhood_ind = np.append(neighborhood_ind, idx)

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
        if self.scene_wise:
            # do this per scene
            for i_rel, i in enumerate(neighborhood_ind):
                for j_rel, j in enumerate(neighborhood_ind):
                    if i != j:
                        # global_edge_list.append([i, j])
                        edge_list.append([i_rel, j_rel])
                        edge_feat_list.append(float(self.scene_vlad_sim[scene_idx][i, j]))
                        edge_gt_list.append(self.kf_data.gt_loop_matrix[scene_idx][i, j])
                        edge_rel_poses.append(torch.from_numpy(self.kf_data.compute_rel_trafo_scene(i, j, scene_idx)))
                        if cluster_dict[i] == cluster_dict[j]:
                            pot_loop_edges.append(0)
                        else:    
                            pot_loop_edges.append(1)

                        if i == idx or j == idx:
                            query_edges.append(1)
                        else:
                            query_edges.append(0)

            node_feats = self.kf_data.kp_descriptor_matrix[scene_idx][neighborhood_ind, :, :].reshape(-1, 
                                                                                                      self.params.preprocessing.kp[self.kp_method].num_kp * self.params.preprocessing.kp[self.kp_method].desc_dim)
            node_poses = self.kf_data.pose_matrix[scene_idx][neighborhood_ind, :, :].reshape(-1, 4*4)
            node_kps = self.kf_data.kp_matrix[scene_idx][neighborhood_ind, :, :].reshape(-1, self.params.preprocessing.kp[self.kp_method].num_kp*2)
            

        else:
            # perform this globally
            for i_rel, i in enumerate(neighborhood_ind):
                for j_rel, j in enumerate(neighborhood_ind):
                    if i != j:
                        global_edge_list.append([i, j])
                        edge_list.append([i_rel, j_rel])
                        edge_feat_list.append(float(self.dense_vlad_sim[i, j]))
                        edge_gt_list.append(self.kf_data.gt_loop_matrix[i, j])
                        edge_rel_poses.append(torch.from_numpy(self.kf_data.compute_rel_trafo(i, j)))
                        if cluster_dict[i] == cluster_dict[j]:
                            pot_loop_edges.append(0)
                        else:    
                            pot_loop_edges.append(1)

                        if i == idx or j == idx:
                            query_edges.append(1)
                        else:
                            query_edges.append(0)

            node_feats = self.kf_data.kp_descriptor_matrix[neighborhood_ind, :, :].reshape(-1, self.params.preprocessing.kp[self.kp_method].num_kp * self.desc_dim)
            node_poses = self.kf_data.pose_matrix[neighborhood_ind, :, :].reshape(-1, 4*4)
            node_kps = self.kf_data.kp_matrix[neighborhood_ind, :, :].reshape(-1, self.params.preprocessing.kp[self.kp_method].num_kp*2)

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
                                        #  batch_idx=idx,
                                        )
            
        return data
