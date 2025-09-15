import os
import time
from typing import List, Tuple

import hydra
import numpy as np
import torch
import tqdm
from loopgnn.data.td2_keyframes import TD2Keyframes
from loopgnn.utils.keypoints import get_extractor
from loopgnn.utils.vlad import VLAD, compute_retrieval_recall, get_top_k_recall
from omegaconf import DictConfig, OmegaConf

# set seed
np.random.seed(42)
torch.manual_seed(42)

@hydra.main(version_base=None, config_path="../../config", config_name="config_td2")
def main(params: DictConfig):

    # Load keypoint extractor model
    num_kp = params.preprocessing.kp[params.main.kp_method].num_kp
    desc_dim = params.preprocessing.kp[params.main.kp_method].desc_dim
    extractor = get_extractor(params.main.kp_method, device=params.main.device, max_num_keypoints=num_kp)
    img_size = params.preprocessing.img_size

    dataset_path = params.paths.data[params.main.dataset]

    train_scenes = params.data[params.main.dataset].train
    test_scenes = params.data[params.main.dataset].test

    data_train = TD2Keyframes(dataset_path, 
                              split="train", 
                              seqs=train_scenes, 
                              scene_wise=params.preprocessing.scene_wise["train"], 
                              cross_seq_loops=params.preprocessing.cross_seq_loops["train"])
    data_test = TD2Keyframes(dataset_path, 
                             split="test", 
                             seqs=test_scenes, 
                             scene_wise=params.preprocessing.scene_wise["test"], 
                             cross_seq_loops=params.preprocessing.cross_seq_loops["test"])

    save_path = os.path.join(params.paths.project, "data", params.main.dataset, params.main.kp_method)
    os.makedirs(save_path, exist_ok=True)

    data_train.kp_method = params.main.kp_method
    data_test.kp_method = params.main.kp_method

    train_descriptor_matrix, train_kp = data_train.generate_descriptors(extractor, num_kp, img_size, params.main.device, recompute=params.main.recompute_desc)
    test_descriptor_matrix, test_kp = data_test.generate_descriptors(extractor, num_kp, img_size, params.main.device, recompute=params.main.recompute_desc)

    vlad = VLAD(num_clusters=params.preprocessing.kp[params.main.kp_method].vlad_clusters, 
                desc_dim=desc_dim, 
                dist_mode=params.preprocessing.kp[params.main.kp_method].vlad_dist_mode, 
                vlad_mode=params.preprocessing.kp[params.main.kp_method].vlad_mode, 
                cache_dir=os.path.join(params.paths.project, "data", params.main.dataset, params.main.kp_method))

    start = time.time()
    # adjust the number of considered samples to fit into GPU memory (original: 10)
    vlad.fit(data_train.kp_descriptor_matrix.reshape(-1, train_descriptor_matrix.shape[-1])[::30]) 
    end = time.time()
    print(f"Fitting took {end-start} seconds")

    start = time.time()
    global_train_descriptors = []
    global_test_descriptors = []
    for i in tqdm.tqdm(range(train_descriptor_matrix.shape[0]), total=train_descriptor_matrix.shape[0]):
        non_nan_mask = ~torch.isnan(train_descriptor_matrix[i].reshape(-1, desc_dim)).any(dim=1)
        single_keyframe_descriptors = train_descriptor_matrix[i].reshape(-1, desc_dim)[non_nan_mask]
        global_train_descriptors.append(vlad.generate_multi(single_keyframe_descriptors.unsqueeze(0)))
    
    for i in tqdm.tqdm(range(test_descriptor_matrix.shape[0]), total=test_descriptor_matrix.shape[0]):
        non_nan_mask = ~torch.isnan(test_descriptor_matrix[i].reshape(-1, desc_dim)).any(dim=1)
        single_keyframe_descriptors = test_descriptor_matrix[i].reshape(-1, desc_dim)[non_nan_mask]
        global_test_descriptors.append(vlad.generate_multi(single_keyframe_descriptors.unsqueeze(0)))
    


    data_train.add_global_descriptors(torch.stack(global_train_descriptors).squeeze(1),
                                      save_path=os.path.join(save_path, f"{params.main.dataset}_train_global_descriptors.pt"))
    data_test.add_global_descriptors(torch.stack(global_test_descriptors).squeeze(1), 
                                     save_path=os.path.join(save_path, f"{params.main.dataset}_test_global_descriptors.pt"))
    end = time.time()
    print(f"Generating global descriptors at {(len(data_test.keyframes))/(end-start)}Hz")

    recall = compute_retrieval_recall(data_test)
    print(recall)



if __name__ == "__main__":
    main()
