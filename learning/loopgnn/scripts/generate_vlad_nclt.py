import os
import time
from typing import List, Tuple

import hydra
import numpy as np
import torch
import tqdm
from loopgnn.data.nclt_keyframes import NCLTKeyframes
from loopgnn.utils.keypoints import get_extractor
from loopgnn.utils.vlad import VLAD, compute_retrieval_recall, get_top_k_recall
from omegaconf import DictConfig, OmegaConf

# set seed
np.random.seed(42)
torch.manual_seed(42)

@hydra.main(version_base=None, config_path="config", config_name="config_nclt")
def main(params: DictConfig):

    # Load keypoint extractor model
    device = params.main.device
    model_name = params.main.kp_method
    num_kp = params.preprocessing.kp[model_name].num_kp
    desc_dim = params.preprocessing.kp[model_name].desc_dim
    extractor = get_extractor(model_name, device=device, max_num_keypoints=num_kp)  # Choose any of the ~30+ matchers
    img_size = params.preprocessing.img_size
    
    dataset_path = params.paths.data[params.main.dataset]
    train_scenes = params.data[params.main.dataset].train
    test_scenes = params.data[params.main.dataset].test

    data_train = NCLTKeyframes(dataset_path, 
                              split="train", 
                              seqs=train_scenes, 
                              scene_wise=params.preprocessing.scene_wise["train"], 
                              cross_seq_loops=params.preprocessing.cross_seq_loops["train"],
                              crop_params=params.data[params.main.dataset].crop_params)
    data_test = NCLTKeyframes(dataset_path, 
                             split="test", 
                             seqs=test_scenes, 
                             scene_wise=params.preprocessing.scene_wise["test"], 
                             cross_seq_loops=params.preprocessing.cross_seq_loops["test"],
                             crop_params=params.data[params.main.dataset].crop_params)

    K = np.array(params.data[params.main.dataset].intrinsics).reshape(3, 3)
    data_train.kp_method = model_name
    data_test.kp_method = model_name
    data_train.set_crop_params_intrinsics(params.data[params.main.dataset].crop_params, img_size, K)
    data_test.set_crop_params_intrinsics(params.data[params.main.dataset].crop_params, img_size, K)
    train_descriptor_matrix, train_kp = data_train.generate_descriptors(extractor, num_kp, img_size, device, recompute=params.main.recompute_desc)
    test_descriptor_matrix, test_kp = data_test.generate_descriptors(extractor, num_kp, img_size, device, recompute=params._maybe_resolve_interpolation.recompute_desc)


    vlad = VLAD(num_clusters=params.preprocessing.kp[model_name].vlad_clusters, 
                desc_dim=desc_dim, 
                dist_mode=params.preprocessing.kp[model_name].vlad_dist_mode, 
                vlad_mode=params.preprocessing.kp[model_name].vlad_mode, 
                cache_dir=os.path.join(params.paths.project, "data", params.main.dataset, model_name))

    start = time.time()
    # extract non-nan rows of kp_descriptor_matrix
    non_nan_mask = ~torch.isnan(train_descriptor_matrix.reshape(-1, desc_dim)).any(dim=1)
    valid_descriptors = train_descriptor_matrix.reshape(-1, desc_dim)[non_nan_mask]
    vlad.fit(valid_descriptors[::10])
    end = time.time()
    print(f"Fitting took {end-start} seconds")

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
    
    assert len(global_train_descriptors) == train_descriptor_matrix.shape[0]
    assert len(global_test_descriptors) == test_descriptor_matrix.shape[0]

    start = time.time()
    data_train.add_global_descriptors(torch.stack(global_train_descriptors).squeeze(1),
                                      save_path=os.path.join(params.paths.project, "data", params.main.dataset, model_name, f"{params.main.dataset}_train_global_descriptors.pt"))
    data_test.add_global_descriptors(torch.stack(global_test_descriptors).squeeze(1), 
                                     save_path=os.path.join(params.paths.project, "data", params.main.dataset, model_name, f"{params.main.dataset}_test_global_descriptors.pt"))
    
    end = time.time()
    print(f"Generating test global descriptors at {(len(data_test.keyframes))/(end-start)}Hz")

    recall = compute_retrieval_recall(data_test)
    print(recall)



if __name__ == "__main__":
    main()
