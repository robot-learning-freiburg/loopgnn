import os
from collections import defaultdict

import hydra
import kornia
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torcheval.metrics import BinaryAUPRC, BinaryRecallAtFixedPrecision, Mean
from tqdm import tqdm

torch.multiprocessing.set_start_method('spawn', force=True)
import torch.nn.functional as F
import torch.utils.data
import torch_geometric.data
from loopgnn.data.graph_data import GraphDataset
from loopgnn.models.network import LoopGNN
from loopgnn.utils.metrics import compute_max_recall

torch.cuda.empty_cache()

class LoopGNNTrainer:

    def __init__(self, params: DictConfig):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.params = params
        self.epochs = params.gnn.num_epochs
        self.epoch = 0
        print(f"Initializing LoopGNN training with {self.params.main.kp_method} KPs and depth {self.params.gnn.depth}")

        self.gnn = LoopGNN(self.params)
        # Define model checkpoint to continue training
        # chckpt = torch.load(os.path.join(params.paths.models, params.predict.checkpoint), map_location=self.device)
        # missing_keys, unexpected_keys = self.gnn.load_state_dict(chckpt, strict=False)
        self.gnn = self.gnn.to(self.device)
        

        self.train_scenes = params.data[params.main.dataset].train
        self.test_scenes = params.data[params.main.dataset].test

        self.train_graphdata = GraphDataset(self.params, self.train_scenes, split='train')
        self.train_graph_loader = torch_geometric.loader.DataLoader(self.train_graphdata, batch_size=self.params.gnn.batch_size, num_workers=2, shuffle=True)

        self.val_graphdata = GraphDataset(self.params, self.test_scenes, split='test')
        self.val_graph_loader = torch_geometric.loader.DataLoader(self.val_graphdata, batch_size=self.params.gnn.batch_size, num_workers=1, shuffle=True)
    

    def train(self):

        torch.cuda.empty_cache()

        optimizer = torch.optim.Adam(self.gnn.parameters(),
                                        lr=float(self.params.gnn.lr),
                                        weight_decay=float(self.params.gnn.weight_decay),
                                        betas=(self.params.gnn.beta_low, self.params.gnn.beta_high))

        average_precision = BinaryAUPRC()
        maximum_recall = BinaryRecallAtFixedPrecision(min_precision=1.0)

        print(f"Training GNN: topk{self.params.preprocessing.retrieval_ratio}, b{self.params.gnn.batch_size}, lr{self.params.gnn.lr}, wd{self.params.gnn.weight_decay}, beta{self.params.gnn.beta_low, self.params.gnn.beta_high}")
        for epoch in range(self.epochs):
            self.epoch += 1
            
            if self.params.gnn.train:
                self.gnn.train()

                metrics = defaultdict(list)
                train_progress = tqdm(self.train_graph_loader)
                train_loss = 0.0

                if self.epoch >= 3:
                        # freeze main GNN parameters
                        for name, param in self.gnn.named_parameters():
                            if "conv" in name or "edge" in name or "netvlad" in name:
                                param.requires_grad = False

                for train_it, data in enumerate(train_progress):
                    
                    data = data.to(self.device)
                    out = self.gnn.forward(data)
                
                    pred_scores = out[0].squeeze(-1)

                    if torch.isnan(pred_scores).any():
                        print("observed NaNs in network output")
                        binary_loss = 0.0
                    else:
                        binary_loss = F.binary_cross_entropy(pred_scores, data.y.float()) + 0.2 * F.binary_cross_entropy(pred_scores[data.query_edges == 1], data.y[data.query_edges == 1].float())
                    
                    loss = binary_loss
                
                    average_precision.update(pred_scores[data.query_edges == 1], data.y[data.query_edges == 1])
                    maximum_recall.update(pred_scores[data.query_edges == 1], data.y[data.query_edges == 1])
                    
                    train_loss += loss.item()

                    train_progress.set_description(
                        f"Loss: {loss.item():.4f}")

                    optimizer.zero_grad()
                    try: 
                        loss.backward(retain_graph=True)
                        optimizer.step()
                    except Exception as e:
                        print(e, "Error in backward pass, reducing to just binary loss")
                        loss = binary_loss
                        loss.backward()

                    # break episode after max_it_epoch iterations
                    if train_it > self.params.gnn.max_it_epoch:
                        break
                metrics['train/loss'] = train_loss / train_it
                metrics['train/avgprec'] = average_precision.compute().item()
                metrics['train/max_rec'] = maximum_recall.compute()[0].item()

                average_precision.reset()
                maximum_recall.reset()


            if self.params.gnn.eval:
                self.gnn.eval()

                with torch.no_grad():
                    
                    val_progress = tqdm(self.val_graph_loader)
                    val_loss = 0.0

                    for val_it, data in enumerate(val_progress):
                        
                        data = data.to(self.device)
                        out = self.gnn.forward(data)

                        pred_scores = out[0].squeeze(-1)

                        # try: 
                        if not torch.isnan(pred_scores).any():
                            binary_loss = F.binary_cross_entropy(pred_scores, data.y.float())
                            if torch.sum(data.query_edges == 1) > 0:
                                average_precision.update(pred_scores[data.query_edges == 1], data.y[data.query_edges == 1])
                                maximum_recall.update(pred_scores[data.query_edges == 1], data.y[data.query_edges == 1])
                            
                        else:
                            print(f"observed NaNs in network output {torch.isnan(pred_scores).any()}, {torch.isnan(data.y).any()}")

                        loss = binary_loss
                        
                        val_loss += loss.item()

                        val_progress.set_description(f"Loss: {loss.item():.4f}")  
                        if val_it > self.params.gnn.max_it_epoch:
                            break

                    metrics['val/loss'] = val_loss / val_it
                    metrics['val/avgprec'] = average_precision.compute().item()
                    metrics['val/max_rec'] = maximum_recall.compute()[0].item()
                    average_precision.reset()
                    maximum_recall.reset()
            
            tqdm.write(f"Epoch {epoch}:")
            tqdm.write('train/avgprec: ' + str(metrics['train/avgprec']))
            tqdm.write('train/max_rec: ' + str(metrics['train/max_rec']))
            tqdm.write('val/avgprec: ' + str(metrics['val/avgprec'])) if self.params.gnn.eval else None
            tqdm.write('val/max_rec: ' + str(metrics['val/max_rec'])) if self.params.gnn.eval else None
            wandb.log(metrics)
            
            if epoch > -1 and self.params.gnn.save_checkpoint:
                checkpoint_folder = os.path.join(self.params.paths.checkpoints,
                                               wandb.run.name)
                if not os.path.exists(checkpoint_folder):
                    os.makedirs(checkpoint_folder, exist_ok=True)
                torch.save(self.gnn.state_dict(), os.path.join(checkpoint_folder, f"gnn_ep{epoch}.pth"))

            del metrics


@hydra.main(version_base=None, config_path="../../config", config_name="config_td2")
def main(params: DictConfig):

    wandb.init(project="[loopgnn]",
            entity="add-yours",
            config=OmegaConf.to_container(params),
            mode="online" if params.main.wandb else "disabled",
            )

    if params.main.wandb:
        wandb_run_name = wandb.run.name

    trainer = LoopGNNTrainer(params)
    trainer.train()

    wandb.run.finish()

if __name__ == "__main__":
    main()
