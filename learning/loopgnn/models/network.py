import copy
from typing import List, Optional, Set

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from loopgnn.models.netvlad import NetVLAD, NetVLADLoupe
from torch import Tensor
from torch_geometric.nn.conv import (GATConv, GATv2Conv, GENConv,
                                     MessagePassing, TransformerConv)
from torch_scatter import gather_csr, scatter, segment_csr
from torch_sparse import SparseTensor


def build_mlp(dim_list, activation='relu', do_bn=False,
              dropout=0, on_last=False):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(torch.nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or on_last:
            if do_bn:
                layers.append(torch.nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(torch.nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(torch.nn.LeakyReLU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(p=dropout))
    return torch.nn.Sequential(*layers)


class TripletGCN(MessagePassing):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_feat_dim, aggr='mean', dropout_r=0.1, with_bn=True):
        super().__init__(aggr=aggr)
        # print('============================')
        # print('aggr:',aggr)
        # print('============================')
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_feat_dim = hidden_feat_dim
        self.nn1 = build_mlp([node_feat_dim*2+edge_feat_dim, hidden_feat_dim, hidden_feat_dim*2+edge_feat_dim],
                             do_bn=with_bn, on_last=True)
        self.nn2 = build_mlp([hidden_feat_dim, hidden_feat_dim, node_feat_dim], do_bn=with_bn)
        self.dropout = torch.nn.Dropout(p=dropout_r)
        self.reset_parameter()

    def reset_parameter(self):
        pass

    def forward(self, x, edge_feat, edge_index):
        gcn_x, gcn_e = self.propagate(edge_index,x=x, edge_feature=edge_feat)
        gcn_x = x + self.nn2(gcn_x)
        return gcn_x, gcn_e

    def message(self, x_i, x_j, edge_feature):
        x = torch.cat([x_i, edge_feature, x_j], dim=1)
        x = self.nn1(x)  # .view(b,-1)
        x =  self.dropout(x)
        new_x_i = x[:, :self.hidden_feat_dim]
        new_e = x[:, self.hidden_feat_dim:(self.hidden_feat_dim+self.edge_feat_dim)]
        new_x_j = x[:, (self.hidden_feat_dim+self.edge_feat_dim):]
        x = new_x_i+new_x_j
        return [x, new_e]

    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim,
                       dim_size=dim_size, reduce=self.aggr)
        return x


class TripletGCNModel(torch.nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, num_layers, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            self.gconvs.append(TripletGCN(**kwargs))

    def forward(self, node_feature, edge_feature, edges_indices):
        for i in range(self.num_layers):
            node_feature, edge_feature = self.gconvs[i](
                node_feature, edge_feature, edges_indices)
            if i < (self.num_layers-1):
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)
        return node_feature, edge_feature


class GAT(torch.nn.Module):
    def __init__(self, num_layers, dropout, **kwargs):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.gconvs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            self.gconvs.append(GATConv(**kwargs))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = F.elu(self.gconvs[i](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
class GenConv(torch.nn.Module):
    def __init__(self, num_layers, dropout, **kwargs):
        super(GenConv, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.gconvs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            self.gconvs.append(GENConv(**kwargs))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = F.elu(self.gconvs[i](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    

class TransConv(torch.nn.Module):
    def __init__(self, num_layers, dropout, **kwargs):
        super(TransConv, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.gconvs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            self.gconvs.append(TransformerConv(**kwargs))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = F.elu(self.gconvs[i](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
        
def tempsigmoid(x):
    nd=3.0 
    temp=nd/torch.log(torch.tensor(9.0)) 
    return torch.sigmoid(x/(temp)) 


import torch
import torch.nn as nn
import torch.nn.functional as F


class LoopGNN(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.depth = params.gnn.depth
        self.node_feat_dim = params.gnn.node_feat_dim
        self.K = torch.tensor(params.data[params.main.dataset].intrinsics).reshape(3,3)
        self.kp_method = params.main.kp_method
        self.num_kp = params.preprocessing.kp[self.kp_method].num_kp
        self.desc_dim = params.preprocessing.kp[self.kp_method].desc_dim
        

        self.netvlad_layer = NetVLADLoupe(feature_size=self.desc_dim, 
                                           cluster_size=params.gnn.netvlad.dim,
                                           output_dim=self.num_kp * 2,
                                           add_norm=False,
                                           )    
    
        self.node_encoder = nn.Sequential(
            nn.Linear(self.num_kp * 2, int(self.num_kp * 2/2)),
            nn.ReLU(),
            nn.Linear(int(self.num_kp * 2/2), params.gnn.node_feat_dim)
        )

        if self.depth == 0:
            self.edge_classifier = nn.Sequential(
                nn.Linear(int(params.gnn.node_feat_dim * 2 + params.gnn.edge_feat_dim), int(params.gnn.node_feat_dim * 2)),
                nn.ReLU(inplace=True),
                nn.Linear(int(params.gnn.node_feat_dim * 2), int(params.gnn.node_feat_dim * 2)),
                nn.ReLU(inplace=True),
                nn.Linear(int(params.gnn.node_feat_dim * 2), params.gnn.node_feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(params.gnn.node_feat_dim, params.gnn.node_feat_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(params.gnn.node_feat_dim // 2, params.gnn.node_feat_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(params.gnn.node_feat_dim // 4, params.gnn.node_feat_dim // 8),
                nn.ReLU(inplace=True),
                nn.Linear(params.gnn.node_feat_dim // 8, 1),
                nn.Sigmoid()
            )
        elif self.depth > 0:
            self.edge_classifier = nn.Sequential(
                nn.Linear(params.gnn.edge_feat_dim, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

        self.edge_encoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, params.gnn.edge_feat_dim),
            
        )

        # self.gconv_net = TripletGCNModel(num_layers=params.gnn.depth,
        #                                     node_feat_dim=params.gnn.node_feat_dim,
        #                                     edge_feat_dim=params.gnn.edge_feat_dim,
        #                                     hidden_feat_dim=params.gnn.hidden_feat_dim,
        #                                     aggr=params.gnn.aggr,
        #                                     dropout_r=params.gnn.dropout_rate,
        #                                     )
        
        # self.inter_gconv_net = TripletGCNModel(num_layers=params.gnn.depth,
        #                             node_feat_dim=params.gnn.node_feat_dim,
        #                             edge_feat_dim=params.gnn.edge_feat_dim,
        #                             hidden_feat_dim=params.gnn.hidden_feat_dim,
        #                             aggr=params.gnn.aggr,
        #                             dropout_r=params.gnn.dropout_rate,
        #                             )
        
        self.gatconv_net = GAT(num_layers=params.gnn.depth, 
                               dropout=params.gnn.dropout_rate,
                               in_channels=params.gnn.node_feat_dim,
                               out_channels=params.gnn.node_feat_dim, 
                               heads=1, 
                               )
        
        # self.gatconv_net = TransConv(num_layers=6, 
        #                 dropout=params.gnn.dropout_rate,
        #                 in_channels=params.gnn.node_feat_dim,
        #                 out_channels=params.gnn.node_feat_dim, 
        #                heads=1, 
        #                 )

        self.pose_threshold = nn.Parameter(torch.tensor(0.0))

        self.downproject_edges = nn.Linear(params.gnn.node_feat_dim * 2 + params.gnn.edge_feat_dim, params.gnn.edge_feat_dim)
        self.downproject_to_kp = nn.Linear(params.gnn.node_feat_dim, self.desc_dim)

        self.fuse_global_local_kp = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, self.desc_dim),
        )
        # self.keypoint_attention  = nn.MultiheadAttention(embed_dim=64, num_heads=1, batch_first=True)

    def forward(self, data):

        node_feats, node_kp, edge_index, edge_attr, pot_loop_edges = (
            data.node_feats,
            data.node_kps,
            data.edge_index,
            data.edge_attr,
            data.pot_loop_edges,
        )

        # reshape node_feats and feed through netvlad layer
        node_kp = node_kp.reshape(-1, self.num_kp, 2)
        node_feats = torch.nan_to_num(node_feats.reshape(-1, self.num_kp, self.desc_dim), nan=0.0)
        
        # input_feats = node_feats + self.kp_encoder(node_kp)
        vlad_node_feats = self.netvlad_layer.forward(node_feats)
        x = self.node_encoder(vlad_node_feats)

        # initial_x = copy.copy(x)
        edge_feats = self.edge_encoder(edge_attr)

        if self.depth > 0:
            # x, edge_feats = self.gconv_net.forward(node_feature=x, edge_feature=edge_feats, edges_indices=edge_index)
            x = self.gatconv_net.forward(x, edge_index)
            edge_feats = self.downproject_edges.forward(torch.cat([x[edge_index[0]], x[edge_index[1]], edge_feats], dim=1))
        else:
            edge_feats = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_feats], dim=1)

        edge_scores = self.edge_classifier(edge_feats)
        
        return [edge_scores]
    
        
