import copy
from typing import List, Optional, Set

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from loopgnn.models.netvlad import NetVLAD, NetVLADLoupe
from torch import Tensor
from torch_geometric.nn.conv import GATConv, GATv2Conv, MessagePassing, GENConv, TransformerConv
from torch_scatter import gather_csr, scatter, segment_csr
from torch_sparse import SparseTensor


import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import torchvision

import json

import matplotlib.pyplot as plt


# ---------------------- convolution dims helper functions  ----------------------

def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

def convtrans2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = ((img_size[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0],
                (img_size[1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1])
    return outshape

# ----------------------    ----------------------


def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.resblock1 = ResNetBlock(64, 64)
        self.resblock2 = ResNetBlock(64, 512)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 512)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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
        
        self.resnet = ResNetEncoder()

        self.edge_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, data):

        node_feats, node_kp, edge_index, edge_attr, pot_loop_edges = (
            data.node_feats,
            data.node_kps,
            data.edge_index,
            data.edge_attr,
            data.pot_loop_edges,
        )

        update_node_feats = self.resnet.forward(node_feats.reshape(-1, 3, 512, 512))

        edge_feats = update_node_feats[edge_index[0]] - update_node_feats[edge_index[1]]
        edge_scores = self.edge_classifier(edge_feats)

        return [edge_scores]
    
        
