import numpy as np
import scipy as sp
import torch
from captum.attr import IntegratedGradients
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv ,GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import os
import math
from collections import OrderedDict

import numpy as np
import torchvision.models as models
from GAT import GAT
from GraphAttentionLayer import GraphAttentionLayer
from sklearn.manifold import TSNE

class wModel2(torch.nn.Module):
    def __init__(self):
        super(wModel2, self).__init__()
        self.GCNconv1 = GCNConv(101, 10)
        self.GraphA = GAT(30,30,30,0.2,0.2,4)
        dims = 60  # 所有头总共需要的输入维度
        heads = 3  # 单注意力头的总共个数
        dropout_pro = 0.0  # 单注意力头
        self.attentionLayer = torch.nn.MultiheadAttention(embed_dim=dims, num_heads=heads)
        self.linear_kmer = torch.nn.Linear(84, 30)
        self.linear_vec = torch.nn.Linear(92, 101)
        # self.linear_DPCP_vec = torch.nn.Linear(30, 10)
        self.linear_out = torch.nn.Linear(1000, 1)
        self.x01_linear1 = torch.nn.Linear(21, 21)
        self.linear_x_DPCP =torch.nn.Linear(7, 5)
        self.linear1 = torch.nn.Linear(21, 1)
        self.linear2 = torch.nn.Linear(101, 2)
        self.convx_gat = nn.Conv1d(101, 30, kernel_size=21)
        self.convx_out = nn.Conv1d(101, 4, kernel_size=60)
        self.convx_out_shape = nn.Conv1d(101, 1, kernel_size=5)
        self.BN =nn.BatchNorm1d(10)
        self.BN30 = nn.BatchNorm1d(30)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.APool = torch.nn.AvgPool2d([100,29])
        self.RES = models.resnet18(pretrained=True)
        self.RES.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)#RES能处理单通道的序列特征
        self.maxpool2d_t_p = nn.MaxPool2d(kernel_size=(1,30), stride=1, padding=0, dilation=1)
        self.maxpool2d_t_end = nn.MaxPool2d(kernel_size=(4,60), stride=1, padding=0, dilation=1)
        self.tsne = TSNE(n_components=2)
        self.x01_linear1 = torch.nn.Linear(60, 60)
        self.BN1 = nn.BatchNorm1d(4)
        self.relu1 = nn.ReLU()
        self.BN2 = nn.BatchNorm1d(1)
        self.relu2 = nn.ReLU()
    def forward(self,data,epoch,TrainOrPre):
        #data:
        # 0：101,84-kmer
        # 1: 92,30 -vec
        # 2: 101,101-pair
        Y = data["Y"].to(torch.float32)
        shape = data["shape"].to(torch.float32)
        x_Kmer = data["Kmer"].to(torch.float32)
        x_Kmer = self.linear_kmer(x_Kmer)
        x_vec = data["vec"].to(torch.float32)
        x_vec = x_vec.permute(0,2,1)
        x_vec = self.linear_vec(x_vec)
        x_vec = x_vec.permute(0, 2, 1)
        x_pair = data["pair"].to(torch.float32)
        x_gat = torch.zeros_like(x_Kmer)
        for i in range(x_pair.size()[0]):
            a = x_pair[i]
            # a[a > 0.8] = 1
            # a[a <= 0.8] = 0
            e = x_Kmer[i]
            x_gat_1 = self.GraphA(e, a)
            x_gat_1 = self.BN30(x_gat_1)
            x_gat_1 = self.relu(x_gat_1)
            x_gat[i]=x_gat_1
        x_out = torch.cat((x_gat,x_vec), dim=2)    #32,101,60
        query = self.x01_linear1(x_out)
        key = self.x01_linear1(x_out)
        value = self.x01_linear1(x_out)
        attn_output, _ = self.attentionLayer(query, key, value)
        x_out = self.convx_out(x_out) #32,4,1
        # x_out = self.BN1(x_out)
        # x_out = self.relu1(x_out)
        batchlenth = shape.size(0)
        shape = shape.view(batchlenth,5, 101).permute(0,2,1)#32.101.5
        shape = self.convx_out_shape(shape.cuda(0)) #32,4,1
        # shape = self.BN2(shape)
        # shape = self.relu2(shape)
        x_out = torch.cat((x_out, shape), dim=1)

        x_out =x_out.sum(axis=1).squeeze()
        x_out = self.sigmoid(x_out)

        return x_out  # 759x1
