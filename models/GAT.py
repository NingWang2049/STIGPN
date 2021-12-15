import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from dgl.nn import GATConv

class unit_GAT(nn.Module):
    def __init__(self, in_channels, out_channels,feat_drop=0,attn_drop=0,activation=None):
        super(unit_GAT, self).__init__()
        self.gat = GATConv(in_feats=in_channels, out_feats=out_channels, num_heads=1,feat_drop=feat_drop,attn_drop=attn_drop,residual=True,activation=activation)
    
    def forward(self, graph, x):
        #batch_size,nodes,frame,feature_size = x.shape
        #y = x.reshape(batch_size*nodes*frame,feature_size)
        y = self.gat(graph,x)
        #y = y.reshape(batch_size,nodes,frame,-1)
        
        return y

class GAT(nn.Module):
    def __init__(self, in_channels, nhidden, out_channels,feat_drop=0,attn_drop=0,activation=None):
        super(GAT, self).__init__()

        #self.l1 = unit_GAT(in_channels, in_channels,feat_drop,attn_drop,activation)
        self.l2 = unit_GAT(in_channels, out_channels,feat_drop,attn_drop,activation)

    def forward(self, graph, x):
        #x = self.l1(graph, x)
        x = self.l2(graph, x)
        return x
