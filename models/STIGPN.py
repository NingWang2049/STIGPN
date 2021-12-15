import os
import torch
import torch.nn as nn
import sys
import numpy as np
import dgl

import torch.nn.functional as F
from models.GAT import GAT

class VisualModelV(nn.Module):
    def __init__(self,args,out_type = None):
        super(VisualModelV, self).__init__()
        self.nr_boxes = args.nr_boxes
        self.nr_frames = args.nr_frames
        self.subact_classes = args.subact_classes
        self.afford_classes = args.afford_classes
        self.feat_drop = args.feat_drop
        self.attn_drop = args.attn_drop
        self.cls_dropout = args.cls_dropout

        self.embedding_feature_dim = 256
        self.res_feat_dim = 2048
        self.preprocess_dim = 1024
        self.out_dim = 512
        self.appearence_in_dim = 2*(self.embedding_feature_dim + self.preprocess_dim)
        
        #pre process
        self.appearence_preprocess = nn.Linear(self.res_feat_dim, self.preprocess_dim)
        #self.category_embed_layer = nn.Embedding(12, self.embedding_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)
        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.embedding_feature_dim//2, bias=False),
            nn.BatchNorm1d(self.embedding_feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_feature_dim//2, self.embedding_feature_dim, bias=False),
            nn.BatchNorm1d(self.embedding_feature_dim),
            nn.ReLU()
        )

        #edge_list = [(0,1),(0,2),(0,3),(0,4),(0,5)]
        edge_list = [(0,n) for n in range(1,self.nr_boxes)]
        src, dst = tuple(zip(*edge_list))
        self.spatial_graph = dgl.graph((src, dst))
        self.spatial_graph = dgl.to_bidirected(self.spatial_graph)
        self.spatial_graph = self.spatial_graph.to('cuda')
        
        node_list = [x for x in range(self.nr_boxes)]
        node_frame_list = []
        for f_idx in range(self.nr_frames):
            temp = []
            for n_idx in node_list:
                temp.append(f_idx*self.nr_boxes+n_idx)
            node_frame_list.append(temp)
        edge_list = []
        for i in range(self.nr_frames):
            for j in range(self.nr_frames):
                if i == j:
                    continue
                src_nodes = node_frame_list[i]
                dst_nodes = node_frame_list[j]
                for src in src_nodes:
                    for idx,dst in enumerate(dst_nodes):
                        # if idx == 0:
                        #     continue
                        edge_list.append((src,dst))
        src, dst = tuple(zip(*edge_list))
        temp = []
        for frame_idx in range(10):
            temp_ = []
            for idx,dst_idx in enumerate(dst):
                if dst_idx == frame_idx*6:
                    temp_.append(idx)
            temp.append(temp_)
            
        self.temporal_graph = dgl.graph((src, dst))
        self.temporal_graph = dgl.to_bidirected(self.temporal_graph)
        self.temporal_graph = self.temporal_graph.to('cuda')

        self.appearence_RNN = nn.RNN(input_size=self.appearence_in_dim, hidden_size=self.appearence_in_dim//2, num_layers=1, batch_first=True, bidirectional=True)
        self.appearence_RNN.flatten_parameters()

        self.appearence_spatial_subnet = GAT(self.appearence_in_dim,-1,self.out_dim,feat_drop=self.feat_drop,attn_drop=self.attn_drop,activation=nn.ReLU())
        #self.gat = GATConv(in_feats=self.appearence_in_dim,out_feats=self.out_dim,num_heads=1,feat_drop=self.feat_drop,attn_drop=self.attn_drop,residual=True,activation=nn.ReLU())
        self.spatial_temporal_subnet = GAT(self.appearence_in_dim,-1,self.out_dim,feat_drop=self.feat_drop,attn_drop=self.attn_drop,activation=nn.ReLU())
        
        # RNN blocks for frame-level temporal subnet
        self.subact_frame_RNN = nn.RNN(input_size=2*self.out_dim, hidden_size=2*self.out_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.afford_frame_RNN = nn.RNN(input_size=2*self.out_dim, hidden_size=2*self.out_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.subact_frame_RNN.flatten_parameters()
        self.afford_frame_RNN.flatten_parameters()

        self.classifier_human = nn.Sequential(
            nn.Linear(4*self.out_dim, 2*self.out_dim),
            # nn.BatchNorm1d(self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.out_dim, 512), #self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.subact_classes)
        )

        self.classifier_object = nn.Sequential(
            nn.Linear(4*self.out_dim, 2*self.out_dim),
            # nn.BatchNorm1d(self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.out_dim, 512), #self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.afford_classes)
        )
    
    def forward(self,num_objs, node_features, box_input, box_categories,out_type='scores'):
        batch_size = box_input.size(0)
        batch_spatial_graph = [self.spatial_graph for x in range(batch_size*self.nr_frames)]
        batch_spatial_graph = dgl.batch(batch_spatial_graph)

        batch_temporal_graph = [self.temporal_graph for x in range(batch_size)]
        batch_temporal_graph = dgl.batch(batch_temporal_graph)

        #spatial
        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(batch_size*self.nr_boxes*self.nr_frames, 4)
        spatial_feats = self.coord_to_feature(box_input)

        #appearence
        appearence_feats = self.appearence_preprocess(node_features.reshape(batch_size*self.nr_boxes*self.nr_frames,self.res_feat_dim))
        
        #appearence_spatial
        appearence_spatial_feats = torch.cat([spatial_feats, appearence_feats], dim=1)
        appearence_spatial_feats = appearence_spatial_feats.reshape(batch_size,self.nr_boxes,self.nr_frames,-1)
        appearence_spatial_node_feats = torch.zeros((batch_size, self.nr_boxes, self.nr_frames, self.appearence_in_dim)).float().cuda()
        appearence_spatial_node_feats[:,0,:,:self.appearence_in_dim//2] = appearence_spatial_feats[:,0,:,:]
        appearence_spatial_node_feats[:,1:,:,self.appearence_in_dim//2:] = appearence_spatial_feats[:,1:,:,:]

        appearence_spatial_node_feats = self.appearence_RNN(appearence_spatial_node_feats.reshape(batch_size*self.nr_boxes,self.nr_frames,self.appearence_in_dim))[0].reshape(batch_size,self.nr_boxes,self.nr_frames,self.appearence_in_dim).permute(0,2,1,3)

        appearence_spatial_node_feats = appearence_spatial_node_feats.reshape(batch_size*self.nr_frames*self.nr_boxes, self.appearence_in_dim)

        appearence_spatial_subnet_node_feats = self.appearence_spatial_subnet(batch_spatial_graph,appearence_spatial_node_feats)
        appearence_spatial_subnet_node_feats = appearence_spatial_subnet_node_feats.reshape(batch_size,self.nr_frames,self.nr_boxes, self.out_dim)
        
        appearence_spatial_temporal_node_feats = self.spatial_temporal_subnet(batch_temporal_graph,appearence_spatial_node_feats)
        appearence_spatial_temporal_node_feats = appearence_spatial_temporal_node_feats.reshape(batch_size,self.nr_frames,self.nr_boxes, self.out_dim)
        
        spatial_temproal_feats = torch.cat([appearence_spatial_subnet_node_feats,appearence_spatial_temporal_node_feats], dim=3)

        human_node_feats = spatial_temproal_feats[:, :, 0, :]

        obj_node_feats = []
        for b in range(batch_size):
            obj_feats = spatial_temproal_feats[b, :, 1: 1+num_objs[b], :]
            
            obj_node_feats.append(obj_feats)
    
        # obj_node_feats = []
        # for b in range(batch_size):
        #     obj_feats = spatial_graph[b, :, 1: 1+num_objs[b], :]

        #     concat_feats = torch.zeros((self.nr_frames, num_objs[b], 2*self.out_dim)).float().cuda()
        #     for o in range(num_objs[b]):
        #         concat_feats[:, o, :] = torch.cat((human_node_feats[b, :, :], obj_feats[:, o, :]), 1)

        #     obj_node_feats.append(concat_feats)
            
        obj_node_feats = torch.cat(obj_node_feats, dim=1)
        obj_node_feats = obj_node_feats.permute(1, 0, 2)

        ## Frame-level Temporal subnet
        human_rnn_feats = self.subact_frame_RNN(human_node_feats, None)[0]
        obj_rnn_feats = self.afford_frame_RNN(obj_node_feats, None)[0]

        subact_cls_scores = torch.sum(self.classifier_human(human_rnn_feats), dim=1)
        afford_cls_scores = torch.sum(self.classifier_object(obj_rnn_feats), dim=1)
        
        return subact_cls_scores, afford_cls_scores

class SemanticModelV(nn.Module):
    def __init__(self,args,out_type = None):
        super(SemanticModelV, self).__init__()
        self.nr_boxes = args.nr_boxes
        self.nr_frames = args.nr_frames
        self.subact_classes = args.subact_classes
        self.afford_classes = args.afford_classes
        self.feat_drop = args.feat_drop
        self.attn_drop = args.attn_drop
        self.cls_dropout = args.cls_dropout

        self.embedding_feature_dim = 256
        self.spatial_dim = 256
        self.semantic_in_dim = self.embedding_feature_dim+self.spatial_dim
        self.out_dim = 512
        
        self.category_embed_layer = nn.Embedding(12, self.embedding_feature_dim, padding_idx=0, scale_grad_by_freq=True)
        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.spatial_dim//2, bias=False),
            nn.BatchNorm1d(self.spatial_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.spatial_dim//2, self.spatial_dim, bias=False),
            nn.BatchNorm1d(self.spatial_dim),
            nn.ReLU()
        )

        #edge_list = [(0,1),(0,2),(0,3),(0,4),(0,5)]
        edge_list = [(0,n) for n in range(1,self.nr_boxes)]
        src, dst = tuple(zip(*edge_list))
        self.spatial_graph = dgl.graph((src, dst))
        self.spatial_graph = dgl.to_bidirected(self.spatial_graph)
        self.spatial_graph = self.spatial_graph.to('cuda')
        
        node_list = [x for x in range(self.nr_boxes)]
        node_frame_list = []
        for f_idx in range(self.nr_frames):
            temp = []
            for n_idx in node_list:
                temp.append(f_idx*self.nr_boxes+n_idx)
            node_frame_list.append(temp)
        edge_list = []
        for i in range(self.nr_frames):
            for j in range(self.nr_frames):
                if i == j:
                    continue
                src_nodes = node_frame_list[i]
                dst_nodes = node_frame_list[j]
                for src in src_nodes:
                    for idx,dst in enumerate(dst_nodes):
                        # if idx == 0:
                        #     continue
                        edge_list.append((src,dst))
        src, dst = tuple(zip(*edge_list))
        self.temporal_graph = dgl.graph((src, dst))
        self.temporal_graph = dgl.to_bidirected(self.temporal_graph)
        self.temporal_graph = self.temporal_graph.to('cuda')

        self.semantic_RNN = nn.RNN(input_size=self.semantic_in_dim, hidden_size=self.semantic_in_dim//2, num_layers=1, batch_first=True, bidirectional=True)
        self.semantic_RNN.flatten_parameters()

        self.semantic_spatial_subnet = GAT(self.semantic_in_dim,-1,self.out_dim,feat_drop=self.feat_drop,attn_drop=self.attn_drop,activation=nn.ReLU())
        #self.gat = GATConv(in_feats=self.semantic_in_dim,out_feats=self.out_dim,num_heads=1,feat_drop=self.feat_drop,attn_drop=self.attn_drop,residual=True,activation=nn.ReLU())
        self.spatial_temporal_subnet = GAT(self.semantic_in_dim,-1,self.out_dim,feat_drop=self.feat_drop,attn_drop=self.attn_drop,activation=nn.ReLU())
        
        # RNN blocks for frame-level temporal subnet
        self.subact_frame_RNN = nn.RNN(input_size=2*self.out_dim, hidden_size=2*self.out_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.afford_frame_RNN = nn.RNN(input_size=2*self.out_dim, hidden_size=2*self.out_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.subact_frame_RNN.flatten_parameters()
        self.afford_frame_RNN.flatten_parameters()

        self.classifier_human = nn.Sequential(
            nn.Linear(4*self.out_dim, 2*self.out_dim),
            # nn.BatchNorm1d(self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.out_dim, 512), #self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.subact_classes)
        )

        self.classifier_object = nn.Sequential(
            nn.Linear(4*self.out_dim, 2*self.out_dim),
            # nn.BatchNorm1d(self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.out_dim, 512), #self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.afford_classes)
        )
    
    def forward(self,num_objs, node_features, box_input, box_categories,out_type='scores'):
        batch_size = box_input.size(0)
        batch_spatial_graph = [self.spatial_graph for x in range(batch_size*self.nr_frames)]
        batch_spatial_graph = dgl.batch(batch_spatial_graph)

        batch_temporal_graph = [self.temporal_graph for x in range(batch_size)]
        batch_temporal_graph = dgl.batch(batch_temporal_graph)

        #spatial
        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(batch_size*self.nr_boxes*self.nr_frames, 4)
        spatial_feats = self.coord_to_feature(box_input)
        spatial_feats = spatial_feats.view(batch_size,self.nr_boxes,self.nr_frames, -1)

        #embedding
        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        box_categories = box_categories.view(batch_size*self.nr_boxes*self.nr_frames)
        box_category_embeddings = self.category_embed_layer(box_categories).view(batch_size,self.nr_boxes,self.nr_frames, self.embedding_feature_dim)
        
        #spatial_embedding
        spatial_embedding = torch.cat([spatial_feats,box_category_embeddings],dim=3)

        semantic_spatial_node_feats = self.semantic_RNN(spatial_embedding.reshape(batch_size*self.nr_boxes,self.nr_frames,self.semantic_in_dim))[0].reshape(batch_size,self.nr_boxes,self.nr_frames,self.semantic_in_dim).permute(0,2,1,3)

        semantic_spatial_node_feats = semantic_spatial_node_feats.reshape(batch_size*self.nr_frames*self.nr_boxes, self.semantic_in_dim)

        semantic_spatial_subnet_node_feats = self.semantic_spatial_subnet(batch_spatial_graph,semantic_spatial_node_feats)
        semantic_spatial_subnet_node_feats = semantic_spatial_subnet_node_feats.reshape(batch_size,self.nr_frames,self.nr_boxes, self.out_dim)
        
        semantic_spatial_temporal_node_feats = self.spatial_temporal_subnet(batch_temporal_graph,semantic_spatial_node_feats)
        semantic_spatial_temporal_node_feats = semantic_spatial_temporal_node_feats.reshape(batch_size,self.nr_frames,self.nr_boxes, self.out_dim)
        
        spatial_temproal_feats = torch.cat([semantic_spatial_subnet_node_feats,semantic_spatial_temporal_node_feats], dim=3)
        
        human_node_feats = spatial_temproal_feats[:, :, 0, :]

        obj_node_feats = []
        for b in range(batch_size):
            obj_feats = spatial_temproal_feats[b, :, 1: 1+num_objs[b], :]
            
            obj_node_feats.append(obj_feats)
    
        # obj_node_feats = []
        # for b in range(batch_size):
        #     obj_feats = spatial_graph[b, :, 1: 1+num_objs[b], :]

        #     concat_feats = torch.zeros((self.nr_frames, num_objs[b], 2*self.out_dim)).float().cuda()
        #     for o in range(num_objs[b]):
        #         concat_feats[:, o, :] = torch.cat((human_node_feats[b, :, :], obj_feats[:, o, :]), 1)

        #     obj_node_feats.append(concat_feats)
            
        obj_node_feats = torch.cat(obj_node_feats, dim=1)
        obj_node_feats = obj_node_feats.permute(1, 0, 2)

        ## Frame-level Temporal subnet
        human_rnn_feats = self.subact_frame_RNN(human_node_feats, None)[0]
        obj_rnn_feats = self.afford_frame_RNN(obj_node_feats, None)[0]

        subact_cls_scores = torch.sum(self.classifier_human(human_rnn_feats), dim=1)
        afford_cls_scores = torch.sum(self.classifier_object(obj_rnn_feats), dim=1)
        
        return subact_cls_scores, afford_cls_scores