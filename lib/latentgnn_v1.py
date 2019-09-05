#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   latentgnn_v1.py
@Time    :   2019/05/27 13:39:43
@Author  :   Songyang Zhang 
@Version :   1.0
@Contact :   sy.zhangbuaa@hotmail.com
@License :   (C)Copyright 2019-2020, PLUS Group@ShanhaiTech University
@Desc    :   None
'''

import torch
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np

class LatentGNNV1(nn.Module):
    """
    Latent Graph Neural Network for Non-local Relations Learning

    Args:
        in_channels (int): Number of channels in the input feature 
        latent_dims (list): List of latent dimensions  
        channel_stride (int): Channel reduction factor. Default: 4
        num_kernels (int): Number of latent kernels used. Default: 1
        mode (str): Mode of bipartite graph message propagation. Default: 'asymmetric'.
        without_residual (bool): Flag of use residual connetion. Default: False
        norm_layer (nn.Module): Module used for batch normalization. Default: nn.BatchNorm2d.
        norm_func (function): Function used for normalization. Default: F.normalize
        graph_conv_flag (bool): Flag of use graph convolution layer. Default: False

    """
    def __init__(self, in_channels, latent_dims, 
                    channel_stride=4, num_kernels=1, 
                    mode='asymmetric', without_residual=False, 
                    norm_layer=nn.BatchNorm2d, norm_func=F.normalize,
                    graph_conv_flag=False):
        super(LatentGNNV1, self).__init__()
        self.without_resisual = without_residual
        self.num_kernels = num_kernels
        self.mode = mode
        self.norm_func = norm_func

        inter_channel = in_channels // channel_stride

        # Reduce the channel dimension for efficiency
        if mode == 'asymmetric':
            self.down_channel_v2l = nn.Sequential(
                                    nn.Conv2d(in_channels=in_channels, 
                                            out_channels=inter_channel,
                                            kernel_size=1, padding=0, bias=False),
                                    norm_layer(inter_channel),
            )

            self.down_channel_l2v = nn.Sequential(
                                    nn.Conv2d(in_channels=in_channels, 
                                            out_channels=inter_channel,
                                            kernel_size=1, padding=0, bias=False),
                                    norm_layer(inter_channel),
            )

        elif mode == 'symmetric':   
            self.down_channel = nn.Sequential(
                                    nn.Conv2d(in_channels=in_channels, 
                                            out_channels=inter_channel,
                                            kernel_size=1, padding=0, bias=False),
                                    norm_layer(inter_channel),
            )
            # nn.init.kaiming_uniform_(self.down_channel[0].weight, a=1)
            # nn.init.kaiming_uniform_(self.down_channel[0].weight, mode='fan_in')
        else:
            raise NotImplementedError

        # Define the latentgnn kernel
        assert len(latent_dims) == num_kernels, 'Latent dimensions mismatch with number of kernels'

        for i in range(num_kernels):
            self.add_module('LatentGNN_Kernel_{}'.format(i), 
                                LatentGNN_Kernel(in_channels=inter_channel, 
                                                num_kernels=num_kernels,
                                                latent_dim=latent_dims[i],
                                                norm_layer=norm_layer,
                                                norm_func=norm_func,
                                                mode=mode,
                                                graph_conv_flag=graph_conv_flag))
        # Increase the channel for the output
        self.up_channel = nn.Sequential(
                                    nn.Conv2d(in_channels=inter_channel*num_kernels,
                                                out_channels=in_channels,
                                                kernel_size=1, padding=0,bias=False),
                                    norm_layer(in_channels),
        )

        # Residual Connection
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, conv_feature):
        # Generate visible space feature 
        if self.mode == 'asymmetric':
            v2l_conv_feature = self.down_channel_v2l(conv_feature)
            l2v_conv_feature = self.down_channel_l2v(conv_feature)
            v2l_conv_feature = self.norm_func(v2l_conv_feature, dim=1)
            l2v_conv_feature = self.norm_func(l2v_conv_feature, dim=1)
        elif self.mode == 'symmetric':
            v2l_conv_feature = self.norm_func(self.down_channel(conv_feature), dim=1)
            l2v_conv_feature = None
        out_features = []
        for i in range(self.num_kernels):
            out_features.append(eval('self.LatentGNN_Kernel_{}'.format(i))(v2l_conv_feature, l2v_conv_feature))
        
        out_features = torch.cat(out_features, dim=1) if self.num_kernels > 1 else out_features[0]
        
        out_features = self.up_channel(out_features)

        if self.without_resisual:
            return out_features
        else:
            return conv_feature + out_features*self.gamma

class LatentGNN_Kernel(nn.Module):
    """
    A LatentGNN Kernel Implementation

    Args:

    """
    def __init__(self, in_channels, num_kernels,
                        latent_dim, norm_layer,
                        norm_func, mode, graph_conv_flag):
        super(LatentGNN_Kernel, self).__init__()
        self.mode = mode
        self.norm_func = norm_func
        #----------------------------------------------
        # Step1 & 3: Visible-to-Latent & Latent-to-Visible
        #----------------------------------------------

        if mode == 'asymmetric':
            self.psi_v2l = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels,
                                        out_channels=latent_dim,
                                        kernel_size=1, padding=0,
                                        bias=False),
                            norm_layer(latent_dim),
                            nn.ReLU(inplace=True),
            )
            # nn.init.kaiming_uniform_(self.psi_v2l[0].weight, a=1)
            # nn.init.kaiming_uniform_(self.psi_v2l[0].weight, mode='fan_in')
            self.psi_l2v = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels,
                                        out_channels=latent_dim,
                                        kernel_size=1, padding=0,
                                        bias=False),
                            norm_layer(latent_dim),
                            nn.ReLU(inplace=True),
            )

        elif mode == 'symmetric':
            self.psi = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels,
                                        out_channels=latent_dim,
                                        kernel_size=1, padding=0,
                                        bias=False),
                            norm_layer(latent_dim),
                            nn.ReLU(inplace=True),
            )

        #----------------------------------------------
        # Step2: Latent Messge Passing
        #----------------------------------------------
        self.graph_conv_flag = graph_conv_flag
        if graph_conv_flag:
            self.GraphConvWeight = nn.Sequential(
                            # nn.Linear(in_channels, in_channels,bias=False),
                            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
                            norm_layer(in_channels),
                            nn.ReLU(inplace=True),
                        )
            nn.init.normal_(self.GraphConvWeight[0].weight, std=0.01)

    def forward(self, v2l_conv_feature, l2v_conv_feature):
        B, C, H, W = v2l_conv_feature.shape

        # Generate Bipartite Graph Adjacency Matrix
        if self.mode == 'asymmetric':
            v2l_graph_adj = self.psi_v2l(v2l_conv_feature)
            l2v_graph_adj = self.psi_l2v(l2v_conv_feature)
            v2l_graph_adj = self.norm_func(v2l_graph_adj.view(B,-1, H*W), dim=2)
            l2v_graph_adj = self.norm_func(l2v_graph_adj.view(B,-1, H*W), dim=1)
            # l2v_graph_adj = self.norm_func(l2v_graph_adj.view(B,-1, H*W), dim=2)
        elif self.mode == 'symmetric':
            assert l2v_conv_feature is None
            l2v_graph_adj = v2l_graph_adj = self.norm_func(self.psi(v2l_conv_feature).view(B,-1, H*W), dim=1)

        #----------------------------------------------
        # Step1 : Visible-to-Latent 
        #----------------------------------------------
        latent_node_feature = torch.bmm(v2l_graph_adj, v2l_conv_feature.view(B, -1, H*W).permute(0,2,1))

        #----------------------------------------------
        # Step2 : Latent-to-Latent 
        #----------------------------------------------
        # Generate Dense-connected Graph Adjacency Matrix
        latent_node_feature_n = self.norm_func(latent_node_feature, dim=-1)
        affinity_matrix = torch.bmm(latent_node_feature_n, latent_node_feature_n.permute(0,2,1))
        affinity_matrix = F.softmax(affinity_matrix, dim=-1)

        latent_node_feature = torch.bmm(affinity_matrix, latent_node_feature)

        #----------------------------------------------
        # Step3: Latent-to-Visible 
        #----------------------------------------------
        visible_feature = torch.bmm(latent_node_feature.permute(0,2,1), l2v_graph_adj).view(B, -1, H, W)

        if self.graph_conv_flag:
            visible_feature = self.GraphConvWeight(visible_feature)

        return visible_feature