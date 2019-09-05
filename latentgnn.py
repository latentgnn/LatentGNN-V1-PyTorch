#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   latentgnn.py
@Time    :   2019/05/27 12:05:11
@Author  :   Songyang Zhang 
@Version :   1.0
@Contact :   sy.zhangbuaa@hotmail.com
@License :   (C)Copyright 2019-2020, PLUS Group@ShanhaiTech University
@Desc    :   None
'''

import torch

from lib.latentgnn_v1 import LatentGNNV1

def test_latentgnn():
    network = LatentGNNV1(in_channels=1024,
                        latent_dims=[100,100],
                        channel_stride=8,
                        num_kernels=2,
                        mode='asymmetric',
                        graph_conv_flag=False)
    
    dump_inputs = torch.rand((8,1024, 30,30))
    print(str(network))
    output = network(dump_inputs)


# if __name__ == "__main__":
    # test_latentgnn()
    # test_group_latentgnn()