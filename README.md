# LatentGNN-PyTorch
Official Implementation of ”LatentGNN: Learning Efficient Non-local Relations for Visual Recognition“ (ICML 2019)

# Introduction
Capturing long-range dependencies in feature representations is crucial for many visual recognition tasks. Despite recent successes of deep convolutional networks, it remains challenging to model non-local context relations between visual features. A promising strategy is to model the feature context by a fully-connected graph neural network (GNN), which augments traditional convolutional features with an estimated non-local context representation. However, most GNN-based approaches require computing a dense graph affinity matrix and hence have difficulty in scaling up to tackle complex real-world visual problems. In this work, we propose an efficient and yet flexible non-local relation representation based on a novel class of graph neural networks. Our key idea is to introduce a latent space to reduce the complexity of graph, which allows us to use a low-rank representation for the graph affinity matrix and to achieve a linear complexity in computation. Extensive experimental evaluations on three major visual recognition tasks show that our method outperforms the prior works with a large margin while maintaining a low computation cost.

# Installation
```
git clone https://github.com/latentgnn/LatentGNN-V1-PyTorch.git
cd LatentGNN-V1-PyTorch
python setup.py build develop
```

# Experiment with Detection
Please go to [maskrcnn-benchmark-latentgnn](https://github.com/latentgnn/maskrcnn-benchmark-latentgnn) to use the latentgnn for object detection on COCO dataset.

# Citations
Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference. 
```
@InProceedings{pmlr-v97-zhang19f,
  title = 	 {{L}atent{GNN}: Learning Efficient Non-local Relations for Visual Recognition},
  author = 	 {Zhang, Songyang and Yan, Shipeng and He, Xuming },
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning(ICML)},
  pages = 	 {7374--7383},
  year = 	 {2019},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Long Beach, California, USA},
  month = 	 {09--15 Jun},
  publisher = 	 {PMLR},
}
```