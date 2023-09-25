from typing import Union

import torch
from torch import nn
from torch_geometric.data import Data

from graphormer.functional import calculate_batchwise_distances
from graphormer.layers import GraphormerEncoderLayer, CentralityEncoding, RadialBasisEmbedding


class Graphormer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 input_dim: int,
                 emb_dim: int,
                 input_edge_attr_dim: int,
                 edge_attr_dim: int,
                 output_dim: int,
                 num_radial: int,
                 radial_min: float,
                 radial_max: float,
                 num_heads: int,
                    ):
        """
        :param num_layers: number of Graphormer layers
        :param input_dim: input dimension of node features
        :param emb_dim: hidden dimensions of node features
        :param input_edge_attr_dim: input dimension of edge features
        :param edge_attr_dim: hidden dimensions of edge features
        :param output_dim: number of output node features
        :param num_heads: number of attention heads
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.input_edge_attr_dim = input_edge_attr_dim
        self.edge_attr_dim = edge_attr_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.num_radial = num_radial
        self.radial_min = radial_min
        self.radial_max = radial_max

        self.node_in_lin = nn.Linear(self.input_dim, self.emb_dim)
        self.edge_in_lin = nn.Linear(self.input_edge_attr_dim, self.edge_attr_dim)

        self.radial_basis = RadialBasisEmbedding(num_radial=self.num_radial, radial_min=self.radial_min, radial_max=self.radial_max)
        self.centrality_encoding = CentralityEncoding(num_radial=self.num_radial, emb_dim=self.emb_dim)

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                num_radial=self.num_radial,
                edge_attr_dim=self.edge_attr_dim
            ) 
            for _ in range(self.num_layers)
        ])

        self.node_out_lin = nn.Linear(self.emb_dim, self.output_dim)

    def forward(self, data: Union[Data]) -> torch.Tensor:
        """
        :param data: input graph or batch of graphs
        :return: torch.Tensor, output node embeddings
        """
        x = data.x.float()
        edge_attr = data.edge_attr.float()

        if 'batch' not in data:
            ptr = None
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        else:
            ptr = data.ptr
            batch = data.batch

        pos = data.pos

        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)

        dist_matrix = calculate_batchwise_distances(pos, batch)
        rb_embedding = self.radial_basis(dist_matrix)

        centrality_encoding = self.centrality_encoding(rb_embedding)
        x += centrality_encoding

        for layer in self.layers:
            x = layer(x, rb_embedding, edge_attr, ptr)

        x = self.node_out_lin(x)

        return x
