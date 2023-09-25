from typing import Tuple

import torch
from torch import nn
from torch_geometric.utils import degree

from graphormer.utils import decrease_to_max_value

class RadialBasisEmbedding(nn.Module):
    def __init__(self, num_radial: int, radial_min: float, radial_max: float):
        """
        :param num_radial: number of radial basis functions
        :param radial_min: minimum distance center
        :param radial_max: maximum distance center
        """
        super(RadialBasisEmbedding, self).__init__()

        self.num_radial = num_radial
        self.radial_min = radial_min
        self.radial_max = radial_max

        centers = torch.linspace(radial_min, radial_max, num_radial).view(-1, 1)
        widths = (centers[1] - centers[0]) * torch.ones_like(centers)

        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

    def forward(self, distance):
        """
        :param distance: pairwise distance vector/matrix
        :return: torch.Tensor, radial basis functions
        """
        shape = distance.shape
        distance = distance.view(-1, 1)
        rbf = torch.exp(-((distance.squeeze() - self.centers) ** 2) / (2 * self.widths ** 2))
        
        rbf = rbf.transpose(-1, -2)
        rbf = rbf.view(*shape, self.num_radial)

        return rbf


class SpatialBias(nn.Module):
    def __init__(self, num_heads:int, num_radial:int):
        """
        :param num_heads: number of attention heads
        :param num_radial: number of radial basis functions
        :param radial_min: minimum distance center
        :param radial_max: maximum distance center
        """
        super(SpatialBias, self).__init__()

        self.projection = nn.Linear(num_radial, num_heads)


    def forward(self, rb_embedding: torch.Tensor):
        """
        :param rb_embedding: radial basis functions
        :return: torch.Tensor, spatial bias matrix
        """
        spatial_bias = self.projection(rb_embedding)

        return spatial_bias.permute(2, 0, 1)


class CentralityEncoding(nn.Module):
    def __init__(self, num_radial: int, emb_dim: int):
        super(CentralityEncoding, self).__init__()

        self.projection = nn.Linear(num_radial, emb_dim)

    def forward(self, rb_embedding: torch.Tensor):
        """
        :param rb_embedding: spatial Encoding matrix, block diagonal matrix of radial basis functions
        :return: torch.Tensor, centrality Encoding matrix, sum of radial basis functions
        """
        centrality_encoding = self.projection(rb_embedding)
        centrality_encoding = torch.sum(centrality_encoding, dim=-2)
        
        return centrality_encoding # N, D


class EdgeBias(nn.Module):
    def __init__(self, num_heads:int, edge_attr_dim: int):
        """
        :param emb_dim: embedding dimension
        :param edge_attr_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.projection = nn.Linear(edge_attr_dim, num_heads)

    def forward(self, edge_attr: torch.Tensor):
        """
        :param edge_attr: edge feature matrix
        :return: torch.Tensor, edge bias matrix
        """
        edge_bias = self.projection(edge_attr)
        return edge_bias.permute(2, 0, 1) # [H, N, N]


class GraphormerMHA(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, num_radial: int, edge_attr_dim: int):
        """
        :param emb_dim: embedding dimension
        :param num_heads: number of attention heads
        :param num_radial: number of radial basis functions for spatial bias
        :param edge_attr_dim: edge attribute dimension
        """
        super().__init__()
        self.spatial_bias = SpatialBias(num_heads, num_radial)
        self.edge_bias = EdgeBias(num_heads, edge_attr_dim)

        self.q = nn.Linear(emb_dim, emb_dim)
        self.k = nn.Linear(emb_dim, emb_dim)
        self.v = nn.Linear(emb_dim, emb_dim)

        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.per_head_dim = emb_dim // num_heads

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                rb_embedding: torch.Tensor,
                edge_attr: torch.Tensor,
                ptr) -> torch.Tensor:
        """
        :param query: node feature matrix
        :param key: node feature matrix
        :param value: node feature matrix
        :param rb_embedding: spatial Encoding matrix, block diagonal matrix of radial basis functions
        :param edge_attr: edge feature matrix
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after attention operation
        """
        block_sizes = ptr[1:] - ptr[:-1]
        blocks = [torch.ones((size, size)) for size in block_sizes]
        block_diag_matrix = torch.block_diag(*blocks)

        batch_mask_zeros = block_diag_matrix.clone().unsqueeze(0)

        batch_mask_neg_inf = block_diag_matrix.clone()
        batch_mask_neg_inf[batch_mask_neg_inf == 0] = -1e6
        batch_mask_neg_inf = batch_mask_neg_inf.unsqueeze(0)

        query = self.q(query).reshape(-1, self.num_heads, self.per_head_dim) # [N, H, D // H]
        key = self.k(key).reshape(-1, self.num_heads, self.per_head_dim) # [N, H, D // H]
        value = self.v(value).reshape(-1, self.num_heads, self.per_head_dim) # [N, H, D // h]

        edge_bias = self.edge_bias(edge_attr)
        spatial_bias = self.spatial_bias(rb_embedding) 

        qk = query.permute(1, 0, 2).bmm(key.permute(1, 2,0)) / query.size(-1) ** 0.5 # [H, N, N]
        qk = (qk + edge_bias + spatial_bias) * batch_mask_neg_inf

        softmax = torch.softmax(qk, dim=-1) * batch_mask_zeros # [H, N, N]
        x = softmax.bmm(value.permute(1, 0, 2)) # [H, N, D]
        x = x.permute(1, 0, 2).reshape(-1, self.emb_dim) # [N, H, D // H] -> [N, D]

        return x

class GraphormerEncoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, num_radial, edge_attr_dim):
        """
        :param emb_dim: embedding dimension
        :param num_heads: number of attention heads
        :param num_radial: number of radial basis functions for spatial bias
        :param edge_attr_dim: edge attribute dimension
        """
        super().__init__()

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_radial = num_radial
        self.edge_attr_dim = edge_attr_dim

        self.attention = GraphormerMHA(
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_radial=num_radial,
            edge_attr_dim=edge_attr_dim
        )
        self.ln_1 = nn.LayerNorm(emb_dim)
        self.ln_2 = nn.LayerNorm(emb_dim)
        self.ff = nn.Linear(emb_dim, emb_dim)

    def forward(self,
                x: torch.Tensor,
                rb_embedding: torch.Tensor,
                edge_attr: torch.Tensor,
                ptr) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param rb_embedding: radial basis embedding
        :param edge_attr: edge feature matrix
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """
        x_prime = self.attention(self.ln_1(x), x, x, rb_embedding, edge_attr, ptr) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime

        return x_new