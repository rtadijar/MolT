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


def calculate_batchwise_distances(self, pos: torch.Tensor, batch: torch.Tensor = None):
    """
    :param pos: node position matrix
    :param batch: batch pointer that shows graph indexes in batch of graphs
    :return: torch.Tensor, pairwise distance matrix
    """
    if batch is None:
        batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    
    all_distances = torch.cdist(pos, pos)
    
    batch_mask = (batch[:, None] == batch[None, :])
    dist_mat = all_distances * batch_mask.float()
    
    return dist_mat


class SpatialBias(nn.Module):
    def __init__(self, num_heads:int, num_radial:int):
        """
        :param num_heads: number of attention heads
        :param num_radial: number of radial basis functions
        :param radial_min: minimum distance center
        :param radial_max: maximum distance center
        """
        super(NodeCentralityEncodingEmbedding, self).__init__()

        self.projection = nn.Linear(num_radial, num_heads)


    def forward(self, rb_embedding: torch.Tensor):
        """
        :param rb_embedding: radial basis functions
        :return: torch.Tensor, spatial bias matrix
        """
        spatial_bias = self.projection(rb_embedding)

        return spatial_bias.unsqueeze(-1) # [N, N, H, 1]


class CentralityEncoding(nn.Module):
    def __init__(self):
        super(CentralityEncoding, self).__init__()

    def forward(self, rb_embedding: torch.Tensor):
        """
        :param rb_embedding: spatial Encoding matrix, block diagonal matrix of radial basis functions
        :return: torch.Tensor, centrality Encoding matrix, sum of radial basis functions
        """
        centrality_encoding = torch.sum(rb_embedding, dim=-2)
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
        return edge_bias.unsqueeze(-1) # [N, N, H, 1]


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_encoding = EdgeEncoding(edge_dim, max_path_distance)

        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch.Tensor,
                edge_paths,
                ptr) -> torch.Tensor:
        """
        :param query: node feature matrix
        :param key: node feature matrix
        :param value: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after attention operation
        """
        batch_mask_neg_inf = torch.full(size=(query.shape[0], query.shape[0]), fill_value=-1e6).to(next(self.parameters()).device)
        batch_mask_zeros = torch.zeros(size=(query.shape[0], query.shape[0])).to(next(self.parameters()).device)

        # OPTIMIZE: get rid of slices: rewrite to torch
        if type(ptr) == type(None):
            batch_mask_neg_inf = torch.ones(size=(query.shape[0], query.shape[0])).to(next(self.parameters()).device)
            batch_mask_zeros += 1
        else:
            for i in range(len(ptr) - 1):
                batch_mask_neg_inf[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
                batch_mask_zeros[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1

        query = self.q(query)
        key = self.k(key)
        value = self.v(value)

        c = self.edge_encoding(query, edge_attr, edge_paths)
        a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        a = (a + b + c) * batch_mask_neg_inf
        softmax = torch.softmax(a, dim=-1) * batch_mask_zeros
        x = softmax.mm(value)
        return x


# FIX: sparse attention instead of regular attention, due to specificity of GNNs(all nodes in batch will exchange attention)
class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param num_heads: number of attention heads
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k, edge_dim, max_path_distance) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch.Tensor,
                edge_paths,
                ptr) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after all attention heads
        """
        return self.linear(
            torch.cat([
                attention_head(x, x, x, edge_attr, b, edge_paths, ptr) for attention_head in self.heads
            ], dim=-1)
        )


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, n_heads, max_path_distance):
        """
        :param node_dim: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix input number of dimension
        :param n_heads: number of attention heads
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=n_heads,
            edge_dim=edge_dim,
            max_path_distance=max_path_distance,
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        self.ff = nn.Linear(node_dim, node_dim)

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch,
                edge_paths,
                ptr) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """
        x_prime = self.attention(self.ln_1(x), edge_attr, b, edge_paths, ptr) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime

        return x_new
