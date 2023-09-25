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


class SpatialEncoding(nn.Module):
    def __init__(self, node_dim: int, num_radial: int, radial_min: float, radial_max: float):
        """
        :param num_radial: number of radial basis functions
        :param radial_min: minimum distance center
        :param radial_max: maximum distance center
        """
        super(NodeCentralityEncodingEmbedding, self).__init__()
        self.radial_basis = RadialBasisEmbedding(num_radial, radial_min, radial_max)

        if node_dim != num_radial:
            self.projection = nn.Linear(num_radial, node_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, pos: torch.Tensor, batch: torch.Tensor = None):
        """
        :param pos: node position matrix
        :param batch: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, spatial Encoding matrix, block diagonal matrix of radial basis functions
        """
        pos = data.pos
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)

        dist_mat = self.calculate_batchwise_distances(pos, batch)
        spatial_encoding = self.radial_basis(dist_mat)
        spatial_encoding = self.projection(spatial_encoding)

        return spatial_encoding


    def calculate_batchwise_distances(self, pos, batch):
        """
        :param pos: node position matrix
        :param batch: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, pairwise distance matrix
        """
        all_distances = torch.cdist(pos, pos)
        
        batch_mask = (batch[:, None] == batch[None, :])
        dist_mat = all_distances * batch_mask.float()
        
        return dist_mat

class CentralityEncoding(nn.Module):
    def __init__(self):
        super(CentralityEncoding, self).__init__()

    def forward(self, spatial_encoding: torch.Tensor):
        """
        :param spatial_encoding: spatial Encoding matrix, block diagonal matrix of radial basis functions
        :return: torch.Tensor, centrality Encoding matrix, sum of radial basis functions
        """
        centrality_encoding = torch.sum(spatial_encoding, dim=-2)
        return centrality_encoding


def dot_product(x1, x2) -> torch.Tensor:
    return (x1 * x2).sum(dim=1)


class EdgeEncoding(nn.Module):
    def __init__(self, edge_dim: int, max_path_distance: int):
        """
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance
        self.edge_vector = nn.Parameter(torch.randn(self.max_path_distance, self.edge_dim))

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_paths) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param edge_paths: pairwise node paths in edge indexes
        :return: torch.Tensor, Edge Encoding matrix
        """
        cij = torch.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        for src in edge_paths:
            for dst in edge_paths[src]:
                path_ij = edge_paths[src][dst][:self.max_path_distance]
                weight_inds = [i for i in range(len(path_ij))]
                cij[src][dst] = dot_product(self.edge_vector[weight_inds], edge_attr[path_ij]).mean()

        cij = torch.nan_to_num(cij)
        return cij


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
