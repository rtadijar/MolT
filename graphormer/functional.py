import torch

def calculate_batchwise_distances(pos: torch.Tensor, batch: torch.Tensor = None):
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
