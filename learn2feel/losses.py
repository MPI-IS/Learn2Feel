import torch
import torch.nn as nn
from scipy import stats
from fast_soft_sort import pytorch_ops

class spearmans_rank(nn.Module):
    r"""Compute soft spearman's correlation using differentiable soft ranking.

    The regularization strength determines how close the returned ranking 
    values are to the actual ranks. 

    :Args:
        - **regularization_strength** (*float*): The regularization strength to be used. The smaller this number, the closer the values are to the true ranks, defaults to .1 
        - **regularization** (*str*): Which regularization method to use. It must be set to one of ("l2", "kl", "log_kl"), defaults to 'l2'

    :Shape:
        - Input: :math:`(N)`, :math:`(N)`
        - Output: :math:`(1)`

    :return Tensor: The loss, defined as :math:`1.0-\rho_\mathrm{soft\_sp}` (soft spearman's correlation).
    """
    def __init__(self,
                 regularization_strength: float = .1,
                 regularization_type: str = 'l2'):
        super().__init__()
        self.rank_func = pytorch_ops.soft_rank
        self.reg_strength = regularization_strength
        self.reg_type = regularization_type

    def forward(self,dists,ratings) -> torch.Tensor:
        ""
        if len(dists.shape)==1:
            dists = dists.unsqueeze(0).to('cpu')
        hard_ranks_D = torch.tensor(stats.rankdata(dists.detach()),dtype=dists.dtype).unsqueeze(0)
        soft_ranks_D = self.rank_func(torch.cat([dists,hard_ranks_D]),
                                      regularization_strength = self.reg_strength,
                                      regularization = self.reg_type)[0]  #ranking function
        #prepare ratings and rating ranks
        rank_R = torch.tensor(stats.rankdata(ratings))
        #spearmans
        vD = soft_ranks_D - torch.mean(soft_ranks_D)
        vR = rank_R - torch.mean(rank_R)
        sp_rho = torch.sum(vD * vR)/(torch.sqrt(torch.sum(vD**2)) * torch.sqrt(torch.sum(vR**2)))
        loss = 1.0 - sp_rho # type: ignore
        return loss
