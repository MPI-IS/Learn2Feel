"""
Module to define distances between probability distributions.
To add a distance, define a new class of type torch.nn.Module.
"""
import torch
import torch.nn as nn

class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D}` and :math:`P_2` locations 
    :math:`y\in\mathbb{R}^{D}` (and optionally their respective 
    marginal distributions :math:`\mu\in\mathbb{R}^{P_1}` and 
    :math:`\nu\in\mathbb{R}^{P_2}`),
    outputs an approximation of the regularized OT cost for point clouds.
    Adapted from `here <https://github.com/gpeyre/SinkhornAutoDiff>`_.

    :Args:
        - **eps** (*float*): regularization coefficient, defaults to 0.1.
        - **max_iter** (*int*): maximum number of Sinkhorn iterations, defaults to 50.
        - **p** (*float*): distance exponent (e.g. 2 for :math:`L_2` distance).
    
    :Shape:
        - Input: :math:`(N, P_1, D)`, :math:`(N, P_2, D)`, (optional: :math:`(N, P_1)`, :math:`(N, P_2)`)
        - Output: :math:`(N)` 
    """
    def __init__(self, 
                 eps = 0.1, 
                 max_iter = 50, 
                 p=1,
                 track_error=False):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.p = p
        self.track_error = track_error

    def forward(self, x, y, mu=None, nu=None):
        ""
        C = self._cost_matrix(x, y, self.p)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # BAR - add custom marginals
        # if marginals are not already defined, 
        # define both marginals with equal weights
        if mu is None:
            mu = torch.empty(batch_size, x_points, dtype=torch.float,
                             requires_grad=False).fill_(1.0 / x_points).squeeze().to(x.device)
        if nu is None:
            nu = torch.empty(batch_size, y_points, dtype=torch.float,
                             requires_grad=False).fill_(1.0 / y_points).squeeze().to(x.device)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        #error tracker
        if self.track_error:
            err_arr = []
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()
            if self.track_error:
                err_arr.append(err)  # type: ignore
            actual_nits += 1
            
            #print(err.item())
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.track_error:
            return cost.pow(1/(self.p)), pi, C, torch.tensor(err_arr)  # type: ignore
        return cost.pow(1/(self.p)), pi, C, None

    def M(self, C, u, v):
        r"""
        Modified cost for logarithmic updates
        $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$

        :meta private:
        """
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=1):
        r"Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

def distance_parser(distfunc_params):
    """
    Construct distance function class instance of the type defined in 
    ``distfunc_params.name``.

    :param dict distfunc_params: A dictionary with keys-values defining 
        all necessary parameters for the designated distance function.
        Requires the key 'name', which must be the name of a valid 
        distance function class. 
    :raises NotImplementedError: The distance function class is not defined.
    :return: <distance_class>(Module) 
    """
    distance_function_name = distfunc_params.pop('name',None)
    try:
        distfunc = globals()[distance_function_name]
    except:
        raise NotImplementedError('The distance function '
                                  f'{distance_function_name} '
                                  'is not implemented.')
    return distfunc(**distfunc_params)
