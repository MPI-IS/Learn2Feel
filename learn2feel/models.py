import torch
import torch.nn as nn
from typing import Optional, Union


class fc_mapping_function(nn.Module):
    r"""Model class for tactile feature mapping. To create a simple linear 
    mapping, simply leave hidden_dims, activation, and regularizations as 
    None.

    :Args:
        - **input_dim** (*int*): dimensionality of input feature space
        - **output_dim** (*int*): dimensionality of embedding space
        - **hidden_dims** (*int,list,optional*): number of nodes per hidden layer, defaults to None
        - **activation** (*str,optional*): activation function to use, defaults to None
        - **regularizations** (*list,optional*): regularization methods to use, defaults to None

    :Shape:
        - Input: :math:`(N, *, H_{in})`, where :math:`*` represents any number of
          dimensions (greater than 0) and :math:`H_{in}=\mathrm{input\_dim}`.
        - Output: :math:`(N, *, H_{in})`, where all but the last dimension
          match the input shape and :math:`H_{out}=\mathrm{output\_dim}`.
    
    :raises ValueError: Activation function is not defined. Must be one of
        'relu', 'leakyrelu', 'tanh', and 'sigmoid'.

    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: Optional[Union[int, list]]=None,
                 seed: Optional[int]=None,
                 activation: Optional[str]=None,
                 regularizations: Optional[Union[str, list]]=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        layers=[]
        if hidden_dims is not None:
            if isinstance(hidden_dims, int):
                hidden_dims = [hidden_dims]

            if activation=='relu':
                act_func = nn.ReLU()
            elif activation=='leakyrelu':
                act_func = nn.LeakyReLU()
            elif activation=='tanh':
                act_func = nn.Tanh()
            elif activation=='sigmoid':
                act_func = nn.Sigmoid()
            elif activation is None:
                act_func = None
            else:
                raise ValueError('Activation should be relu, leakyrelu,'
                                 'tanh, or sigmoid.')

            for hdim in hidden_dims:
                layers.append(nn.Linear(input_dim,hdim))
                if act_func is not None: layers.append(act_func)
                if regularizations is not None:
                    layers.append(Permute())
                    if 'batchnorm' in regularizations:
                        layers.append(nn.BatchNorm1d(hdim))
                    if 'dropout' in regularizations:
                        layers.append(nn.Dropout())
                    layers.append(Permute())
                input_dim = hdim

        layers.append(nn.Linear(input_dim,output_dim))
        self.layers=nn.Sequential(*layers)

    def forward(self,X):
        ""
        return self.layers(X)
        
class Permute(nn.Module):
    def __init__(self):
        super(Permute, self).__init__()
    def forward(self, x):
        return x.permute(0,2,1)