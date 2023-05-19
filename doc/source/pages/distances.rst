Distances
=========
.. contents::
    :local:
    :depth: 1

Introduction
------------
This module contains classes of type :py:class:`torch.nn.Module` 
that can be used to compute distances between two probability distributions 
in a differentiable manner. Currently, the only implemented distance is the 
entropy-regularized Wasserstein (optimal transport) distance solved with 
the Sinkhorn algorithm.

The :py:meth:`distance_parser<learn2feel.distances.distance_parser>` can 
be used to create an instance of any defined loss class. Just pass a 
dictionary into the config argument `probability_distance_params` with 
the structure 
``{'name': <name of distance function class>, <key: value> pairs for 
remaining parameters}``. 

New distance functions
----------------------
To define a new distance function, define or import a class of type 
:py:class:`torch.nn.Module` and write a constructor and 
``forward()``. 

.. code:: python

    class new_distance_function(nn.Module):
        def __init__(self, <additional parameters of distance function>):
            # constructor code
        ...

The ``forward()`` method should take at least four arguments. The first 
two should represent the right and left handed data, as in 
:py:class:`SinkhornDistance<learn2feel.distances.SinkhornDistance>`. If 
using the marginals, use the remaining two arguments.

.. code:: python

    class new_distance_function(nn.Module):    
        ...
        def forward(self, x, y, mu, nu):
            # compute distance on batch of samples using marginals mu and nu
        
If not using the marginals, then use placeholders for the last two 
arguments. 

.. code:: python

    class new_distance_function(nn.Module):    
        ...
        def forward(self, x, y, _, _):
            # compute distance on batch of samples ignoring marginals

References
----------
.. automodule:: learn2feel.distances
    :members:
    :member-order: bysource
    