Learn2Feel
=========

**Learn2Feel** is a `Python` and `Matlab` framework for learning to predict the perceptual similarity of two surfaces using the force and torque data gathered while a human blindly explored two surfaces with their fingertips.

Requirements
------------

- [Python 3.6+](https://www.python.org/)
- [Numpy](https://numpy.org/)
- [PyTorch](https://pytorch.org/)
- [Fast Differentiable Sorting and Ranking](https://github.com/google-research/fast-soft-sort)

Installation
------------

**CityGraph** releases can be instaled from [PyPI](https://pypi.org/):

```
$ pip install city-graph
```

Alternatively, one can also clone the repository and install the package locally:

```
$ git clone https://github.com/MPI-IS/CityGraph.git
$ cd CityGraph
$ pip install .
```

We strongly advise to install the package in a dedicated virtual environment.

Demos
-----
To train a model on all subjects, do: 

```
$ python main.py -c config.yaml
```

To train a model on a single subject, do: 

```
$ python main.py -c config.yaml --subject_ID=<1..10>
```


References
-------
**Learn2Feel** implements the algorithm described in:

Richardson, B. A., Vardar, Y., Wallraven, C., & Kuchenbecker, K. J.. (2022). *Learning to Feel Textures: Predicting Perceptual Similarities From Unconstrained Finger-Surface Interactions*, IEEE Transactions on Haptics, 15(4), pp.705-717, doi: 10.1109/TOH.2022.3212701.

If you use this code please cite this [article](https://ieeexplore.ieee.org/document/9913733).

Authors
-------
[Ben Richardson](https://github.com/benrichardson28),
Haptic Intelligence - Max Planck Institute for Intelligent Systems


License
-------

CC-BY-NC 4.0 (see LICENSE.md).

The Sinkhorn loss routine is adapted from the project [SinkhornAutoDiff](https://github.com/gpeyre/SinkhornAutoDiff).


Copyright
---------
© 2023, Max Planck Society / Software Workshop - Max Planck Institute for Intelligent Systems
