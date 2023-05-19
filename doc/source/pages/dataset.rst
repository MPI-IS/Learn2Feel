Dataset
=======

.. contents::
    :local:
    :depth: 1

Introduction
------------
An instance of the class :py:class:`surface_pair_perception<learn2feel.dataset.surface_pair_perception>` 
contains a :py:class:`pandas.DataFrame` with each row containing a trial from 
the experiments described in the `paper <10.1109/TOH.2022.3212701>`_. Each 
row is labeled with the participant, right and left surfaces, similarity 
rating, and features that are extracted from partitions of the full sequences. 

The constructor can take a list of training indices that are used to 
compute standardization transformations. These transformations can also 
be computed after construction using 
:py:meth:`surface_pair_perception.set_transforms<learn2feel.dataset.surface_pair_perception.set_transforms>`.

Because the samples are not all the same length, 
:py:func:`collate_fn<learn2feel.dataset.collate_fn>` is used to create 
batches. To improve efficiency and process the samples as a single tensor, 
all samples in the batch are padded to the same length.

The function :py:func:`define_marginals<learn2feel.dataset.define_marginals>` 
is used to create marginal densities for each of the right and left handed 
sequences. Currently, probability mass is assigned uniformly across all 
points in the sequence. 

References
----------
.. automodule:: learn2feel.dataset
    :members:
    :member-order: bysource
    :exclude-members: __weakref__