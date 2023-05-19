# Learn2Feel
<b> Learning to Feel Textures: Predicting Perceptual Similarities From Unconstrained Finger-Surface Interactions. </b>\
[[Paper](https://doi.org/10.1109/TOH.2022.3212701)] [[MPI Project Page](https://hi.is.mpg.de/research_projects/surface-interactions-as-probability-distributions-in-embedding-spaces)]


**Learn2Feel** is a `Python` framework for learning to predict the perceptual similarity of two surfaces using the force and torque data gathered while a human blindly explored two surfaces with their fingertips.


Installation
------------

To install the code, first create and activate a virtual or conda environment 

```
$ python3.7 -m venv .venv/learn2feel
$ . venv/learn2feel/bin/activate
```
```
$ conda create -n learn2feel python=3.7
$ conda activate learn2feel
```

Then clone the repository and install it using the `Makefile` (it will automatically 
download the data from the Max Planck data 
repository [Edmond](https://doi.org/10.17617/3.2HBHR8) ):

```
$ git clone https://github.com/MPI-IS/Learn2Feel.git
$ cd Learn2Feel
$ make
```
If you don't need to download the data, call `make learn2feel`.

We strongly advise to install the package in a dedicated virtual environment.

Test
----
To run the test for a single subject and all subjects, do:
```
$ train_learn2feel -c configs/test_sub.yaml 
$ train_learn2feel -c configs/test_gen.yaml 
```
Verify all fold models and results summary have been saved in `results/test_<subject/general>`.


Execution
-----
To train models on all subjects, make sure `subject_ID` is not set in
`configs/config.yaml` and do: 
```
$ train_learn2feel -c configs/config.yaml
```

To train models on a single subject, set `subject_ID` in
`configs/config.yaml` and repeat above OR do: 
```
$ train_learn2feel -c configs/config.yaml --subject_ID=<1..10>
```

Results and models will be stored in subdirectories of the `config.output_path` 
folder, which will be generated at runtime. 

To view the training and validation metrics, open tensorboard via
```
$ tensorboard --logdir <config.output_path (use '.' for current)> --port=6006
```
and open `localhost:6006` in your browser. A summary of results will be
stored in `config.output_path/summary.csv`.




Documentation
-------------
To build the `Sphinx` documentation:
```
$ pip install sphinx sphinx_rtd_theme
$ cd doc
$ make html
```
and open the file `build/html/index.html` in your web browser.


Citation
--------
```
@article{learn2feel:TOH:2022,
    title={Learning to Feel Textures: Predicting Perceptual Similarities From Unconstrained Finger-Surface Interactions}, 
    author={Richardson, Benjamin A. and Vardar, Yasemin and Wallraven, Christian and Kuchenbecker, Katherine J.},
    journal={IEEE Transactions on Haptics}, 
    year={2022},
    volume={15},
     number={4},
    pages={705-717},
    doi={10.1109/TOH.2022.3212701}
}
```

Authors
-------
[Ben Richardson](https://github.com/benrichardson28),
Haptic Intelligence - Max Planck Institute for Intelligent Systems


License
-------
CC-BY-NC 4.0 (see LICENSE.md).
