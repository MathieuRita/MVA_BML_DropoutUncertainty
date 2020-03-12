# Project for the course Bayesian Machine Learning (MVA)

Here is our Pytorch implementation for our project based on the paper: __Yarin Gal, Zoubin Ghahramani., 2015. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. arXiv https://arxiv.org/abs/1506.02142__. We perform several experiments on simple cases and datasets to comment the results provided by the paper

## Description of the code

The repository is mainly composed of 3 files:

- `bml_net.py` : class to build the neural networks

- `utils.py` : provides the useful function to construct Pytorch Dataset object and loader and preprocessing functions

- `experiments.ipynb` : progressive notebook to present the diverse experiments

## Experiments

All the experiments are gathered in the notebook  `experiments.ipynb` . It provides a code that has been commented step by step . The notebook is organized in the parts:

- __Part 1__: Run an experiment (tutorial). This first part shows step by step how we ran our experiments. If you want to reproduce a single experiment, you just have to run this part

- __Part 2__ : Behaviour of the Dropout network on regression task (*code for the section 4.3 of the report*) : analyze of the predictive performance via RMSE and predictive LL evaluation. Comparison with state-of-the-art methods.

![alt text]("figures/bml_figure_3.png")

- __Part 3__ : Behaviour of the Dropout network on classification task (*code for the section 4.2 of the report*): analyze of the uncertainty of the classification on MNIST dataset.

![alt text]("figures/bml_figure.png")

