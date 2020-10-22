# Design space exploration of stagnation temperature probes via dimension reduction

Datasets and example code used in the 2020 ASME Turbo Expo [paper](https://www.researchgate.net/publication/344362850_Design_Space_Exploration_of_Stagnation_Temperature_Probes_via_Dimension_Reduction):

"Design space exploration of stagnation temperature probes via dimension reduction" (GT2020-16277)

## Data

The design of experiment used in the paper is shared in `.npy` files located in the folder `Data/`. The data consists of 128 designs, parameterised with 7 design parameters. All input and output data has been standardised to lie within the interval [-1,1]. 

## Companion code

In the paper, we use the `Subspace()` module from the [Effective Quadratures](https://www.effective-quadratures.org/docs/_documentation/subspaces.html) python library to obtain dimension reducing subspaces for three design objectives. Here we share example code to do this.

At present, two example jupyter notebooks are available in `/Example_Notebooks`:

1) `obtaining_subspaces.ipynb` reads in the example DoE data and obtains subspaces for two of the design objectives.
2) `grid_search.ipynb` performs a grid search to find suitable values for the two `Subspace()` hyperparameters. 

See this [blog post](https://discourse.effective-quadratures.org/t/exploring-the-design-of-a-temperature-probe-with-dimension-reduction/80/3) for examples of how these dimension reducing subspaces can be used for design space exploration. In the near future, notebooks exploring some of these more advanced applications will be shared here. 

Please feel free to leave questions/comments on the blog post!
