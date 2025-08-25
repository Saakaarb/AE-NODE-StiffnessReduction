.. AE-NODE-StiffnessReduction documentation master file, created by
   sphinx-quickstart on Mon Aug 25 10:53:29 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AE-NODE-StiffnessReduction documentation
========================================

Introduction
============

This repository contains an open source implementation of Autoencoder Neural Ordinary Differential Equations (NODE) with dimension reduction and stiffness reduction for surrogate modeling of stiff and irregular time series data. This repository is a GPU-efficient open-source implementation of NODEs for irregularly sampled time series (with potentially different trajectory lengths) data with dimension reduction and stiffness reduction:

- Run through a easy-to-read config file
- Simple, extendable API
- Supports irregularly sampled trajectories with uneven trajectory sizes across datapoints
- Uses a dimension-reduction autoencoder that maps to a stiffness-reduced latent space for faster integration
- Includes a NeuralODE implementation in latent space 
- Implementations optimized for GPU for faster training via masking, vectorization and efficient JAX use.

Why AE-NODE-StiffnessReduction?
===============================

 Modeling of irregularly-sampled time series data that may exhibit stiffness characteristics (for eg. multiple, order-of-magnitude different time scales of feature evolution) is an open and challenging problem. For example, hydrocarbon combustion models are a system of very stiff ODEs. The cost of solving these stiff ODEs at every node (measuring in millions) in a large simulation is prohibitive. A surrogate model that can accurately make predictions in a stiffness reduced manifold is exceptionally useful to use in place of the full equation system while simulating, reducing the cost of integration by order of magnitude.

Recent works have leveraged machine learning to create models that reduce the cost of solving these differential equations. In particular, Neural ODEs have been used in a wide variety of scientific applications, combining data driven modeling and scientific computing techniques. However, they are autoregressive models, and come with training challenges, from both a stability and efficiency perspective:

1.  They can be difficult to train on long time horizons, or on datasets with features exhibiting stiff behavior, as their rollouts can become unstable and diverge.

2. They are also nontrivial to effectively parallelize on a GPU, given their autoregressive nature.

3. There are very few open-source implementations of NeuralODEs for time-series modeling of stiff, challenging problems.

This library is a ready-to-use implementation of stiffness-reduced Neural ODEs with a simple configuration file to run easy experiments.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    api 


