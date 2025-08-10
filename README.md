# Introduction
Autoencoder Neural Ordinary Differential Equations (NODE) with stiffness reduction for surrogate modeling of stiff and irregular time series data.


This repository is an open-source implementation of NeuralODE for irregularly sampled time series data with stiffness reduction techniques. Surrogate modeling of time series data that exhibits stiffness characteristics (for eg. multiple, order-of-magnitude different time scales) is an open and challenging problem. An example domain where this problem is very prevalent is combustion modeling, where reactions are governed by stiff Ordinary Differential equations. The cost of solving these stiff ODEs at every node in a large simulation is prohibitive. 

Recent works have leveraged machine learning to create models that reduce the cost of solving these differential equations. 

# Methodology


Features:

Encoder to change physical space to a "stiffness reduced" and dimension reduced latent space
    - Includes a latent space stiffness reduction term. can be turned off by setting the weight (stiffness reduction weight) to 0
    - 
Neural ODE to step through time, backpropagating gradients in this latent space
    - diffrax
    - euler stepping
    - handles failure
    -
decoder to change back to physical space
Scaling to alleviate training issues
    - automated input normalization
    - Automated scaling of NODE inputs and outputs
GPU optimized implementation
    - Minimal H2D transfers

supports any dataset; see config file
controlled from config file

TODO:
Further GPU optimization
    - Setup prefetch
L-BFGS training

Tips:

1. Train encoder decoder for a while; unstable enc dec causes NODE failure quite quickly
2. You have the option of selecting which rows/columns to include in the training set. Make sure all features 
   that are causative are included in the training data. For example, if modeling a chemical reaction, by not including the temperature, the network cannot learn the appropriate cause effect relationships from data and this will result
   in a poor model
3. The init_dt is kept constant (for now, this package uses only explicit euler training). Make sure this init_dt is     small enough to capture the smallest timescale of your system. Else it will result in poor models

4. try and make key hyper-parameters a power of 2. Key ones include network widths, samples per batch. This allows for more optimum GPU training.
5. Ensure the batch size is small enough such that it plus computations can fit on device


This repo leverages techniques and ideas from the following works:

1. ChemNODE: A neural ordinary differential equations framework for efficient chemical kinetic solvers (link:  https://www.sciencedirect.com/science/article/pii/S2666546821000677)

2. A data-driven reduced-order model for stiff chemical kinetics using dynamics-informed training (link: https://www.sciencedirect.com/science/article/pii/S2666546823000976)

3. Stiffness-Reduced Neural ODE Models for Data-Driven Reduced-Order Modeling of Combustion Chemical Kinetics (link: https://arc.aiaa.org/doi/abs/10.2514/6.2022-0226)

4. A Physics-Constrained Autoencoder-NeuralODE Framework for Learning Complex Hydrocarbon Fuel Chemistry: Methane Combustion Kinetics (link: https://www.frontiersin.org/journals/thermal-engineering/articles/10.3389/fther.2025.1594443/abstract)

5. Stiff Neural Ordinary Differential Equations (Link: https://arxiv.org/abs/2103.15341)

6. Neural Ordinary Differential Equations (Link: https://arxiv.org/abs/1806.07366)

# Installation

# Credits

# Directory Structure

# Input data structure

# Tutorial

