# Introduction
Autoencoder Neural Ordinary Differential Equations (NODE) with stiffness reduction for surrogate modeling of stiff and irregular time series data.


This repository is an open-source implementation of NeuralODE for irregularly sampled time series data with stiffness reduction techniques. Surrogate modeling of time series data that exhibits stiffness characteristics (for eg. multiple, order-of-magnitude different time scales) is an open and challenging problem. An example domain where this problem is very prevalent is combustion modeling, where reactions are governed by stiff Ordinary Differential equations. The cost of solving these stiff ODEs at every node in a large simulation is prohibitive. 

Recent works have leveraged machine learning to create models that reduce the cost of solving these differential equations. 

# Methodology



Features:

Encoder to change physical space to a "stiffness reduced" latent space
Neural ODE to step through time, backpropagating gradients in this latent space
decoder to change back to physical space
Scaling to alleviate training issues

Upcoming features:

GPU-optimized JAX implementation


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

