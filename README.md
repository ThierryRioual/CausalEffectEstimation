# Causal Effect Estimation in pyAgrum

## About 

This repository documents the work of Thierry Rioual during a research internship at LIP6, under the supervision of Pierre-Henri Wuillemin, from July to August 2024.
The content explores the intersection of Causal Bayesian networks with the Rubin Causal Model framework.

## Notebooks

It includes notebooks detailing causal models implemented in  `pyAgrum`, derived from lab work for the second-year master's course "Causal Inference and Treatment Effect Estimation," taught by Marianne Clausel at the Institut Polytechnique de Paris.

After examining various causal paradigms, additional features for integration into `pyAgrum` were identified, leading to the creation of the `CausalEffectEstimation` class, which combines the strengths of both frameworks.

## Causal Effect Estimation

This class leverages `pyAgrum`'s graphical probabilistic models to conduct causal identification using causal Bayesian networks, facilitating the determination of the available adjustment set. Once identified, statistical estimators of causal effect are employed to perform causal inference and compute the Average Causal Effect (ACE), drawing from recent advancements in statistics.

The pipeline of the `CausalEffectEstimation` class streamlines the processes of causal identification and estimation into a single module, enabling the study of causal relationships with minimal code.

## Contact Information

Email: rioualthierry121104@gmail.com
