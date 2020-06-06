# Autotune-v2

[![Autotune-ICL](https://circleci.com/gh/cristianmatache/autotune-v2.svg?style=svg)](https://circleci.com/gh/cristianmatache/autotune-v2)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

## Abstract
Performance of machine learning models relies heavily on finding a good combination of hyperparameters. We aim to design the most efficient hybrid between two best-in-class hyperparameter
optimizers, Hyperband and TPE. On the way there, we identified and solved a few problems:

1. Typical metrics for comparing optimizers are neither quantitative nor informative about how
well an optimizer generalizes over multiple datasets or models.
2. Running an optimizer several times to collect performance statistics is time consuming/impractical.
3. Optimizers can be flawed: implementation-wise (eg. virtually all Hyperband implementations) or design-wise (eg. first published Hyperband-TPE hybrid).
4. Optimizer testing has been impractical because testing on true ML models is time-consuming.

To overcome these challenges, we propose: *Gamma loss function simulation*, *Closest known loss
function approximation* and a more comprehensive set of metrics. All three are benchmarked
against published results on true ML models. The simulation and the approximation complement
each other: the first makes testing practical for the first time and serves as a theoretical ML model
while the latter allows for timely collection of statistics about optimizer performance as if it was
run on a true ML model.

Finally, we use these to find the best hybrid architecture which is validated by comparison with
Hyperband and TPE on 2 datasets and several mathematical hard-to-optimize functions. The
pursuit for a hybrid is legitimate since it outperforms Hyperband and TPE alone, but there is still
some distance to the state-of-the-art performances.


**Keywords:** hyperparameter optimizer design, Hyperband-TPE hybrid, instant optimizer testing,
instant loss function simulation, loss function approximation, optimizer evaluation, deep learning


## Associated works
- *Efficient design of Machine Learning hyperparameter optimizers.* Cristian Matache, Dr. Jonathan Passerat-Palmbach, Dr. Bernhard Kainz. Imperial College London 2019 
https://www.imperial.ac.uk/media/imperial-college/faculty-of-engineering/computing/public/1819-ug-projects/MatacheC-Efficient-Design-of-Machine-Learning-Hyperparameter-Optimizers.pdf
- *Gamma Loss Functions: Enabling Instant Testing, Debugging and Monitoring of AutoML Methods* Cristian Matache, Dr. Jonathan Passerat-Palmbach, Dr. Juste Raimbault, Dr. Romain Reuillon, Dr. Daniel Rueckert
TODO URL

## Preview
#### 1. Better optimizer metrics: EPDF-OFEs
Example loss functions | Example optimal final error (OFE)
-----------------------|----------------------------------
<img src="https://github.com/cristianmatache/autotune-v2/blob/master/static/loss-function-profiles.png" width="300"> | <img src="https://github.com/cristianmatache/autotune-v2/blob/master/static/best-result-metric.png" width="300">
Several profiles e.g. mean, std, order |  Lowest final error of all loss functions

We are therefore characterizing an optimizer by its estimated probability density function of optimal final errors (EPDF-OFE).
This would give us more meaningful quantitative measures like statistical significance.
Example:

Histogram of OFEs occurrences |  Estimated PDFs
------------------------------|----------------
<img src="https://github.com/cristianmatache/autotune-v2/blob/master/static/pdf-ofe-hb-tpe-2tpe-hist.png" width="300">|<img src="https://github.com/cristianmatache/autotune-v2/blob/master/static/pdf-ofe-hb-tpe-2tpe-pdf-ofe.png" width="300">

*Problem:* One needs to run the optimizer several times to find its EPDF-OFE which is very time consuming (order of years) if done on the pure underlying ML model since 
it requires retraining. *Solved later* 

#### 2. Need for testing: optimizer implementations can be flawed
Testing optimizers is usually done on some known hard-to-optimize function like Rastrigin. 
Testing on real ML models is much more difficult due to prolonged times of retraining the models several times for each optimization. 
Hence, for hyperparameter optimizers that employ early stopping there is virtually no way of testing comprehensively. 
This is problematic since popular optimizers have flawed implementations.

**Example flaw - Hyperband:**
We found that several Hyperband implementations suffer from a floating point arithmetic bug. This minor bug has impactful consequences:
- Less exploration (up to an order of magnitude)
- Wasting time and computing resources heavily


#### 3. Testing solved: Gamma loss function simulations
There is a clear need for more comprehensive testing of optimizers, especially for those that employ early stopping. 
We propose a method based on Gamma processes to simulate the loss functions in negligible time in order to test optimizers in several cases.
For illustrative purposes, we reproduced the loss function "landscape" of running logistic regresstion on MNIST dataset.
Real MNIST loss functions | Simulated loss functions 
--------------------------|--------------------------
<img src="https://github.com/cristianmatache/autotune-v2/blob/master/static/mnist-real-profiles.png" width="300">|<img src="https://github.com/cristianmatache/autotune-v2/blob/master/static/mnist-sim-profiles.png" width="300">

The simulation above is using a Gamma process whose distribution at time ```t``` is shown below and the Rastrigin function (however, the simulation is general enough to support any hard-to-optimize function as base e.g. Branin, Drop-wave, Egg-holder).

Gamma distribution at step `t` | Rastrigin function
-------------------------------|-------------------
<img src="https://github.com/cristianmatache/autotune-v2/blob/master/static/gamma.png" width="300">|<img src="https://github.com/cristianmatache/autotune-v2/blob/master/static/rastrigin-surface.png" width="300">

#### 3. Approximation: Closest known loss function (in terms of MSE)
The ﬁrst step of Closest-known-function approximation is collecting data (complete loss functions) for a given machine learning model on a given dataset. By complete loss function we mean that they have a representative length (convergence is clearly observable) and that they have not been "trimmed" by any early stopping method. Every loss function corresponds to a combination of hyperparameters so we store this mapping in a database.

Next, when an optimizer "proposes" a combination of hyperparameters (called an "arm") for its next iteration, instead of retraining the ML model with the proposed arm (to find the corresponding loss function), we look up the database and return the loss function corresponding to the arm that is closest in terms of mean square error (MSE) of the normalized hyperparameters. We approximate the behaviour of the real ML model in negligible time and use it to generate EPDF-OFEs. 

Database sharing brings a comparable metric to optimizer designers. The more loss functions are collected the better the results of the approximation. Because we are interested in ﬁnding the best loss function, it is important to have some density of functions around the optimal one. Similarly, to Bayesian optimizers it assumes some smoothness of the true objective function.

#### 4. Hyperband-TPE hybrids
TODO


## Appendix
#### 1. Optimizers heat map
<img src="https://github.com/cristianmatache/autotune-v2/blob/master/static/heatmap.PNG" width="400">

#### 2. Preliminaries
Gaussian processes| Hyperband
------------------|----------
<img src="https://github.com/cristianmatache/autotune-v2/blob/master/static/gaussian-processes.png" width="300">|<img src="https://github.com/cristianmatache/autotune-v2/blob/master/static/hyperband-table.PNG" width="300">

#### 3. Some of the flawed Hyperband implementations:
1. https://homes.cs.washington.edu/~jamieson/hyperband.html used by Hyperband authors (Li et al., 2016)
2. https://github.com/automl/HpBandSter/blob/367b6c4203a63ff8b395740995b22dab512dcfef/hpbandster/optimizers/hyperband.py#L60 used by BOHB (Falkner et al., 2018). 
3. https://github.com/zygmuntz/hyperband/blob/master/hyperband.py#L18 
4. https://gist.github.com/PetrochukM/2c5fae9daf0529ed589018c6353c9f7b#ﬁle-hyperband-py-L204 
5. https://github.com/electricbrainio/hypermax/blob/master/hypermax/algorithms/adaptive_bayesian_hyperband_optimizer.py#L26
6. https://github.com/polyaxon/polyaxon/blob/ee3fe8a191d96fc8ba3c1affd13f7ed5e7b471c7/core/polyaxon/polytune/search_managers/hyperband/manager.py#L75
7. https://github.com/thuijskens/scikit-hyperband/blob/master/hyperband/search.py#L346
8. At the time of writing, before June 2019, ray (https://ray.readthedocs.io) also had the same issue but was since resolved.

In 2020, we discovered that Microsoft's nni has independently fixed the same issue around the same time with us in 2019:
https://github.com/microsoft/nni/commit/c6b7cc8931042f318693d5ddcd1cc430d7734144 but no way of testing has been provided.
