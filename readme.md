# Structural Privacy in GNNs

This repository contains the implementation and experimental setup for the paper "[GraphPrivatizer: Improved Structural Differential Privacy for Graph Neural Networks](https://openreview.net/pdf?id=lcPtUhoGYc)", accepted at TMLR.

## Requirements

This code is implemented in Python 3.9. Refer to `environment.yml` for the complete list of requirements.

## Usage

### Replicating the paper's results
In order to replicate our experiments and reproduce the paper's results, you can set up the desired datasets and parameters in `experiments.py` and then run:  
1. Run ``python experiments.py -n test_job create --attack --device cpu --repeats 10``
2. Run ``python experiments.py -n test_job exec --all``
   All the datasets will be downloaded automatically into ``/datasets`` folder, and the results will be stored in ``/results`` directory.