
## Multi-fidelity Hierarchical Neural Processes
## Paper: 
Dongxia Wu, Matteo Chinazzi, Alessandro Vespignani, Yi-An Ma, Rose Yu, [Multi-fidelity Hierarchical Neural Processes](https://arxiv.org/abs/2206.04872), 
KDD 2022

## Abstract:
Science and engineering fields use computer simulation extensively. These simulations are often run at multiple levels of sophistication to balance 
accuracy and efficiency. Multi-fidelity surrogate modeling reduces the computational cost by fusing different simulation outputs. Cheap data generated 
from low-fidelity simulators can be combined with limited high-quality data generated by an expensive high-fidelity simulator. Existing methods based 
on Gaussian processes rely on strong assumptions of the kernel functions and can hardly scale to high-dimensional settings. We propose Multi-fidelity 
Hierarchical Neural Processes (MF-HNP), a unified neural latent variable model for multi-fidelity surrogate modeling. MF-HNP inherits the flexibility 
and scalability of Neural Processes. The latent variables transform the correlations among different fidelity levels from observations to latent space. 
The predictions across fidelities are conditionally independent given the latent states. It helps alleviate the error propagation issue in existing 
methods. MF-HNP is flexible enough to handle non-nested high dimensional data at different fidelity levels with varying input and output dimensions. 
We evaluate MF-HNP on epidemiology and climate modeling tasks, achieving competitive performance in terms of accuracy and uncertainty estimation. 
In contrast to deep Gaussian Processes with only low-dimensional (< 10) tasks, our method shows great promise for speeding up high-dimensional 
complex simulations (over 7000 for epidemiology modeling and 45000 for climate modeling).

## Requirements
* torch
* scipy>=0.19.0
* numpy>=1.12.1
* pandas>=0.19.2
* pyyaml
* statsmodels
* tensorflow>=1.3.0
* torch
* tables
* future
* sklearn
* matplotlib
* gpytorch
* math

To install requirements:
```
pip install -r requirements.txt
```
## Neural Processes Model Training and Evaluation
```
cd sir_np/BA/*
python train.py
cd sir_np/MA/*
python train.py
cd climate_np/BA/*
python train.py
cd climate_np/MA/*
python train.py
```
## Gaussian process Model Training and Evaluation
```
cd sir_gp
run *.ipynb
cd climate_gp
run *.ipynb
```

## [Dataset](https://drive.google.com/drive/folders/1l5gqueulNXIrNc6yElx3WU8w-joxFiYj?usp=sharing)
For Neural Processes Model, download sir_np/*/data to dataset_dir: sir_np/data/, download climate_np/*/data to dataset_dir: climate_np/data/  
For Gaussian process Model, download sir_gp/nargp_data to dataset_dir: sir_gp/, download sir_gp/sfgp_data to dataset_dir: sir_gp/, 
download climate_gp/* to dataset_dir: climate_gp/


<!-- ## Cite
```
@article{wu2022multi,
  title={Multi-fidelity Hierarchical Neural Processes},
  author={Wu, Dongxia and Chinazzi, Matteo and Vespignani, Alessandro and Ma, Yi-An and Yu, Rose},
  journal={arXiv preprint arXiv:2206.04872},
  year={2022}
}
``` -->
