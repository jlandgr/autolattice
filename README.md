# AutoLattice

AutoLattice is an efficient tool to automatically discover periodic lattice models with desired transport properties. It optimizes the discrete and continuous properties of the lattice's unit cell and provides an exhaustive list of all possible lattice models that fulfill the desired target characteristics.

This repository allows you to design your own lattice models with your desired transport behaviour and to reproduce the examples discussed in our publication: \
[Artificial discovery of lattice models for wave transport](https://arxiv.org/abs/2508.10693) \
*Jonas Landgraf, Clara Wanjura, Vittorio Peano, and Florian Marquardt* (2025) 

AutoLattice builds on our previous work [AutoScatter](https://github.com/jlandgr/autoscatter), which automatically designs few-mode coupled-mode systems with desired scattering properties. Please check out the corresponding [GitHub repository](https://github.com/jlandgr/autoscatter) and [PRX article](https://link.aps.org/doi/10.1103/PhysRevX.15.021038).

## Overview over this repository:
run_optimization.py: This script discovers all lattice models fulfilling the target transport characteristics specified in a setup file. \
all_setups/*.py: Setup files for the target transport properties discussed in our article. This includes  directional amplifiers with optimized gain and bandwidth, isolators with enhanced bandwidth, and frequency demultiplexers that selectively amplify signals within different frequency ranges in different directions. \
plot_*.ipynb: Analyze the discovered lattice structures \
dataset_generation_amplifier.py: This script generates a large dataset of lattice parameters while keeping the lattice structure constant. This dataset is used to extract functional dependencies between the coupling parameters and transport properties by using symbolic regression. \
dataset_merge.ipynb: Merge the individual files created by dataset_generation_amplifier.py into a single file. Furthermore, this notebook provides some plotting routines to analyze the properties of the lattices.\

Note that the scripts run_optimization.py and dataset_generation_amplifier.py are supposed to run on a cluster with Slurm as a workload manager. If you want to test these scripts locally on your computer, you have to set certain SLURM environment variables, see the respective scripts for more details.

## Installation
We recommend creating a new environment with venv or conda. The package can be installed using:
```
pip install git+https://github.com/jlandgr/autolattice.git
```

Please be aware that the code runs only on the CPU, as Jax does not support computing non-Hermitian eigenspectra on a GPU.

## Cite us

Are you using AutoLattice in your project or research, or do some related research? Then, please cite us!
```
@article{AutoLattice,
  title={Artificial discovery of lattice models for wave transport},
  author={Landgraf, Jonas and Wanjura, Clara C and Peano, Vittorio and Marquardt, Florian},
  journal={arXiv:2508.10693},
  year={2025},
  url={https://doi.org/10.48550/arXiv.2508.10693}
}
```
