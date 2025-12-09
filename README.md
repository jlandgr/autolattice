# AutoLattice

AutoLattice is an efficient tool to automatically discover periodic lattice models with desired transport properties. It optimizes the discrete and continuous properties of the lattice's unit cell and provides an exhaustive list of all possible lattice models that fulfill the desired target characteristics.

This repository allows you to design your own lattice models with your desired transport behaviour and to reproduce the examples discussed in our publication: \
[Artificial discovery of lattice models for wave transport](https://arxiv.org/abs/2508.10693) \
*Jonas Landgraf, Clara Wanjura, Vittorio Peano, and Florian Marquardt* (2025) 

AutoLattice builds on our previous work [AutoScatter](https://github.com/jlandgr/autoscatter), which automatically designs few-mode coupled-mode systems with desired scattering properties. Please check out the corresponding [GitHub repository](https://github.com/jlandgr/autoscatter) and [PRX article](https://link.aps.org/doi/10.1103/PhysRevX.15.021038).

## Overview over this repository:
TODO

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
