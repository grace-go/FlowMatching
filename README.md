# LieFlow
This repository contains code to perform Flow Matching (FM) on Lie Groups, as described in [[1]](#1). The results of [[1]](#1) can be reproduced with the notebooks in the `experiments` directory.

This work extends FM on Euclidean space by Lipman et al. [[2]](#2). 
The goal is somewhat similar to Riemannian FM by Chen & Lipman [[3]](#3), though they use geodesics of (pre)metrics. As a consequence, their method works on manifolds that aren't Lie groups, but require the design of a (pre)metric. Additionally, Riemannian FM requires integration of the learned vector field during training in all but the most simple cases, whereas FM on Lie Groups is simulation-free.

## Installation
The core functionality of this repository requires:
* `python>=3.12`
* `pytorch==2.6`
* `numpy`
* `tqdm`

To reproduce the experiments, one additionally needs:
* `jupyter`
* `matplotlib`

The figures were made using LaTeX. Hence, to run the notebooks one either needs to have a LaTeX installation, or to remove the `matplotlib` configuration code
```python
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": 10.0,
    })
```
from the top of the notebooks.

Alternatively, one can create the required conda environment from `env.yml`:
```
conda env create -f env.yml
```
This creates a conda environment called `flow-matching`.

Subsequently, one must install the code of this project as a package, by running:
```
pip install -e .
```

## Functionality


## Cite
If you use this code in your own work, please cite our paper:

<a id="1">[1]</a> Sherry, F.M., Smets, B. "Flow Matching on Lie Groups." 10th International Conference on Geometric Science of Information (GSI) (2025).
```
@inproceedings{Sherry2025LieFlow,
  author =       {Sherry, Finn M. and Smets, Bart M.N.},
  title =        {{Diffusion-Shock Filtering on the Space of Positions and Orientations}},
  booktitle =    {7th International Conference on Geometric Science of Information},
  publisher =    {Springer},
  year =         {2025},
  address =      {Saint-Malo, France},
  pages =        {},
  doi =          {},
  editor =       {Nielsen, Frank and Barbaresco, Frédéric}
}
```

We extend FM on Euclidean space by Lipman et al:

<a id="2">[2]</a> Lipman, Y., Havasi, M., Holderrieth, P., Shaul, N., Le, M., Karrer, B., Chen, R.T.Q., Lopez-Paz, D., Ben-Hamu, H., Gat, I. "Flow Matching Guide and Code." arXiv preprint (2024). https://arxiv.org/abs/2412.06264
```
@article{Lipman2024FlowCode,
  author = {Lipman, Yaron and Havasi, Marton and Holderrieth, Peter and Shaul, Neta and Le, Matt and Karrer, Brian and Chen, Ricky T. Q. and Lopez-Paz, David and Ben-Hamu, Heli and Gat, Itai},
  title = {{Flow Matching Guide and Code}},
  journal = {arXiv preprint},
  year = 2024,
  url = {https://arxiv.org/abs/2412.06264}
}
```

The goal of our work is similar to FM on Riemannian manifolds by Chen & Lipman:

<a id="3">[3]</a> Chen, R.T.Q., Lipman, Y. "Flow Matching on General Geometries." 12th International Conference on Learning Representations (ICLR) (2024). https://openreview.net/forum?id=g7ohDlTITL
```
@inproceedings{Chen2024FlowGeometries,
  author = {Chen, Ricky T. Q. and Lipman, Yaron},
  title = {{Flow Matching on General Geometries}},
  booktitle = {12th International Conference on Learning Representations},
  year = {2024},
  url = {https://openreview.net/forum?id=g7ohDlTITL}
}
```