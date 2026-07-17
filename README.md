# PINN: Physics-Informed Neural Networks for Membrane Reconstruction and Curvature Estimation

PINN is a physics-informed framework for reconstructing smooth membrane geometries from volumetric cryo-electron tomography (cryo-ET) data. Instead of representing membranes as discrete voxels or meshes, the framework learns a continuous **implicit neural representation (INR)** of the membrane surface, enabling direct computation of differential geometric quantities such as surface normals, mean curvature, and Gaussian curvature.

To improve geometric accuracy under noisy and incomplete imaging conditions, membrane mechanics are incorporated into the optimization through a **physics-informed neural network (PINN)** framework. The resulting membrane reconstruction is consistent with both the observed image data and the underlying membrane physics, providing an accurate representation for quantitative membrane analysis.

<p align="center">
  <img src="asset/overview_pinn.png" width="900">
</p>

---

## Features

* Continuous implicit neural representation (INR) of membrane geometry
* Physics-informed optimization using membrane mechanics
* Direct estimation of surface normals, mean curvature, and Gaussian curvature
* JAX/Flax implementation with automatic differentiation
* Example dataset and complete analysis workflow

---

## Repository Structure

```text
.
├── asset/                  Figures used in the documentation
├── example/
│   ├── data/               Example input data
│   └── output/             Example reconstruction results
├── notebooks/
│   ├── 1-signal-extraction.ipynb
│   ├── 2-phase-field-optimization.ipynb
│   └── 3-curvature-analysis.ipynb
├── src/
│   └── pinn/
│       ├── analysis.py
│       ├── model.py
│       ├── plot.py
│       ├── preprocess.py
│       ├── run.py
│       ├── train.py
│       └── utils.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

Create a Python environment

```bash
conda create -n pinn python=3.12
conda activate pinn
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## Quick Start

The example workflow is organized into three Jupyter notebooks and should be executed in the following order.

### 1. Signal extraction

```text
notebooks/1-signal-extraction.ipynb
```

This notebook extracts membrane signal points from the input volumetric image and prepares the training data.

### 2. Physics-informed phase-field optimization

```text
notebooks/2-phase-field-optimization.ipynb
```

This notebook reconstructs the membrane as a continuous implicit phase field using the physics-informed neural network.

### 3. Curvature analysis

```text
notebooks/3-curvature-analysis.ipynb
```

This notebook extracts the membrane surface and computes geometric quantities including

* Surface normal
* Mean curvature
* Gaussian curvature

---

## Example Data

Example data and reconstruction results are provided under

```text
example/
```

to demonstrate the complete workflow without requiring external datasets.

---

## Method

The reconstruction pipeline consists of three main steps:

```text
Volumetric image
        │
        ▼
Signal extraction
        │
        ▼
Physics-informed membrane reconstruction
        │
        ▼
Continuous phase field
        │
        ▼
Surface extraction
        │
        ▼
Curvature estimation
```

The membrane geometry is represented as a continuous implicit neural representation, allowing geometric quantities to be computed directly through automatic differentiation. A physics-based regularization derived from membrane mechanics is incorporated during optimization to produce geometrically smooth and physically plausible reconstructions.

---

## Dependencies

The implementation is based on

* JAX
* Flax
* Optax
* NumPy
* SciPy
* Matplotlib

See `requirements.txt` for the complete list of dependencies.

---

## Citation

If you use this repository in your research, please cite

```bibtex
@article{YOUR_PAPER,
  title   = {Physics-Informed Neural Networks for Membrane Reconstruction and Curvature Estimation},
  author  = {...},
  journal = {...},
  year    = {...}
}
```

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.



