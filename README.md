# pinn-exploration-model

## Physics-Guided Membrane Segmentation from Cryo-ET Data

This repository provides tools for:
- Generating **pseudo cryo-ET data** from **Mem3DG-generated membrane meshes**.
- Implementing **physics-informed neural networks (PINNs)** for membrane segmentation.

## 📂 Repository Structure
```
membrane_segmentation/
│── data/                          # Data directory
│   ├── raw/                        # Raw Mem3DG input files
│   ├── synthetic/                  # Generated pseudo cryo-ET data
│   ├── processed/                  # Preprocessed data for training
│
│── src/                           # Source code
│   ├── data_generation/            # Code for synthetic data generation
│   │   ├── mesh_to_cryoet.py       # Converts Mem3DG meshes into pseudo cryo-ET
│   ├── pinn/                       # PINN model for membrane segmentation
│
│── notebooks/                     # Jupyter notebooks for experiments
│   ├── data_generation_demo.ipynb  # Walkthrough of synthetic data generation
│   ├── pinn_demo.ipynb             # Walkthrough of PINN-based segmentation
│
│── outputs/                       # General outputs from executions
│   ├── figs/                       # Figures/plots from experiments
│   ├── logs/                       # Logs and metrics
│   ├── models/                     # Saved models/checkpoints
│   ├── predictions/                # Final predictions/results
│
│── requirements.txt                # Dependencies for pip installation
│── README.md                       # Project documentation
│── LICENSE                         # MIT License file
│── .gitignore                      # Files ignored by Git
```

## 🚀 Installation

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-username/membrane_segmentation.git
cd membrane_segmentation
```

### 2️⃣ Set Up the Virtual Environment
```sh
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

## Alternative Installation using pixi

[Pixi](https://pixi.sh/) is a package management tool that helps to harmonize conda and pypi dependencies.
It also provides benefits such as preconfigured tasks and pipelining.

Instructions for installing pixi can be found here: https://pixi.sh/latest/installation/

Once pixi is installed you can simply run `pixi run biconcave`.
It should automatically set up a local conda environment and execute the biconcave "task". 
The tasks, dependency specificiations, and other configurations can be found in `pixi.toml`. 
Notably, changes to this file can be done by hand or by the pixi cli tool.
The `pixi.lock` file is a human readable list of solved dependency versions.
Changes to the lockfile are handled by pixi and reflect changes to the environment.

## 📜 License
This project is licensed under the **MIT License**.

## 📬 Contact
If you have questions, feel free to reach out via:
- **Email:** atsumat@uw.edu
