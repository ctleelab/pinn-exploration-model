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
│   ├── pinn_segmentation/          # PINN model for membrane segmentation
│
│── notebooks/                     # Jupyter notebooks for experiments
│   ├── data_generation_demo.ipynb  # Walkthrough of synthetic data generation
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

## 📜 License
This project is licensed under the **MIT License**.

## 📬 Contact
If you have questions, feel free to reach out via:
- **Email:** atsumat@uw.edu

