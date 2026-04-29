#!/bin/bash

#SBATCH --job-name=run_pinn
##SBATCH --mail-type=all
##SBATCH --mail-user=atsumat@uw.edu

#SBATCH --account=biology
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00

#SBATCH --chdir=/gscratch/matsulab/sim/pinn-exploration-model/notebooks
#SBATCH --export=all
#SBATCH --output=../job_scheduler/log_%x_%j.out
#SBATCH --error=../job_scheduler/log_%x_%j.err

# Modules to use (optional).
module purge

# Your programs to run.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pinn-env
export PYTHONPATH="${PYTHONPATH}:../src"

python routine_curv.py
# python routine_cme.py
# python routine_sign.py
# python routine_exp.py