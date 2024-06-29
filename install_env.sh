#!/bin/bash
# 
# Installer for pocsdeblend
# 
# Run: ./install_env.sh
# 
# M. Ravasi, 24/05/2024

echo 'Creating pocsdeblend environment'

# create conda env
conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pocsdeblend
conda env list
echo 'Created and activated environment:' $(which python)

# check cupy work as expected
echo 'Checking cupy version and running a command...'
python -c 'import cupy as cp; print(cp.__version__); cp.ones(10000)*10'

echo 'Done!'

