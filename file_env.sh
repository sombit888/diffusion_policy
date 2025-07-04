#!/bin/bash

# scratch directory  
export SCRATCH_DIR="/scratch/sombit_dey/projects"
if [ ! -d "$SCRATCH_DIR" ]; then
    mkdir -p "$SCRATCH_DIR"
fi
cd "$SCRATCH_DIR"
# git hub repo link
# Add your repository URL below
REPO_URL="git@github.com:sombit888/diffusion_policy.git"
# Clone the repository if it doesn't exist
if [ ! -d "diffusion_policy" ]; then
    git clone "$REPO_URL"
    cd "diffusion_policy"
else
    echo "Repository already exists. Pulling latest changes..."
    cd "diffusion_policy"
    git pull
fi

git checkout lerobot_dataset

# eval micromamba 
eval "$(micromamba shell hook --shell=bash)"
micromamba env create -n robot_diffusion -f conda_environment.yaml
pip install backports.strenum