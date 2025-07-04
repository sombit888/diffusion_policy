# scratch directory  
export SCRATCH_DIR="/home/sombit_dey/projects/diffusion_policy/scratch"
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
else
    echo "Repository already exists. Pulling latest changes..."
    cd "diffusion_policy"
    git pull 
git checkout lerobot_dataset

# eval micromamba 
eval "$(micromamba shell hook --shell=bash)"
micromamba env create -n robot_diffusion -f environment.yml
micromamba activate robot_diffusion
