
# Set repo root
export PYTHONPATH=~/Desktop/reclone-2/param-tuning/autotune:$PYTHONPATH

# Create conda environment
conda create --name autotune --file requirements.txt

# Activate conda environment
conda activate autotune

# Run script
python experiments/run_experiment.py 

# Conda deactivate (do not used deactivate as per docs.conda.io)
conda activate
