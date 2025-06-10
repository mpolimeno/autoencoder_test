#!/bin/bash

#SBATCH --job-name=JuliaML_Kestrel
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=4096MB
#SBATCH --output=julia_ml_kestrel_%j.out
#SBATCH --error=julia_ml_kestrel_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com # REPLACE WITH YOUR NREL EMAIL

# Load necessary modules:
echo "Loading modules..."
module purge # Unload any existing modules to avoid conflicts
module load julia    # Example: Replace with the actual Julia version on Kestrel
module load cuda     # Example: Replace with the actual CUDA toolkit version on Kestrel

# Print loaded modules
echo "Currently loaded modules:"
module list

# Navigate to project directory.
echo "Changing to working directory..."
cd /path/to/your/julia/ml/project

# Set Julia environment variables if necessary.
# export JULIA_NUM_THREADS=1

# Display job information
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Assigned nodes: $SLURM_NNODES"
echo "Assigned GPUs: $SLURM_GPUS_PER_NODE"
echo "Current directory: $(pwd)"
echo "Julia version: $(julia --version)"

# Run your Julia script
# The --project=. argument tells Julia to use the Project.toml/Manifest.toml in the current directory.
echo "Starting Julia ML application..."
julia --project=. nn_for_HOS_CUDA.jl

# Check the exit status of the Julia script
if [ $? -eq 0 ]; then
    echo "Julia ML application completed successfully."
else
    echo "Julia ML application failed. Check the .err file for details."
fi

echo "Job finished at $(date)"
