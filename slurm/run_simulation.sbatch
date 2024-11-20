#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<ansoan@dtu.dk>  # The default value is the submitting user.
#SBATCH --partition=epyc96 #xeon24el8
#SBATCH -N 1      # Minimum of 2 nodes
#SBATCH -n 1     # with n 112: 56 MPI processes per node, 112 tasks in total, appropriate for xeon56 nodes
#SBATCH -c 96     # cpus-per-task
#SBATCH --time=2-02:00:00 # 1 day, 2 hours, 0 min, 0 sec.
#SBATCH --begin=now+0 # 0 is in minutes
#SBATCH --mem 100Gb # Memory per node
#SBATCH --output=../ErrorLogSbatchFiles/%j_simulate_SAS.log
#SBATCH --error=../ErrorLogSbatchFiles/%j_simulate_SAS.err

# Write the content of this script into the log file
cat $0 >> "../ErrorLogSbatchFiles/${SLURM_JOB_ID}_SBATCH.log"

# Define the root directory
ROOT_DIR="../"

# Navigate to the project root
cd $ROOT_DIR

# Add the project root and the src directory to PYTHONPATH
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:$PYTHONPATH"

# Install the package in editable mode (if needed)
module load Python/3.11.3-GCCcore-12.3.0
pip3 install --upgrade pip
pip3 install --user -e .

# Run the simulation script with the config file
srun python3 -u PythonFiles/simulate_sas.py --config configs/simulation_config_large.yaml