#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<ansoan@dtu.dk>  # The default value is the submitting user.
#SBATCH --partition=sm3090_devel #xeon24el8
#SBATCH --nodes=1 # nodes
#SBATCH --ntasks=1 # 10 per node
#SBATCH --ntasks-per-node=1 # always 10
#SBATCH --gpus-per-node=1 # always 10
#SBATCH --cpus-per-task=8 # always 8
#SBATCH --time=0-00:12:00 # 1 day, 2 hours, 0 min, 0 sec.
#SBATCH --begin=now+0hour # 0 is in seconds
#SBATCH --output=../ErrorLogSbatchFiles/%j_classification.log
#SBATCH --error=../ErrorLogSbatchFiles/%j_classification.err

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


# Run the classification script with the config file
srun python3 -u PythonFiles/classification.py --config configs/classification_config.yaml
