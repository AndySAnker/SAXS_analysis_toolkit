#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<ansoan@dtu.dk>  # The default value is the submitting user.
#SBATCH --partition=sm3090el8
#SBATCH -N 1-1      # Minimum of 2 nodes
#SBATCH -n 1     # with n 112: 56 MPI processes per node, 112 tasks in total, appropriate for xeon56 nodes
#SBATCH -c 8     # cpus-per-task
#SBATCH --gres=gpu:1 # GPU's
#SBATCH --time=2-02:00:00 # 1 day, 2 hours, 0 min, 0 sec.
#SBATCH --begin=now+0hour # 0 is in seconds
#SBATCH --mem 100Gb # Memory per node
#SBATCH --output=ErrorLogFiles/%j_XGBoost_Parameters_0.0percentStructureFactor_noResolution_Keith_small.log
#SBATCH --error=ErrorLogFiles/%j_XGBoost_Parameters_0.0percentStructureFactor_noResolution_Keith_small.err

# Write the content of this script into the log file
cat $0 >> "ErrorLogSbatchFiles/${SLURM_JOB_ID}_SbatchFile.log"

module load h5py/3.9.0-foss-2023a
module load SciPy-bundle/2023.07-iimkl-2023a
module load scikit-learn/1.3.1-gfbf-2023a
pip3 install XGBoost
module load matplotlib/3.7.2-iimkl-2023a
#source XGBoost/bin/activate

#srun python3 -u PythonFiles/XGBoost.py
#srun python3 -u PythonFiles/XGBoost_NumDatapoints.py
#srun python3 -u PythonFiles/XGBoost_ensemble.py
#srun python3 -u PythonFiles/XGBoost_parameters.py
srun python3 -u PythonFiles/XGBoost_NumDatapoints_Keith.py

