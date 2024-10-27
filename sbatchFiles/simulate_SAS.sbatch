#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<ansoan@dtu.dk>  # The default value is the submitting user.
#SBATCH --partition=xeon24
#SBATCH -N 1      # Minimum of 2 nodes
#SBATCH -n 1     # with n 112: 56 MPI processes per node, 112 tasks in total, appropriate for xeon56 nodes
#SBATCH -c 24     # cpus-per-task
#SBATCH --time=2-02:00:00 # 1 day, 2 hours, 0 min, 0 sec.
#SBATCH --begin=now+0 # 0 is in minutes
#SBATCH --mem 100Gb # Memory per node
#SBATCH --output=ErrorLogFiles/%j_simulate_SAS_n100000_0.0percentStructureFactor_float16.log
#SBATCH --error=ErrorLogFiles/%j_simulate_SAS_n100000_0.0percentStructureFactor_float16.err

# Write the content of this script into the log file
cat $0 >> "ErrorLogSbatchFiles/${SLURM_JOB_ID}_SbatchFile.log"

module load h5py/3.9.0-foss-2023a
module load SciPy-bundle/2023.07-iimkl-2023a
module use /home/modules/energy/modules/all
module load sasmodels/1.0.7-foss-2023a

srun python3 -u PythonFiles/simulate_SAS.py -n 1000 -d SAXS_datasets/SAXS_dataset_Keith_medium --dtype float16 -s 0.0
