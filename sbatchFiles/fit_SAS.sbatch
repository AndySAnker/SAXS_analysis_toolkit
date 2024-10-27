#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<ansoan@dtu.dk>  # The default value is the submitting user.
#SBATCH --partition=xeon56
#SBATCH -N 1      # Minimum of 2 nodes
#SBATCH -n 1     # with n 112: 56 MPI processes per node, 112 tasks in total, appropriate for xeon56 nodes
#SBATCH -c 56     # cpus-per-task
#SBATCH --time=2-02:00:00 # 1 day, 2 hours, 0 min, 0 sec.
#SBATCH --begin=now+0 # 0 is in minutes
#SBATCH --mem 100Gb # Memory per node
#SBATCH --output=ErrorLogFiles/fit_SAS_refactored_N1_10times_fTrue_pTrue_wsqrt_noResolution.log
#SBATCH --error=ErrorLogFiles/fit_SAS_refactored_N1_10times_fTrue_pTrue_wsqrt_noResolution.err

module load h5py/3.9.0-foss-2023a
module load SciPy-bundle/2023.07-iimkl-2023a
module use /home/modules/energy/modules/all
module load sasmodels/1.0.7-foss-2023a
#pip3 install bumps
pip3 install XGBoost
module load matplotlib/3.7.2-iimkl-2023a

# Print the current date and time
echo "Script started on: $(date)"

for i in {1..1}
do
    echo  # This will print a new line
    echo "Running iteration: $i"
    srun python3 -u PythonFiles/fit_SAS_refactored.py -n 10 -f -p -w sqrt
done