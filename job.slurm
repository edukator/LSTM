#!/bin/bash

#SBATCH --account=uerdogan
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=00:15:00
#SBATCH --job-name=test

#SBATCH --partition=orfoz
#SBATCH --ntasks=112
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1


###SBATCH --mal-user=utkuerdogan@eskisehir.edu.tr
###SBATCH --mail-type=BEGIN,END,FAIL
###SBATCH --mail-type=ALL

### Load modules

module load apps/truba-ai/cpu-2024.0

conda activate cpu-2024.0

### jobs
python torch_test.py

exit