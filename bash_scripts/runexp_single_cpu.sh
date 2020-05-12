#!/bin/bash
#SBATCH -n 1 #Number of cores
#SBATCH -N 1 # All cores on one machine
#SBATCH -t 40 #Runtime in minutes
#SBATCH -p shared  #Partition to submit to
#SBATCH --mem-per-cpu=10000 #Memory per cpu in MB (see also --mem)
#SBATCH --mail-type=END #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=tdeutsch@college.harvard.edu #Email to which notifications will be sent

echo ${RUN_CONFIG}
module load Anaconda3/5.0.1-fasrc02
module load cuda/10.0.130-fasrc01
source activate 2tf1.14_cuda10
python -W ignore runner.py -a ${SLURM_ARRAY_TASK_ID} ${RUN_CONFIG}