#!/bin/bash
#SBATCH -n 8 #Number of cores
#SBATCH --gres=gpu:1
#SBATCH -N 1 # All cores on one machine
#SBATCH -t 0-08:00 #Runtime in minutes
#SBATCH -p gpu_requeue #Partition to submit to
#SBATCH --mem-per-cpu=10000 #Memory per cpu in MB (see also --mem)
#SBATCH --mail-type=END #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=tdeutsch@college.harvard.edu #Email to which notifications will be sent

echo ${RUN_CONFIG}
module load Anaconda3/5.0.1-fasrc02
# module load cuda/10.0.130-fasrc01
module load cuda/10.0.130-fasrc01 cudnn/7.4.1.5_cuda10.0-fasrc01
source activate 2tf1.14_cuda10
python -W ignore runner.py -a ${SLURM_ARRAY_TASK_ID} ${RUN_CONFIG}