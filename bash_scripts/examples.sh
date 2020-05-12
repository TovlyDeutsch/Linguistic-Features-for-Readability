sbatch --array=1-2 bash_scripts/runexp_multi_cpu.sh

IN=input.data OUT=output.res <bash_scripts/runexp_multi_cpu.sh sbatch 
#SBATCH -o %OUT_%A_%a.out # Standard output


sbatch --array=1-2 -o test_%A_%a.out -e bash_scripts/runexp_multi_cpu.sh