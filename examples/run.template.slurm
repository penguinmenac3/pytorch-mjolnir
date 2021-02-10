#!/bin/bash
#SBATCH --job-name={name}
#SBATCH -p {partition}
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH -o {remote_results_path}/{name}/{version}/out.txt
#SBATCH -e {remote_results_path}/{name}/{version}/out.txt

chmod +x {remote_results_path}/{name}/{version}/src/{repository}/run.sh
srun {remote_results_path}/{name}/{version}/src/{repository}/run.sh
cat {remote_results_path}/{name}/{version}/out.txt | mail -s "[SLURM] {repository}: {name}" mail@example.com
