#!/bin/sh
#SBATCH --job-name=training-bert
#SBATCH --gpus=a100-80:1
#SBATCH --output=training_job_output.txt
#SBATCH --error=training_job_error.txt
#SBATCH --time=3:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0725272@u.nus.edu

echo "job is running on $(hostname), started at $(date)"
echo "\n--- Installing packages ---\n"
pip install torch pandas transformers scikit-learn
echo "\n--- Finished installing packages, starting model training ---\n"

start_time=$(date +%s)
srun python training_scripts/bert_classification.py

end_time=$(date +%s)
exec_time=$((end_time - start_time))

echo -e "\nJob completed at $(date)\n"

echo -e "total execution time: $exec_time seconds\n"

