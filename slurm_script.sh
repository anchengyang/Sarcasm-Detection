#!/bin/sh
#SBATCH --job-name=training-bert
#SBATCH --gpus=1
#SBATCH --time=3:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0725272@u.nus.edu
srun python training_scripts/bert_classification.py