#!/bin/sh
#SBATCH --job-name=training-transformer
#SBATCH --gpus=a100-80:1
#SBATCH --output=results/${1:-bert}_training_job_output.txt
#SBATCH --error=results/${1:-bert}_training_job_error.txt
#SBATCH --time=3:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0726078@u.nus.edu

# Default model is BERT if not specified
MODEL=${1:-bert}

echo "job is running on $(hostname), started at $(date)"
echo "Training model: $MODEL"
echo "\n--- Installing packages ---\n"
pip install torch pandas transformers scikit-learn
echo "\n--- Finished installing packages, starting model training ---\n"

start_time=$(date +%s)

if [ "$MODEL" = "roberta" ]; then
    echo "Running RoBERTa model training..."
    srun python training_scripts/roberta_classification.py
elif [ "$MODEL" = "albert" ]; then
    echo "Running ALBERT model training..."
    srun python training_scripts/albert_classification.py
elif [ "$MODEL" = "distilbert" ]; then
    echo "Running DistilBERT model training..."
    srun python training_scripts/distilbert_classification.py
elif [ "$MODEL" = "sbert" ]; then
    echo "Running Sentence-BERT model training..."
    srun python training_scripts/sbert_classification.py
else
    echo "Running BERT model training..."
    srun python training_scripts/bert_classification.py
fi

end_time=$(date +%s)
exec_time=$((end_time - start_time))

echo -e "\nJob completed at $(date)\n"

echo -e "total execution time: $exec_time seconds\n"

