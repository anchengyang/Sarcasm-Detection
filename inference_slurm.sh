#!/bin/sh
#SBATCH --job-name=running-inference
#SBATCH --gpus=a100-80:1
#SBATCH --time=1:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0725272@u.nus.edu

# Default model is BERT if not specified
MODEL=${1:-bert}

# Create the output folder based on the model name
OUTPUT_DIR="profiler_results/${MODEL}"
mkdir -p "$OUTPUT_DIR"

# Redirect stdout and stderr to dynamic files based on the model
exec > "${OUTPUT_DIR}/inference_job_output.txt" 2> "${OUTPUT_DIR}/inference_job_error.txt"


echo "job is running on $(hostname), started at $(date)"
echo "Running inference with model: $MODEL"
echo "\n--- Installing packages ---\n"
pip install torch pandas transformers scikit-learn torch-tb-profiler tensorboard
echo "\n--- Finished installing packages, starting model inference ---\n"

start_time=$(date +%s)

if [ "$MODEL" = "roberta" ]; then
    echo "Running RoBERTa model inference..."
    srun python inference_scripts/measure_bert.py --model roberta
elif [ "$MODEL" = "albert" ]; then
    echo "Running ALBERT model inference..."
    srun python inference_scripts/measure_bert.py --model albert
elif [ "$MODEL" = "distilbert" ]; then
    echo "Running DistilBERT model inference..."
    srun python inference_scripts/measure_bert.py --model distilbert
elif [ "$MODEL" = "sbert" ]; then
    echo "Running Sentence-BERT model inference..."
    srun python inference_scripts/measure_bert.py --model sbert
else
    echo "Running BERT model inference..."
    srun python inference_scripts/measure_bert.py --model bert
fi

end_time=$(date +%s)
exec_time=$((end_time - start_time))

echo -e "\nJob completed at $(date)\n"

echo -e "total execution time: $exec_time seconds\n"

