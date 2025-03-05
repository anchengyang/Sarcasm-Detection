# News Headlines Dataset For Sarcasm Detection

This project implements a binary text classification model to detect sarcasm in news headlines using transformer-based models including BERT (Bidirectional Encoder Representations from Transformers) and RoBERTa (Robustly Optimized BERT Pretraining Approach).

## Dataset
The dataset is sourced from Kaggle and contains news headlines labeled as sarcastic (1) or not sarcastic (0).

Link to the dataset: [News Headlines Dataset For Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection/data)

## Model Architectures

### BERT Model
- Fine-tuned BERT model (`bert-base-uncased`)
- Binary classification head
- Trained for 3 epochs with AdamW optimizer (learning rate: 2e-5)
- Batch size of 32

### RoBERTa Model
- Fine-tuned RoBERTa model (`roberta-base`)
- Binary classification head
- Trained for 3 epochs with AdamW optimizer (learning rate: 2e-5)
- Batch size of 32

## Results

### BERT Results
After training for 3 epochs, the BERT model achieves:
- Test Accuracy: 92.42%
- Test Precision: 90.17%
- Test Recall: 94.05%
- Test F1-Score: 92.07%

### RoBERTa Results
Results will be available after training.

## Project Structure
- `notebooks/`: Contains Jupyter notebooks for exploration and development
  - `bert/`: BERT-specific notebooks
    - `01_eda.ipynb`: Exploratory data analysis of the dataset
    - `02_embeddings.ipynb`: Text embedding exploration with BERT
    - `03_training.ipynb`: BERT model training and evaluation
  - `roberta/`: RoBERTa-specific notebooks
    - `01_eda.ipynb`: Exploratory data analysis for RoBERTa
    - `02_embeddings.ipynb`: Text embedding exploration with RoBERTa
    - `03_training.ipynb`: RoBERTa model training and evaluation
  - `04_model_comparison.ipynb`: Comparative analysis of BERT and RoBERTa models
- `training_scripts/`: Contains production-ready training code
  - `bert_classification.py`: End-to-end script for BERT model training
  - `roberta_classification.py`: End-to-end script for RoBERTa model training
- `data/`: Contains the dataset files and training statistics
- `models/`: Contains saved model checkpoints
  - `fine_tuned_bert/`: Saved BERT model and tokenizer
  - `fine_tuned_roberta/`: Saved RoBERTa model and tokenizer
- `slurm_script.sh`: Script for running the training job on a SLURM cluster

## Setup and Installation

### Local Development
1. Clone this repository
2. Create a virtual environment
3. Install required packages:
   ```
   pip install torch pandas transformers scikit-learn matplotlib seaborn bertviz
   ```
4. Run the notebooks in the `notebooks/` directory

### Training on SLURM Cluster
1. Ensure data is properly placed in the `data/` directory
2. Submit the job using:
   ```
   # For BERT model (default)
   sbatch slurm_script.sh
   
   # For RoBERTa model
   sbatch slurm_script.sh roberta
   ```
3. Monitor output in `training_job_output.txt` and errors in `training_job_error.txt`

## Usage
Once trained, the models can be loaded and used for inference:

### Using BERT Model
```python
from transformers import BertForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("path/to/saved/bert/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/saved/bert/model")

# Prepare input
headline = "Scientists discover new planet that looks exactly like Earth"
inputs = tokenizer(headline, return_tensors="pt", padding=True, truncation=True)

# Get prediction
outputs = model(**inputs)
prediction = outputs.logits.argmax().item()
print("Sarcastic" if prediction == 1 else "Not sarcastic")
```

### Using RoBERTa Model
```python
from transformers import RobertaForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("path/to/saved/roberta/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/saved/roberta/model")

# Prepare input
headline = "Scientists discover new planet that looks exactly like Earth"
inputs = tokenizer(headline, return_tensors="pt", padding=True, truncation=True)

# Get prediction
outputs = model(**inputs)
prediction = outputs.logits.argmax().item()
print("Sarcastic" if prediction == 1 else "Not sarcastic")
```
