# News Headlines Dataset For Sarcasm Detection

This project implements a binary text classification model to detect sarcasm in news headlines using BERT (Bidirectional Encoder Representations from Transformers).

## Dataset
The dataset is sourced from Kaggle and contains news headlines labeled as sarcastic (1) or not sarcastic (0).

Link to the dataset: [News Headlines Dataset For Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection/data)

## Model Architecture
- Fine-tuned BERT model (`bert-base-uncased`)
- Binary classification head
- Trained for 3 epochs with AdamW optimizer (learning rate: 2e-5)
- Batch size of 32

## Results
After training for 3 epochs, the model achieves:
- Test Accuracy: 92.42%
- Test Precision: 90.17%
- Test Recall: 94.05%
- Test F1-Score: 92.07%

## Project Structure
- `notebooks/`: Contains Jupyter notebooks for exploration and development
  - `01_eda.ipynb`: Exploratory data analysis of the dataset
  - `02_embeddings.ipynb`: Text embedding exploration
  - `03_training.ipynb`: Model training and evaluation (development version)
- `training_scripts/`: Contains production-ready training code
  - `bert_classification.py`: End-to-end script for model training
- `data/`: Contains the dataset files and training statistics
- `slurm_script.sh`: Script for running the training job on a SLURM cluster

## Setup and Installation

### Local Development
1. Clone this repository
2. Create a virtual environment
3. Install required packages:
   ```
   pip install torch pandas transformers scikit-learn
   ```
4. Run the notebooks in the `notebooks/` directory

### Training on SLURM Cluster
1. Ensure data is properly placed in the `data/` directory
2. Submit the job using:
   ```
   sbatch slurm_script.sh
   ```
3. Monitor output in `training_job_output.txt` and errors in `training_job_error.txt`

## Usage
Once trained, the model can be loaded and used for inference:

```python
from transformers import BertForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("path/to/saved/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/saved/model")

# Prepare input
headline = "Scientists discover new planet that looks exactly like Earth"
inputs = tokenizer(headline, return_tensors="pt", padding=True, truncation=True)

# Get prediction
outputs = model(**inputs)
prediction = outputs.logits.argmax().item()
print("Sarcastic" if prediction == 1 else "Not sarcastic")
```
