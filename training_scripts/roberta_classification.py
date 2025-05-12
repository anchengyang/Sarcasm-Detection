import torch
import pandas as pd
from torch.utils.data import DataLoader
import os
import time
import datetime
import numpy as np
import json
import psutil
import gc
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch import cuda
from transformers import RobertaForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
torch.cuda.empty_cache()
device = 'cuda' if cuda.is_available() else 'cpu'
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)
print('device:', device)
model.to(device)

class TextDataset(Dataset):
    def __init__(self, texts, targets, max_length=180, tokenizer_name='roberta-base'):
        self.texts = texts
        self.targets = targets
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        return {
            'input_ids': torch.as_tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.as_tensor(attention_mask, dtype=torch.long),
            'targets': torch.as_tensor(target, dtype=torch.long),
            'text': text
        }

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Construct absolute paths
train_path = os.path.join(BASE_DIR, '../data/train.csv')
test_path = os.path.join(BASE_DIR, '../data/test.csv')
train_label_path = os.path.join(BASE_DIR, '../data/train_labels.csv')
test_label_path = os.path.join(BASE_DIR, '../data/test_labels.csv')

# Read the CSV files
X_train = pd.read_csv(os.path.abspath(train_path))
X_test = pd.read_csv(os.path.abspath(test_path))

# load the labels
y_train = pd.read_csv(os.path.abspath(train_label_path))
y_test = pd.read_csv(os.path.abspath(test_label_path))

# create the datasets and dataloaders
train_dataset = TextDataset(texts=X_train['headline'], targets=y_train['is_sarcastic'])
test_dataset = TextDataset(texts=X_test['headline'], targets=y_test['is_sarcastic'])

BATCH_SIZE = 32
torch.manual_seed(1702)
train_loader = DataLoader(train_dataset, 
                          batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(test_dataset,
                         batch_size=len(test_dataset))
next(iter(train_loader))

EPOCHS = 3

optimizer = AdamW(model.parameters(),
    lr = 2e-5, 
    eps = 1e-8
)

training_stats = []
epoch_loss_train = []
total_t0 = time.time()

# TRAINING
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    print("")
    print("================ Epoch {:} / {:} ================".format(epoch, EPOCHS))
    train_all_predictions = []
    train_all_true_labels = []
    for step, data in enumerate(train_loader):
        if step % 40 == 0 and not step == 0:
            elapsed = int(round(time.time() - t0))
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print(
                "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                    step, len(train_loader), elapsed
                )
            )

        targets = data["targets"].to(device)
        mask = data["attention_mask"].to(device)
        ids = data["input_ids"].to(device)

        model.zero_grad()

        # Note: RoBERTa doesn't use token_type_ids
        loss, logits = model(
            ids, attention_mask=mask, labels=targets
        ).to_tuple()
        epoch_loss_train.append(loss.item())

        cpu_logits = logits.cpu().detach().numpy()
        train_all_predictions.extend(np.argmax(cpu_logits, axis=1).flatten())
        train_all_true_labels.extend(targets.cpu().numpy())

        loss.backward()
        optimizer.step()
    train_accuracy = accuracy_score(train_all_true_labels, train_all_predictions)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        train_all_true_labels, train_all_predictions, average="binary"
    )
    print("")
    print('---TRAIN METRICS---')
    print(f"Loss: {np.mean(epoch_loss_train):.4f}")
    print(f"Accuracy: {train_accuracy:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall: {train_recall:.4f}")
    print(f"F1-Score: {train_f1:.4f}")
    print("")
    
    # VALIDATION
    print("Running validation ...")
    print("")
    model.eval()
    epoch_loss_test = []
    test_all_predictions = []
    test_all_true_labels = []
    for data in test_loader:
        targets = data["targets"].to(device)
        mask = data["attention_mask"].to(device)
        ids = data["input_ids"].to(device)
        
        with torch.no_grad():
            # Note: RoBERTa doesn't use token_type_ids
            loss, logits = model(ids, attention_mask=mask, labels=targets).to_tuple()
            
        epoch_loss_test.append(loss.item())
        cpu_logits = logits.cpu().detach().numpy()
        test_all_predictions.extend(np.argmax(cpu_logits, axis=1).flatten())
        test_all_true_labels.extend(targets.cpu().numpy())
    test_accuracy = accuracy_score(test_all_true_labels, test_all_predictions)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_all_true_labels, test_all_predictions, average="binary"
    )
    print("")
    print('---TEST METRICS---')
    print(f"Loss: {np.mean(epoch_loss_test):.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    
    training_stats.append(
            {
            'epoch': epoch,
            'Training Loss': np.mean(epoch_loss_train),
            'Training Accuracy': train_accuracy,
            'Training Precision': train_precision,
            'Training Recall': train_recall,
            'Training F1': train_f1,
            'Validation Loss': np.mean(epoch_loss_test),
            'Validation Accuracy': test_accuracy,
            'Validation Precision': test_precision,
            'Validation Recall': test_recall,
            'Validation F1': test_f1
        }
    )

df_statistics = pd.DataFrame(data=training_stats)
df_statistics = df_statistics.set_index('epoch')
statistics_path = os.path.join(BASE_DIR, '../data/roberta_training_statistics.csv')
df_statistics.to_csv(os.path.abspath(statistics_path))

# Create models directory if it doesn't exist
models_dir = os.path.join(BASE_DIR, '../models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# save model
model_save_path = os.path.join(models_dir, 'fine_tuned_roberta')
model.save_pretrained(os.path.abspath(model_save_path))

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
tokenizer.save_pretrained(model_save_path)

print(f"Model and tokenizer saved to {model_save_path}")

# After model is saved
# Add deployability metrics measurement
print("\n--- Measuring Deployability Metrics ---\n")

def measure_deployability_metrics(model, test_dataset, batch_size=32):
    """
    Measure and return deployability metrics for the model
    """
    metrics = {}
    
    # 1. Model Size
    model_size_params = sum(p.numel() for p in model.parameters())
    metrics["model_size_parameters"] = model_size_params
    
    # Get model size on disk (in MB)
    model_size_mb = sum(os.path.getsize(os.path.join(model_save_path, f)) 
                      for f in os.listdir(model_save_path) 
                      if os.path.isfile(os.path.join(model_save_path, f))) / (1024 * 1024)
    metrics["model_size_mb"] = round(model_size_mb, 2)
    
    # Create a small single-item loader for latency test
    single_item_dataset = torch.utils.data.Subset(test_dataset, [0])  # Just the first item
    single_loader = DataLoader(single_item_dataset, batch_size=1, shuffle=False)
    
    # Batch loader for throughput test
    batch_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Inference Latency (single item)
    model.eval()
    # Warm up
    for _ in range(5):
        with torch.no_grad():
            for data in single_loader:
                ids = data["input_ids"].to(device)
                mask = data["attention_mask"].to(device)
                _ = model(ids, attention_mask=mask)
    
    # Measure latency
    latencies = []
    for _ in range(20):  # Average over 20 runs
        start_time = time.time()
        with torch.no_grad():
            for data in single_loader:
                ids = data["input_ids"].to(device)
                mask = data["attention_mask"].to(device)
                _ = model(ids, attention_mask=mask)
        latencies.append((time.time() - start_time) * 1000)  # Convert to ms
    
    metrics["inference_latency_ms"] = round(np.mean(latencies), 2)
    metrics["inference_latency_std_ms"] = round(np.std(latencies), 2)
    
    # 3. Throughput (batch processing)
    # Warm up
    with torch.no_grad():
        for data in batch_loader:
            ids = data["input_ids"].to(device)
            mask = data["attention_mask"].to(device)
            _ = model(ids, attention_mask=mask)
    
    # Measure throughput
    start_time = time.time()
    with torch.no_grad():
        for data in batch_loader:
            ids = data["input_ids"].to(device)
            mask = data["attention_mask"].to(device)
            _ = model(ids, attention_mask=mask)
    
    total_time = time.time() - start_time
    total_samples = len(test_dataset)
    metrics["throughput_samples_per_second"] = round(total_samples / total_time, 2)
    
    # 4. Memory Usage
    # Force garbage collection before measuring memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Measure baseline memory
    baseline_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    
    # Memory usage during inference
    with torch.no_grad():
        for data in batch_loader:
            ids = data["input_ids"].to(device)
            mask = data["attention_mask"].to(device)
            _ = model(ids, attention_mask=mask)
            break  # One batch is enough for memory measurement
    
    peak_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    metrics["memory_usage_mb"] = round(peak_mem - baseline_mem, 2)
    
    return metrics

# Run metrics measurement
metrics = measure_deployability_metrics(model, test_dataset)

# Print metrics
print("\nDeployability Metrics:")
print(f"Model Size: {metrics['model_size_parameters']:,} parameters ({metrics['model_size_mb']} MB)")
print(f"Inference Latency: {metrics['inference_latency_ms']} ms (± {metrics['inference_latency_std_ms']} ms)")
print(f"Throughput: {metrics['throughput_samples_per_second']} samples/second")
print(f"Memory Usage: {metrics['memory_usage_mb']} MB")

# Save metrics to file
os.makedirs('../results', exist_ok=True)
metrics_path = os.path.join(BASE_DIR, '../results/roberta_deployability_metrics.json')
with open(os.path.abspath(metrics_path), 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"\nMetrics saved to {metrics_path}") 