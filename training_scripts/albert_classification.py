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
from transformers import AlbertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
torch.cuda.empty_cache()
device = 'cuda' if cuda.is_available() else 'cpu'
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = AlbertForSequenceClassification.from_pretrained(
    "albert-base-v2",
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)
print('device:', device)
model.to(device)

class TextDataset(Dataset):
    def __init__(self, texts, targets, max_length=180, tokenizer_name='albert-base-v2'):
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
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': torch.tensor(target, dtype=torch.long)
        }

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
train_labels = pd.read_csv('data/train_labels.csv')
test_labels = pd.read_csv('data/test_labels.csv')

# Create datasets
train_dataset = TextDataset(
    texts=train_df['headline'].values,
    targets=train_labels['is_sarcastic'].values
)

test_dataset = TextDataset(
    texts=test_df['headline'].values,
    targets=test_labels['is_sarcastic'].values
)

# Create dataloaders
batch_size = 32

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    eps=1e-8
)

# Training loop
epochs = 3
total_steps = len(train_dataloader) * epochs

for epoch in range(epochs):
    print(f'\n================ Epoch {epoch+1} / {epochs} ================')
    
    t0 = time.time()
    total_train_loss = 0
    model.train()
    
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(f'  Batch {step:>5,}  of  {len(train_dataloader):>5,}.    Elapsed: {elapsed}.')
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        
        model.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets
        )
        
        loss = outputs.loss
        total_train_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    
    # Evaluation
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(batch['targets'].cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    print("\n---TRAIN METRICS---")
    print(f"Loss: {avg_train_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nRunning validation ...")
    
    all_preds = []
    all_labels = []
    total_test_loss = 0
    
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )
        
        loss = outputs.loss
        total_test_loss += loss.item()
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(batch['targets'].cpu().numpy())
    
    avg_test_loss = total_test_loss / len(test_dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    print("\n---TEST METRICS---")
    print(f"Loss: {avg_test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# Save the model
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/fine_tuned_albert')
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")

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
    model_size_mb = sum(os.path.getsize(os.path.join(output_dir, f)) 
                      for f in os.listdir(output_dir) 
                      if os.path.isfile(os.path.join(output_dir, f))) / (1024 * 1024)
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
metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results/albert_deployability_metrics.json')
with open(os.path.abspath(metrics_path), 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"\nMetrics saved to {metrics_path}") 