import torch
import pandas as pd
from torch.utils.data import DataLoader
import os

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch import cuda
from transformers import AlbertForSequenceClassification, AdamW
import time
import datetime
import numpy as np
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