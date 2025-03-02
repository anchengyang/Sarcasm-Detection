import torch
import pandas as pd
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch import cuda
from transformers import BertForSequenceClassification, AdamW, BertConfig
import time
import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
torch.cuda.empty_cache()
device = 'cuda' if cuda.is_available() else 'cpu'
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)
print('device:', device)
model.to(device)

class TextDataset(Dataset):
    def __init__(self, texts, targets, max_length=180, tokenizer_name='bert-base-uncased'):
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
    
# load the data
X_train = pd.read_csv('../data/train.csv')
X_test = pd.read_csv('../data/test.csv')

# load the labels
y_train = pd.read_csv('../data/train_labels.csv')
y_test = pd.read_csv('../data/test_labels.csv')

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

        loss, logits = model(
            ids, token_type_ids=None, attention_mask=mask, labels=targets
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
            loss, logits = model(ids, token_type_ids=None, attention_mask=mask, labels=targets).to_tuple()
            
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
df_statistics.to_csv('../data/training_statistics.csv')
