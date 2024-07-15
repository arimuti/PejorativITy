import tensorflow as tf
import torch
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import random
from sklearn.model_selection import train_test_split
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup)
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pathlib import Path

#read data
data_path = Path(__file__).parent.parent.resolve().joinpath('datasets', 'pejorative', 'pejorative_dataset_it_train.xlsx')
data = pd.read_excel(data_path)

# Split dataset in traning and validation(test)
X_train, X_val, Y_train, Y_val = train_test_split(
    data.index.values,
    data.pejorative.values,
    test_size=0.10,
    random_state=17,
    stratify=data.pejorative.values
)
# Check datasets composition
data['data_type'] = ['not_set'] * data.shape[0]
data.loc[X_train, 'data_type'] = 'train'
data.loc[X_val, 'data_type'] = 'val'
data.groupby(['pejorative', 'data_type']).count()


tokenizer = AutoTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0", use_fast=False)

# Encode training dataset using the tokenizer
encoded_data_train = tokenizer.batch_encode_plus(
    data[data.data_type == 'train'].text.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)

# Encode validation dataset using the tokenizer
encoded_data_val = tokenizer.batch_encode_plus(
    data[data.data_type == 'val'].text.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)

# Extract IDs, attention masks and labels from training dataset
input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(data[data.data_type == 'train'].pejorative.values)

# Extract IDs, attention masks and labels from validation dataset
input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(data[data.data_type == 'val'].pejorative.values)

# Create train and validation dataset from extracted features
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

# Define the size of each batch
batch_size = 4

# Load training dataset
dataloader_train= DataLoader(
    dataset_train,
    sampler=RandomSampler(dataset_train),
    batch_size=batch_size)

# Load valuation dataset
dataloader_val= DataLoader(
    dataset_val,
    sampler=RandomSampler(dataset_val),
    batch_size=batch_size)

# Load pre-trained BERT model
from transformers import BertConfig
config = BertConfig.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
config.output_hidden_states = True
model = BertForSequenceClassification.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0", config=config)

# Define model optimizer -> Adam
optimizer = AdamW(
    model.parameters(),
    lr = 1e-5,
    eps=1e-8
)
# Define model scheduler
epochs = 4
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)

# Define random seeds
seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Define processor type for torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

from sklearn.metrics import f1_score

# Returns the F1 score computed on the predictions
def f1_score_func(preds, labels):
    preds_flat=np.argmax(preds, axis=1).flatten()
    labels_flat=labels.flatten()
    return f1_score(labels_flat, preds_flat, average='macro')
  # Evaluates the model using the validation set
def evaluate(dataloader_val):
  model.eval()
  loss_val_total = 0
  predictions, true_vals = [], []

  for batch in dataloader_val:
      batch = tuple(b.to(device) for b in batch)
      inputs = {'input_ids': batch[0],
        'attention_mask': batch[1],
        'labels': batch[2],
        }

      with torch.no_grad():
          outputs = model(**inputs)

      loss = outputs[0]
      logits = outputs[1]
      loss_val_total += loss.item()

      logits = logits.detach().cpu().numpy()
      label_ids = inputs['labels'].cpu().numpy()
      predictions.append(logits)
      true_vals.append(label_ids)

  loss_val_avg = loss_val_total / len(dataloader_val)

  predictions = np.concatenate(predictions, axis=0)
  true_vals = np.concatenate(true_vals, axis=0)

  return loss_val_avg, predictions, true_vals

for epoch in tqdm(range(1, epochs + 1)):

    model.train()  # model is training

    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()  # to backpropagate

        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                      1.0)  # prevents the gradient from being too small or too big

        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})


    torch.save(model, path)
    tqdm.write(f'\nEpoch {epoch}/{epochs}')

    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}') 

    val_loss, predictions, true_vals = evaluate(dataloader_val)  # to check overtraining (or overfitting)
    val_f1 = f1_score_func(predictions, true_vals)

    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score(macro) : {val_f1}')

