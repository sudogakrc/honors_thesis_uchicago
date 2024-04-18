### no stride

import sys
vers = sys.argv[1]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoModel, BertModel, BertTokenizer
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as pl

model_prefix = 'left_right'

class POLUSADataModule(pl.LightningDataModule): #batch_size = 32, max_length = 512
    def __init__(self, batch_size=20, max_length=512, data_dir=f'data/v{vers}', test_nrows=None):
        super().__init__()
        self.data_dir = data_dir
        self.prefix = model_prefix
        self.batch_size = batch_size
        self.max_length = max_length
        self.test_nrows = test_nrows
        self.tokenizer = AutoTokenizer.from_pretrained("data/bert-base-uncased", local_files_only=True, do_lower_case=True)

    def tokenize(self, df):
        tokens = self.tokenizer.__call__(
                df['body'].tolist(),
                padding = True, max_length=self.max_length,
                truncation = True, return_tensors='pt')
                
        # convert the integer sequences to tensors.
        seq = torch.tensor(tokens['input_ids'])
        mask = torch.tensor(tokens['attention_mask'])
        y = torch.tensor(df['label'].tolist())

        return TensorDataset(seq, mask, y)

    def setup(self, stage):
        if stage == "weight_calc":
            df_train = pd.read_csv(f'{self.data_dir}/{self.prefix}_train.csv', nrows=self.test_nrows)
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df_train['label']), y=df_train['label'])

            self.weights = torch.tensor(class_weights, dtype=torch.float)

        if stage == "fit":
            df_train = pd.read_csv(f'{self.data_dir}/{self.prefix}_train.csv', nrows=self.test_nrows)
            df_val = pd.read_csv(f'{self.data_dir}/{self.prefix}_val.csv', nrows=self.test_nrows)

            self.train_data = self.tokenize(df_train) 
            self.val_data = self.tokenize(df_val)

        if stage == "test" or stage == "predict":
            df_test = pd.read_csv(f'{self.data_dir}/{self.prefix}_test.csv', nrows=self.test_nrows)
            self.test_data = self.tokenize(df_test)

    def get_weights(self):
        return self.weights
        
    def train_dataloader(self):
        return DataLoader(self.train_data, drop_last=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, drop_last=True, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, drop_last=True, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_data, drop_last=True, batch_size=self.batch_size)
