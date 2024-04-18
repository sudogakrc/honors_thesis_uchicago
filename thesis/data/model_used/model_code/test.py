import sys
vers = sys.argv[1]

import json
import os
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
from transformers import AutoModel, BertTokenizerFast
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from data import POLUSADataModule 
from model import BERT_unfreeze

def testing():
    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("data/bert-base-uncased", local_files_only=True, do_lower_case=True)
    model = BERT_unfreeze.load_from_checkpoint("models/lightning_logs/version_91868/checkpoints/last.ckpt")
    
    dm = POLUSADataModule(batch_size=20, data_dir=f'data/v{vers}')
    dm.setup(stage="weight_calc")

    # Initialize the data module for test data
    data_module = POLUSADataModule(batch_size=20, max_length=512, data_dir=f'data/v{vers}', test_nrows=None)
    data_module.setup(stage="test")

    # Initialize lists to store results
    article_texts = []
    probabilities = []

    # Set model to evaluation mode
    model.eval()

    # Iterate over the test data
    for batch in data_module.test_dataloader():
        sent_id, mask, labels = batch

        # Perform inference
        with torch.no_grad():
            # Get class probabilities
            preds = model(sent_id, mask)
            probs = torch.exp(preds)  # Convert log probabilities to probabilities

            # Assign an article number based on order of processing
            #article_numbers.extend(range(len(sent_id)))
           
            probabilities.extend(probs.tolist())
            # Get the original text of the articles from the tokenizer
            batch_text = [tokenizer.decode(sent_id[i], skip_special_tokens=True) for i in range(len(sent_id))]

            # Append article text and class probabilities to lists
            article_texts.extend(batch_text)

    # Create a DataFrame to store the results
    result_df = pd.DataFrame({"article_text": article_texts, "probability": probabilities})

    # Save the DataFrame to a CSV file
    result_df.to_csv("models/preds/left_right/v0/test_probabilities.csv", index=False)

if __name__ == '__main__':
    testing()




