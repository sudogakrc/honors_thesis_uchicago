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

def train():
    dm = POLUSADataModule(batch_size=20, data_dir=f'data/v{vers}')
    dm.setup(stage="weight_calc")

    model = BERT_unfreeze(dm.get_weights())

    # stop at lowest loss
    #trainer = pl.Trainer(default_root_dir='models/', gradient_clip_val=1.0, accelerator='auto', devices='1', 
    #                     callbacks=[EarlyStopping(monitor="val_loss", min_delta = 0.0001, patience=3, mode="min"),TQDMProgressBar(refresh_rate=10)])

    
    # stopping at 40 epochs
    trainer = pl.Trainer(default_root_dir='models/', max_epochs=40, gradient_clip_val=1.0, accelerator='auto', devices='1',
                         callbacks=[ModelCheckpoint(save_top_k=2, save_last=True, monitor="val_loss"),TQDMProgressBar(refresh_rate=10)])

    trainer.fit(model, dm)
    trainer.test(model, dm, ckpt_path='best')


if __name__ == '__main__':
    train()



