import sys
vers = sys.argv[1]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AdamW,
    AutoConfig,
    BertModel,
    AutoModel,
    BertTokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding
    )
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as pl
import os.path


model_lean = "left_right"

epoch = 0


class BERT_unfreeze(pl.LightningModule):   
    def __init__(self, weights, learning_rate=1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_step_outputs = []
        self.val_step_outputs = []

        self.loss = nn.NLLLoss(weight=weights)
        self.learning_rate = learning_rate

        self.bert = BertModel.from_pretrained("data/bert-base-uncased", local_files_only=True, num_labels=2, return_dict=False)

        #freeze the pretrained layers except 1-3
        for layer in self.bert.encoder.layer[:9]:
            for param in layer.parameters():
                param.requires_grad = False
        
        # dropout layer
        self.dropout = nn.Dropout(0.2)
        
         # relu activation function
        self.relu =  nn.ReLU()
  
         # dense layer 1
        self.fc1 = nn.Linear(768,512)
        
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,2)
  
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)
  
    #define the forward pass
    def forward(self, sent_id, mask):
        
        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        
        x = self.fc1(cls_hs)
        
        x = self.relu(x)
        
        x = self.dropout(x) #gives you the logits
    
        # output layer
        x = self.fc2(x)
        
        # apply softmax activation
        x = self.softmax(x) #gives you the probablities
        
        return x

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate) 
        return optimizer


    #use with train and validaton step output metrics
    def export_metrics(self, acc, loss, step):
        metrics_filename = f"models/preds/{model_lean}/v{vers}/{step}_metrics.csv"
        
        metrics_df = pd.DataFrame([[epoch, acc, loss]], columns = ['epoch', 'acc', 'loss'])

        if os.path.isfile(metrics_filename):
            metrics_in_data = pd.read_csv(metrics_filename)
            metrics_out_data = pd.concat([metrics_in_data, metrics_df])
            metrics_out_data.to_csv(metrics_filename, index = False)

        else:
            metrics_df.to_csv(metrics_filename, index = False)

        return 

    # function to train the model
    def training_step(self, train_batch, batch_idx):
        sent_id, mask, labels = train_batch

        # get model predictions for the current batch
        preds = self(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = self.loss(preds, labels)
        acc = torch.sum(labels == torch.argmax(preds, dim=1)).item() / len(labels)
        
        self.train_step_outputs.append({"train_acc": acc, "loss": loss})

        metrics = {"train_acc": acc, "loss": loss}

        return metrics

    def on_train_epoch_end(self):
        step = "train"

        outputs = self.train_step_outputs
        
        acc = sum([x['train_acc'] for x in outputs]) / len(outputs)
        loss = sum([x['loss'] for x in outputs]) / len(outputs)

        self.export_metrics(acc, loss, step)

    def validation_step(self, val_batch, batch_idx):
        loss, acc = self._shared_eval_step(val_batch, batch_idx)
        self.val_step_outputs.append({"val_acc": acc, "val_loss": loss})

        metrics = {"val_acc": acc, "val_loss": loss}
        
        self.log_dict(metrics)
        
        return metrics

    def on_validation_epoch_end(self):
        global epoch
        epoch = epoch + 1

        step = "val"

        outputs = self.val_step_outputs
        
        acc = sum([x['val_acc'] for x in outputs]) / len(outputs)
        loss = sum([x['val_loss'] for x in outputs]) / len(outputs)

        self.export_metrics(acc, loss, step)

    def vector_rep(self, test_batch):
        sent_id, mask, labels = test_batch
        
        with torch.no_grad():
            outputs = self.bert(sent_id, attention_mask=mask)

        embeddings = outputs[0]

        vec_filename = f"models/embeddings/polBERT_pol_embeddings.csv"

        vec_df = pd.DataFrame([[labels,embeddings]], columns = ['labels', 'embeddings'])

        if os.path.isfile(vec_filename):
            vec_in_data = pd.read_csv(vec_filename)
            vec_out_data = pd.concat([vec_in_data, vec_df])
            vec_out_data.to_csv(vec_filename, index = False)

        else:
            vec_df.to_csv(vec_filename, index = False)

        return

    def test_step(self, test_batch, batch_idx):
        loss, acc = self._shared_eval_step(test_batch, batch_idx)
       
        metrics = {"test_acc": acc, "test_loss": loss}
        
        self.log_dict(metrics)

        # extracting embeddings
        self.vector_rep(test_batch)
        
        return metrics

    def predict_step(self, predict_batch, batch_idx):
        sent_id, mask, labels = predict_batch

        return self(sent_id, mask)

    def _shared_eval_step(self, batch, batch_idx):
        sent_id, mask, labels = batch

        # model predictions
        preds = self(sent_id, mask)

        # compute the validation loss between actual and predicted values
        loss = self.loss(preds, labels)

        # compute accuracy between actual and predicted values
        acc = torch.sum(labels == torch.argmax(preds, dim=1)).item() / len(labels)

        return loss, acc
