import os
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import logging

logger = logging.getLogger("Brain Segmentation")

class Trainer():
    # TODO: w&b or tensorboard for monitoring logs
    def __init__(self, args, model, train_loader, valid_loader):
        self.args = args

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.model = model
        #self.model.apply(normal_init)
        #self.model.conv_final.add_module("activation", nn.Sigmoid())
        self.model = self.model.to(args.device)

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.loss_logs, self.metric_logs = {'train': [], 'eval': []}, {'train': [], 'eval': []}
        self.best_metric, self.best_epoch = -1, -1

    def run(self):
        for epoch in tqdm(range(self.args.epochs)):
            self.train_one_epoch()
            self.eval_one_epoch(epoch)
            self.save_checkpoint(epoch)
            self.print_logs(epoch)
        
        with open(f'{self.args.output_dir}/loss_logs.pkl', 'wb') as f:
            pickle.dump(self.loss_logs, f)
        with open(f'{self.args.output_dir}/metric_logs.pkl', 'wb') as f:
            pickle.dump(self.metric_logs, f)  
        
    def train_one_epoch(self):
        self.model.train()
        train_loss = 0
        for batch in self.train_loader:
            
            if len(batch) == 2:
                x, label = batch[0].to(self.args.device), batch[1].to(self.args.device)
            elif len(batch) == 3:
                x1, x2, label = batch[0].to(self.args.device), batch[1].to(self.args.device), batch[2].to(self.args.device)
                x = torch.cat([x1,x2], dim=1)
            #label = torch.nn.functional.one_hot(label)
            
            self.optimizer.zero_grad()
            out = nn.functional.softmax(self.model(x), dim=1)
            loss = self.loss_func(out, label)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        train_loss /= self.train_loader.__len__()
        self.loss_logs['train'].append(train_loss)

        return
    
    def eval_one_epoch(self, epoch):
        self.model.eval()

        with torch.no_grad():
            outs, labels = [], []
            for batch in self.valid_loader:
                if len(batch) == 2:
                    x, label = batch[0].to(self.args.device), batch[1].to(self.args.device)
                elif len(batch) == 3:
                    x1, x2, label = batch[0].to(self.args.device), batch[1].to(self.args.device), batch[2].to(self.args.device)
                    x = torch.cat([x1,x2], dim=1)
                
                out = torch.argmax(nn.functional.softmax(self.model(x), dim=1), dim=1)
                outs+=out.float().cpu().tolist()
                labels+=label.float().cpu().tolist()

            self.metric_logs['eval'].append({
                "acc": accuracy_score(labels, outs),
                "precision": precision_score(labels, outs, average='macro', zero_division=0.0),
                "recall": recall_score(labels, outs, average='macro', zero_division=0.0),
                "f1": f1_score(labels, outs, average='macro', zero_division=0.0)
            })

        return

    def save_checkpoint(self, epoch):
        epoch_metric_value = self.metric_logs['eval'][epoch]['acc']
        if epoch_metric_value > self.best_metric:
            self.best_metric = epoch_metric_value
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, "best_model.pth"))
            logger.info(f'model was saved, epoch {epoch} is the best.')


    def print_logs(self, epoch):
        logger.info(
            "\n"
            f"{'='*20}\n"
            f"Epoch: {epoch}\n"
            f"train loss: {self.loss_logs['train'][epoch]}\n"
            f"eval metric: {self.metric_logs['eval'][epoch]}\n"
            f"{'='*20}\n"
        )


