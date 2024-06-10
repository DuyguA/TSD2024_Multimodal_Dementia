from transformers import AdamW
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


import torch
import torch.nn as nn
import numpy as np

import os, copy, json
from tqdm import tqdm

from data_loader_vit import adresso_loader
from multimodal_vit import BertImage

class Trainer:
  def __init__(self, args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs('ckp', exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader = adresso_loader(phase='train', batch_size=args.batch_size, shuffle=True)
    test_loader = adresso_loader(phase='test', batch_size=args.val_batch_size)
    print('Done\n')

    if torch.cuda.device_count() > 0:
      print(f"{torch.cuda.device_count()} GPUs found")

    print('Initializing model....')
    model = BertImage(nfinetune=args.nfinetune)

    model = nn.DataParallel(model)
    model.to(device)
    params = model.parameters()

    optimizer = AdamW(params, lr=args.lr, weight_decay=0.01)
    self.device = device
    self.model = model
    self.optimizer = optimizer
    self.binary_cross = nn.BCELoss()
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.args = args
    self.epoch_accuracies = []
    self.all_losses = []

  def train(self):
    best_epoch = 0
    print("First epoch will start soon")
    for epoch in range(self.args.epochs):
      print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}")
      loss = self.train_epoch()
      binary_acc  = self.eval()
      print(f'Binary Acc: {binary_acc:.3f}')
      self.epoch_accuracies.append(round(binary_acc, 3)) # append epoch accuracies
      print("Training finished here are epoch accuracies")
      with open("epoch_acc.txt", "w") as ofile:
        for eacc in self.epoch_accuracies:
          ofile.write(str(eacc) + "\n")
      print("Now all losses")
      with open("losses.txt", "w") as ofile:
        for eloss in self.all_losses:
          ofile.write(str(eloss) + "\n")

  def train_epoch(self):
    self.model.train()
    epoch_loss = 0
    for i, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
        self.optimizer.zero_grad()
        pixels = batch['pixels'].to(self.device) #input
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)  # target
        label_preds = self.model(pixels, input_ids, attention_mask)
        labels = labels.reshape(-1)
        label_preds = label_preds.reshape(-1)
        loss = self.binary_cross(label_preds, labels.float())
        

        loss.backward()
        self.optimizer.step()
        interval = max(len(self.train_loader) // 20, 1)
        if i % interval == 0 or i == len(self.train_loader) - 1:
            lloss = round(loss.item(), 3)
            #print(f'Batch: {i + 1}/{len(self.train_loader)}\ttotal loss: {loss.item():.3f}\temotion loss:{emotion_loss.item():.3f}\tpair loss:{pair_loss.item():.3f}')
            print(f'Batch: {i + 1}/{len(self.train_loader)}\ttotal loss: {lloss:.3f}')
            self.all_losses.append(lloss) # append epoch accuracies
        epoch_loss += loss.item()
    return epoch_loss / len(self.train_loader)

  def eval(self):
    self.model.eval()
    label_pred = []
    label_true = []

    mimse_pred = []
    mimse_true = []
    loader = self.test_loader 

    with torch.no_grad():
      for i, batch in enumerate(loader):
        pixels = batch['pixels'].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch['labels']
        label_preds  = self.model(pixels, input_ids, attention_mask)
        print(label_preds)
        label_pred += label_preds.detach().to('cpu').flatten().round().long().tolist()

        labels = labels.reshape(-1)
        label_true += labels.detach().to('cpu').tolist()

                
    print(label_true, "true labels")
    print(label_pred, "pred labels")
    label_acc = accuracy_score(label_true, label_pred)
    torch.save(self.model.state_dict(), 'saved_models/attention_cnn.pth')
    target_names = ['NonDementia', 'Dementia']
    print(classification_report(label_true, label_pred, target_names=target_names))
    return label_acc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=48, help='batch size of training')
    parser.add_argument('--val_batch_size', type=int, default=4, help='batch size of testing')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--gpu', type=str, default='', help='GPUs to use')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--nfinetune', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print(args)
    engine = Trainer(args)
    engine.train()

