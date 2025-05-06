import os
import time
import copy
import torch
import datetime
import numpy as np
from tqdm import tqdm
from lib.metrics import *

class trainer(object):
    def __init__(self, args, model, trainset_loader, valset_loader, testset_loader, scaler_dict, adj_dict, loss, 
                 optimizer, scheduler):
        super(trainer, self).__init__()
        self.args = args
        self.model = model
        self.trainset_loader = trainset_loader
        self.valset_loader = valset_loader
        self.testset_loader = testset_loader
        self.scaler_dict = scaler_dict
        self.adj_dict = adj_dict
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def train_one_epoch(self):
        self.model.train()
        batch_loss_list = []
        num_step = 0
        for dataset_name, x_batch, y_batch in self.trainset_loader:
            x_batch = x_batch.to(self.args.device)
            y_batch = y_batch.to(self.args.device)

            out_batch = self.model(x_batch)
            
            if not self.args.use_revin:
                out_batch = self.scaler_dict[dataset_name[0]].inverse_transform(out_batch)
                y_batch = self.scaler_dict[dataset_name[0]].inverse_transform(y_batch)
            
            loss_batch = self.loss(out_batch.float(), y_batch.float())
            self.optimizer.zero_grad()
            loss_batch.backward()
            
            if self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            self.optimizer.step()
            
            batch_loss_list.append(loss_batch.item())
            num_step += 1
            if num_step % self.args.log_step == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(datetime.datetime.now(), "step:  "+str(num_step)+"  train loss is:  "+str(np.mean(batch_loss_list))+"  current_lr is:  "+str(current_lr))

        self.scheduler.step()
        train_loss = np.mean(batch_loss_list)
        return train_loss

    def train_multi_epoch(self):
        wait = 0
        min_val_loss = np.inf

        train_loss_list = []
        val_loss_list = []

        for epoch in tqdm(range(self.args.max_epochs)):
            train_loss = self.train_one_epoch()
            
            if train_loss > 1e6:
                print('Gradient explosion detected...')
                break
            
            train_loss_list.append(train_loss)

            val_loss = self.eval_model(epoch)
            val_loss_list.append(val_loss)
            
            self.test_model(epoch)

            if val_loss < min_val_loss:
                wait = 0
                min_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.model.state_dict())
                if self.args.save:
                    if not os.path.exists(self.args.save):
                        os.makedirs(save_path)
                    save_path = os.path.join(self.args.save, 'compactst_1m.pt')
                    torch.save(best_state_dict, save_path)
                    print(datetime.datetime.now(), '*****************The best model is saved!*****************')
            else:
                wait += 1
                if wait >= self.args.early_stop:
                    break

        print("Pre-training is finished.")

    @torch.no_grad()
    def eval_model(self, epoch):
        self.model.eval()
        batch_loss_list = []
        for dataset_name, x_batch, y_batch in self.valset_loader:
            x_batch = x_batch.to(self.args.device)
            y_batch = y_batch.to(self.args.device)

            out_batch = self.model(x_batch)
            
            if not self.args.use_revin:
                out_batch = self.scaler_dict[dataset_name[0]].inverse_transform(out_batch)
                y_batch = self.scaler_dict[dataset_name[0]].inverse_transform(y_batch)
            
            loss_batch = self.loss(out_batch.float(), y_batch.float())

            batch_loss_list.append(loss_batch.item())
        
        val_loss = np.mean(batch_loss_list)
            
        print(datetime.datetime.now(), '**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    @torch.no_grad()
    def test_model(self, epoch):
        self.model.eval()
        y = []
        out = []

        for dataset_name, x_batch, y_batch in self.testset_loader:
            x_batch = x_batch.to(self.args.device)
            y_batch = y_batch.to(self.args.device)
            out_batch = self.model(x_batch)
            
            if not self.args.use_revin:
                out_batch = self.scaler_dict[dataset_name[0]].inverse_transform(out_batch)
                y_batch = self.scaler_dict[dataset_name[0]].inverse_transform(y_batch)
            
            out_batch = out_batch.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            out.append(out_batch)
            y.append(y_batch)
            
        out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
        y = np.vstack(y).squeeze()
        print('The number of test samples: ', out.shape, y.shape)
        
        rmse, mae, mape = cal_metrics(y, out, dataset_name)
        
        print(datetime.datetime.now(), '**********Test Epoch {}: RMSE: {:.6f} MAE: {:.6f} MAPE: {:.6f}'.format(epoch, rmse, mae, mape))
