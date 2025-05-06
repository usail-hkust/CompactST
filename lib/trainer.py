import os
import time
import copy
import torch
import datetime
import numpy as np
from tqdm import tqdm
from lib.metrics import *

class trainer(object):
    def __init__(self, args, model, trainset_loader, valset_loader, testset_loader, unseen_loader, scaler_dict, uscaler_dict, 
                 adj_dict, uadj_dict, loss, optimizer, scheduler):
        super(trainer, self).__init__()
        self.args = args
        self.model = model
        self.trainset_loader = trainset_loader
        self.valset_loader = valset_loader
        self.testset_loader = testset_loader
        self.unseen_loader = unseen_loader
        self.scaler_dict = scaler_dict
        self.uscaler_dict = uscaler_dict
        self.adj_dict = adj_dict
        self.uadj_dict = uadj_dict
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def train_one_epoch(self):
        self.model.train()
        batch_loss_list = []
        num_step = 0

        cnt = 0
        time_list = []
    
        for dataset_name, data_batch in self.trainset_loader:
            cnt = cnt+1
            x_batch, y_batch = data_batch
            x_batch = x_batch.to(self.args.device)
            y_batch = y_batch.to(self.args.device)
            adj = self.adj_dict[dataset_name].to(self.args.device)

            s1 = time.time()
            out_batch, _, _ = self.model(x_batch, adj)
            
            if not self.args.use_revin:
                out_batch = self.scaler_dict[dataset_name].inverse_transform(out_batch)

            if dataset_name=='pems_bay' or dataset_name=='metr_la':
                loss_batch = self.loss[0](out_batch, y_batch)
            else:
                loss_batch = self.loss[1](out_batch, y_batch)
            self.optimizer.zero_grad()
            loss_batch.backward()
            
            if self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            self.optimizer.step()

            s2 = time.time()
            time_list.append(s2-s1)
            
            batch_loss_list.append(loss_batch.item())
            num_step += 1
            if num_step % self.args.log_step == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(datetime.datetime.now(), "step:  "+str(num_step)+"  train loss is:  "+str(np.mean(batch_loss_list))+"  current_lr is:  "+str(current_lr))

        self.scheduler.step()
        train_loss = np.mean(batch_loss_list)

        print('Training time: ', np.mean(time_list), cnt)
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

            if val_loss < min_val_loss:
                wait = 0
                min_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.model.state_dict())
                if self.args.save_path:
                    # if not os.path.exists(self.args.save):
                    #     os.makedirs(save_path)
                    # save_path = os.path.join(self.args.save, 'compactst_1m.pt')
                    torch.save(best_state_dict, self.args.save_path.split('.')[0]+'_epoch '+str(epoch)+'.'+self.args.save_path.split('.')[1])
                    print(datetime.datetime.now(), '*****************The best model is saved!*****************')
            else:
                wait += 1
                if wait >= self.args.early_stop:
                    break

        print("Training is finished.")
        self.model.load_state_dict(best_state_dict)
        print(datetime.datetime.now(), '**********Early stopping at epoch ', epoch+1)
        print(datetime.datetime.now(), '**********Best at epoch ', best_epoch+1)
        self.test_model()
        
    @torch.no_grad()
    def eval_model(self, epoch):
        self.model.eval()
        batch_loss_list = []
        for dataset_name, val_data in self.valset_loader:
            for x_batch, y_batch in val_data:
                x_batch = x_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)
                adj = self.adj_dict[dataset_name].to(self.args.device)

                out_batch, _, _ = self.model(x_batch, adj)

                if not self.args.use_revin:
                    out_batch = self.scaler_dict[dataset_name].inverse_transform(out_batch)

                if dataset_name=='pems_bay' or dataset_name=='metr_la' or 'air' in dataset_name or 'aqi' in dataset_name:
                    loss_batch = self.loss[0](out_batch, y_batch)
                else:
                    loss_batch = self.loss[1](out_batch, y_batch)

                batch_loss_list.append(loss_batch.item())
        
        val_loss = np.mean(batch_loss_list)

        print(datetime.datetime.now(), '**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    @torch.no_grad()
    def test_model(self):
        self.model.eval()
        acc_dict = {}

        time_list = []
        cnt = 0
        for dataset_name, test_data in self.unseen_loader:
            y = []
            out = []
            for x_batch, y_batch in test_data:
                x_batch = x_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)
                adj = self.uadj_dict[dataset_name].to(self.args.device)

                cnt = cnt+1
                s1 = time.time()
                out_batch, _, _ = self.model(x_batch, adj)
                s2 = time.time()
                time_list.append(s2-s1)

                if not self.args.use_revin:
                    out_batch = self.uscaler_dict[dataset_name].inverse_transform(out_batch)

                out_batch = out_batch.cpu().numpy()
                y_batch = y_batch.cpu().numpy()
                out.append(out_batch)
                y.append(y_batch)

            out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
            y = np.vstack(y).squeeze()

            rmse, mae, mape = cal_metrics(y, out, dataset_name)
            acc_dict[dataset_name] = [rmse, mae, mape]

        print(datetime.datetime.now(), '**********Test accuracy: ', acc_dict, np.mean(time_list), cnt)
