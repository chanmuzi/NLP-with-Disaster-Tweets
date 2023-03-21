from tqdm import tqdm
import time
import gc, os
import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker

class Trainer(BaseTrainer):
    def __init__(self,model,device,criterion,metric_ftn,optimizer,config,
                train_dataloader,valid_dataloader,epochs,lr_scheduler=None):
        super().init__(model,criterion,metric_ftn,optimizer,config)
        self.config = config
        self.device = device
        self.epochs = epochs

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.lr_scheduler = lr_scheduler
        self.metric_ftn = metric_ftn

        self.valid_loss_list = [2]
        self.more_train = True
        self.best_models = [2]

    def train(self):
        for epoch in range(self.epochs):
            if self.more_train == True:
                start_time = time.time()
                self._train_epoch(epoch)
                self._valid_epoch(epoch)
                print(f'Epoch {epoch+1}/{self.epochs} runtime : {(time.time()-start_time)/60}')
            else:
                break
        
        print('Training/Validation compelete!!')

        self.select_best_model()
        print('Best model.pt checked!!')

        torch.cuda.empty_cache()
        del self.model, self.train_dataloader, self.valid_dataloader
        gc.collect()

    def _train_epoch(self,epoch):

        gc.collect()
        self.model.train()

        train_loss = 0
        train_step = 0
        pbar = tqdm(self.train_dataloader,desc=f'Epoch [{epoch+1}/{self.epochs}] Training..')
        for idx,batch in enumerate(pbar):
            self.optimizer.zero_grad()
            train_step += 1

            train_input_ids = batch['input_ids'].to(self.device)
            train_attention_mask = batch['attention_mask'].to(self.device)
            train_labels = batch['labels'].to(self.device)

            logits = self.model(train_input_ids,train_attention_mask)
            loss = self.criterion(logits,train_labels)

            loss.backward()
            self.optimizer.step()
            self.model.zero_grad()

            train_loss += loss.detach().cpu().numpy().item()

        pbar.set_postfix({'train_loss':train_loss/train_step})
        pbar.close()
        print(f'Epoch [{epoch+1}/{self.epochs}] Train_loss : {train_loss/train_step}')

    def _valid_epoch(self,epoch):
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0
            valid_step = 0

            y_pred,y_true = [],[]
            pbar = tqdm(self.valid_dataloader, desc=f'Epoch [{epoch+1}/{self.epochs}] Validatoin..')
            for idx, batch in enumerate(pbar):
                valid_step += 1

                valid_input_ids = batch['input_ids'].to(self.device)
                valid_attention_mask = batch['attention_mask'].to(self.device)
                valid_labels = batch['labels'].to(self.device)

                logits = self.model(valid_input_ids, valid_attention_mask)
                predictions = torch.argmax(logits, dim=-1)

                loss = self.criterion(logits, valid_labels)
                valid_loss += loss.detach().cpu().numpy().item()

                y_pred.extend(predictions.detach().cpu().numpy())
                y_true.extend(valid_labels.detach().cpu().numpy())
            
            valid_loss /= valid_step
            score = self.metric_ftn(y_true,y_pred)

            if valid_loss < self.valid_loss_list[-1]:
                print('model improved')
            else:
                self.more_train = False
                print('model depressed')
            self.valid_loss_list.append(valid_loss)

            print(f'Epoch [{epoch+1}/{self.epochs}] Score: {score}')
            print(f'Epoch [{epoch+1}/{self.epochs}] Valid_loss: {valid_loss}')

            torch.save(self.model.state_dict(), f'save/{epoch}epoch_model.pt')
            self.best_models.append(f'save/{epoch}epoch_model.pt')
    
    def select_best_model(self):
        best_model = self.best_models[np.array(self.valid_loss_list).argmin()]
        os.rename(best_model, best_model.split('.pt')[0] + '_best.pt')