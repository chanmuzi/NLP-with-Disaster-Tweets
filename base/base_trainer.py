from abc import abstractmethod
from numpy import inf
import torch

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self,model,criterion,metric,optimizer,config):
        self.config = config

        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer

        cfg_trainer = config.train
        self.epochs = cfg_trainer.max_epoch
            
        self.start_epoch = 1
        
    @abstractmethod
    def _train_epoch(self,epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        """
        raise NotImplementedError
