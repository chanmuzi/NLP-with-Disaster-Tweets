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

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
            
        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        
    @abstractmethod
    def _train_epoch(self,epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        """
        raise NotImplementedError
