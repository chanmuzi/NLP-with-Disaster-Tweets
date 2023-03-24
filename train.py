import numpy as np
import pandas as pd
import torch
import argparse
from omegaconf import OmegaConf

from utils import prepare_device

from base.base_data_loader import BaseDataLoader as dataloader
import model.loss as Criterion
from model.metric import accuracy as Metric
from model.model import CEModel as Model
import trainer as Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(config):
    model = AutoModelForSequenceClassification.from_pretrained(config.model.model_name)
    device, _ = prepare_device(config['n_gpu'])
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    train_data = pd.read_csv(config.data.train_path + 'train.csv')

    train_loader,valid_loader = dataloader(
        dataset=train_data, 
        tokenizer=tokenizer, 
        batch_size=config.train.batch_size, 
        is_test=False)
    criterion = getattr(Criterion, config.model.loss)
    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate)
    metric = Metric
    epochs = config.train.max_epoch

    trainer = getattr(Trainer, config.model.trainner_class)(
        model = model,
        device = device,
        criterion = criterion,
        metric_ftn = metric,
        optimizer = optimizer,
        config = config,
        train_dataloader = train_loader,
        valid_dataloader = valid_loader,
        epochs = epochs,
        lr_scheduler = None,
    )
    trainer.train()

if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description='chanmuzi Template')
    parser.add_argument('--config', type=str, default='config')
    args, _ = parser.parse_known_args()

    config = OmegaConf.load('config.yaml')
    main(config)