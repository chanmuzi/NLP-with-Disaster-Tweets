import numpy as np
import pandas as pd
import torch
import argparse, os
from omegaconf import OmegaConf

from utils import prepare_device
from tqdm import tqdm

from base.base_data_loader import BaseDataLoader as dataloader
from model.model import CEModel as Model

from transformers import AutoTokenizer

def main(config):
    model = Model(config.model.model_name)

    best_pt = [pt for pt in os.listdir('save/') if 'best' in pt][0]
    check_point = torch.load(f'save/{best_pt}')
    model.load_state_dict(check_point)

    device, _ = prepare_device(config['n_gpu'])
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    test_data = pd.read_csv(config.data.test_path + 'test.csv')

    test_loader = dataloader(
        dataset=test_data, 
        tokenizer=tokenizer, 
        batch_size=config.train.batch_size, 
        is_test=True )
    
    model.eval()
    preds = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Inferencing...')
        for idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)

            preds.extend(predictions.detach().cpu().numpy())
    
    sample = pd.read_csv(config.data.test_path + 'sample_submission.csv')
    sample['label'] = preds
    sample.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config')
    args, _ = parser.parse_known_args()

    config = OmegaConf.load('config.yaml')
    main(config)