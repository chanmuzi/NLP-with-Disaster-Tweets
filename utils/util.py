import json
import torch
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True,exist_ok=False)

def prepare_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(f"Warning: There\'s no GPU available on this machine,"
               "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
                "available on this machine")
        n_gpu_use = n_gpu
    device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device,list_ids
    