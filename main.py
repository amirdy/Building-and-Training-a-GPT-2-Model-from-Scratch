import torch
import torch.nn as nn
import tiktoken
from models.gpt import GPT
from datasets.dataset import Dataset
from datasets.data_module import DataModule
from config import GPTConfig, TrainingConfig
import os
import time
import urllib.request
from modules.trainer import Trainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


gpt_config = GPTConfig()
training_config = TrainingConfig()


dm = DataModule(training_config.batch_size, gpt_config.context_length)
train_dataloader = dm.train_dataloader()
val_dataloader = dm.val_dataloader()



model = GPT(gpt_config)
model.to(device)


trainer = Trainer(
    tokenizer = dm.tokenizer,
    train_dataloader = train_dataloader,
    val_dataloader = val_dataloader,
    model = model,
    config = training_config,
    device = device,
)
# start_time = time.time()
# trainer = trainer.train()
# end_time = time.time()