from torch.utils.data import DataLoader
from dataset.dataset import Dataset
import os
import urllib.request
import tiktoken

class DataModule():
    def __init__(self, batch_size, context_length):
        self.batch_size = batch_size
        file_path = "the-verdict.txt"
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

        if not os.path.exists(file_path):
            with urllib.request.urlopen(url) as response:
                text_data = response.read().decode('utf-8')
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_data)
        else:
            with open(file_path, "r", encoding="utf-8") as file:
                text_data = file.read()

        self.tokenizer = tiktoken.get_encoding("gpt2")
        split_idx = int(0.9 * len(text_data))

        training_set = text_data[0 : split_idx]
        validation_set = text_data[split_idx : ]
        training_set_tokens = self.tokenizer.encode(training_set, allowed_special={"<|endoftext|>"})
        validation_set_tokens = self.tokenizer.encode(validation_set, allowed_special={"<|endoftext|>"})    

        self.train_dataset = Dataset(training_set_tokens, context_length)
        self.val_dataset = Dataset(validation_set_tokens, context_length)



    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, drop_last = True, num_workers = 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False, drop_last = False, num_workers = 0)
