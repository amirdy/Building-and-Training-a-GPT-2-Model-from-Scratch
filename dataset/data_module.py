from torch.utils.data import DataLoader
from dataset.dataset import Dataset
import tiktoken

class DataModule():
    """ Data module for handling training and validation datasets with tokenized inputs. """

    def __init__(self, batch_size, context_length, training_set_tokens, validation_set_tokens):
        """Initializes the DataModule with tokenized datasets.

        Args:
            batch_size (int): Batch size for DataLoader.
            context_length (int): Context length for token sequences.
            training_set_tokens (Any): Tokenized training data.
            validation_set_tokens (Any): Tokenized validation data.
        """
        self.batch_size = batch_size
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.train_dataset = Dataset(training_set_tokens, context_length)
        self.val_dataset = Dataset(validation_set_tokens, context_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, drop_last = True, num_workers = 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False, drop_last = False, num_workers = 0)
