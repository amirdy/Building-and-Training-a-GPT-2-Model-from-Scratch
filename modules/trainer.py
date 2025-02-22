import torch
import torch.nn as nn
import numpy as np

class Trainer:
    def __init__(self, tokenizer, train_dataloader, val_dataloader, model, config,  device):
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.config_optimizer()
        self.device = device
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """ Train epoch """
        losses= []
        for batch, (X,y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device) 
            # X: (bath_size, seq_length)
            # y: (bath_size, seq_length)
            
            pred = self.model(X) # (batch_size, seq_length, vocab_size)
            pred = pred.flatten(0,1) # (batch_size x seq_length, vocab_size)
            y = y.flatten(0,1) # (batch_size x seq_length)

            loss = self.criterion(pred, y)
            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return np.mean(losses)


    def validate_epoch(self):
        """ Validate epoch """
        losses= []
        for batch, (X,y) in enumerate(self.val_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            # X: (bath_size, seq_length)
            # y: (bath_size, seq_length)

            with torch.no_grad():
                pred = self.model(X) # (batch_size, seq_length, vocab_size)
            pred = pred.flatten(0,1) # (batch_size x seq_length, vocab_size)
            y = y.flatten(0,1) # (batch_size x seq_length)
            loss = self.criterion(pred, y)
            losses.append(loss.item())
    
        return np.mean(losses)


    def log_results(self, epoch, train_loss, val_loss):
        """ Log results """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        print(f'Epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}')

    def save_checkpoint(self, epoch, val_loss):
        """ Save checkpoint """
        if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.config.save_path)

    def print_sample_output(self):
        """ Print sample output"""
        sample_context = "Every effort moves you"
        tokens = self.tokenizer.encode(sample_context, allowed_special={'<|endoftext|>'}) # list of indexes [3, 2, 1, ... ]

        for i in range(self.max_new_token):
            tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(tokens_tensor)

            last_seq_logits = logits[0,-1,:] # torch.Size([vocab_size])
            _, top_k_indices = torch.topk(last_seq_logits, self.config.k_top)
            indices = torch.arange(len(last_seq_logits))
            non_top_k_indices = list(set(indices.tolist()) - set(top_k_indices.tolist()))
            last_seq_logits[non_top_k_indices] = float('-inf')

            last_seq_logits = last_seq_logits / self.config.temperature
            probs = torch.softmax(last_seq_logits, dim = 0)
            next_id = torch.multinomial(probs, num_samples=1)
            tokens = tokens + [next_id.item()]

        decoded_text = self.tokenizer.decode(tokens) .replace("\n", " ")
        
        print(f'({decoded_text})')
    
    def train(self):
        """ Train """
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = self.train_epoch()
            self.model.eval()
            val_loss = self.validate_epoch()

            self.log_results(epoch, train_loss, val_loss)
            self.save_checkpoint(epoch, val_loss)
            self.print_sample_output()

    def config_optimizer(self):
        """ Config optimizer """
        optmizer = torch.optim.AdamW(self.model.parameters(), lr = self.config.learning_rate, weight_decay = self.config.weight_decay)
        return optmizer