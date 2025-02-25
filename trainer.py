import torch
import torch.nn as nn
import numpy as np
import math
from pathlib import Path  

class Trainer:
    def __init__(self, tokenizer, train_dataloader, val_dataloader, model, config,  device):
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.configure_optimizer()
        self.device = device
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.checkpoint_dir = Path(self.config.save_path) / 'ckpt'
        try:
            self.checkpoint_dir.mkdir(parents=True)
            print(f"Checkpoint directory is ready at: {self.checkpoint_dir}")

        except FileExistsError:
            print(f"The checkpoint directory ({self.checkpoint_dir}) already exists.")

    def train_epoch(self, epoch):
        """ Train epoch """
        losses= []
        for batch, (X,y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device) 
            # X: (batch_size, seq_length)
            # y: (batch_size, seq_length)
            
            # Forward pass
            pred = self.model(X) # (batch_size, seq_length, vocab_size)
            pred = pred.flatten(0,1) # (batch_size x seq_length, vocab_size)
            y = y.flatten(0,1) # (batch_size x seq_length)
    
            # Calculate loss
            loss = self.criterion(pred, y)
            losses.append(loss.item())
 
            # Learning rate update
            current_step = (epoch - 1) * len(self.train_dataloader)  +  (batch + 1)
            lr = self.get_lr(current_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Backward pass and optimization
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
                model_filename = self.checkpoint_dir / f'best_model_epoch{epoch}.pth'
                torch.save(self.model.state_dict(), model_filename)

    def print_sample_output(self):
        """ Print sample output"""
        sample_context = "Every effort moves you"
        tokens = self.tokenizer.encode(sample_context, allowed_special={'<|endoftext|>'}) # list of indexes [3, 2, 1, ... ]

        for i in range(self.config.max_new_token):
            tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Generate logits from the model for the current token sequence
                logits = self.model(tokens_tensor)
                # Extract logits corresponding to the last token in the sequence (shape: [vocab_size])
                last_seq_logits = logits[0, -1, :]
                # Select the indices of the top-k highest logits
                _, top_k_indices = torch.topk(last_seq_logits, self.config.k_top)
                # Create a boolean mask where top-k logits are True, others are False
                mask = torch.zeros_like(last_seq_logits, dtype=torch.bool)
                mask[top_k_indices] = True
                # Set logits outside the top-k to negative infinity to exclude them from sampling
                last_seq_logits[~mask] = float('-inf')
                # Scale logits using temperature to control randomness in sampling
                scaled_logits = last_seq_logits / self.config.temperature
                # Convert scaled logits to probabilities using softmax
                probs = torch.softmax(scaled_logits, dim=0)
                # Sample the next token based on the probability distribution
                next_token = torch.multinomial(probs, num_samples=1)
                # Append the sampled token to the sequence
                tokens = tokens + [next_token.item()]

        decoded_text = self.tokenizer.decode(tokens) .replace("\n", " ")
        
        print(f'> ({decoded_text})')
    
    def train(self):
        """ Train """

        num_steps = len(self.train_dataloader)  # num_steps per epoch
        epochs = self.config.max_steps // num_steps
        print(f' Number of epochs is {epochs} which contains {epochs * (num_steps)} steps.')
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = self.train_epoch(epoch)
            self.model.eval()
            val_loss = self.validate_epoch()

            self.log_results(epoch, train_loss, val_loss)
            self.save_checkpoint(epoch, val_loss)
            self.print_sample_output()

    def configure_optimizer(self):
        """ Configure optimizer """
                
        # Get all trainable parameters
        param_dict = {name: p for name, p in self.model.named_parameters() if  p.requires_grad}
        # Apply weight decay to all weights except biases and batch norm layers
        param_groups = [
            {'params': [p for name, p in param_dict if p.dim()  ], 'weight_decay': self.config.weight_decay},  
            {'params': [p for name, p in param_dict if p.dim() == 0 or p.requires_grad], 'weight_decay': 0.0}  
        ]

        optmizer = torch.optim.AdamW(param_groups, lr = self.config.learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optmizer
    
    def get_lr(self, current_step):
        """ Get the learning rate"""

        # Warm up
        if current_step < self.config.warmup_steps:
            return   ((current_step + 1) / self.config.warmup_steps) * self.config.max_lr
        # if the step > max_steps then keep the min learning rate 
        elif current_step > self.config.max_steps:
            return self.config.min_lr
        # use cosine decay down to min learning rate
        decay_ratio = (current_step - self.config.warmup_steps)  / (self.config.max_steps - self.config.warmup_steps)
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))

        return self.config.min_lr + coeff * (self.config.max_lr - self.config.min_lr)
    