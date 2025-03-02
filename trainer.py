import torch
import torch.nn as nn
import time
import numpy as np
import math
from pathlib import Path  

class Trainer:
    def __init__(self, tokenizer, train_dataloader, val_dataloader, model, config,  device, sample_context):
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.iter_train_dataloader = iter(train_dataloader)
        self.val_dataloader = val_dataloader
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.configure_optimizer()
        self.device = device
        self.grad_accum_steps = config.grad_accum_steps
        self.best_val_loss = float('inf')
        self.best_step = 0
        self.train_losses = []
        self.val_losses = []
        self.checkpoint_dir = Path(".") / 'ckpt'
        self.sample_context = sample_context
        self._prepare_checkpoint_dir()

    def _prepare_checkpoint_dir(self):
        """ Prepare the checkpoint directory. """
        try:
            self.checkpoint_dir.mkdir(parents=True)
            print(f"Checkpoint directory is ready at: {self.checkpoint_dir}")
        except FileExistsError:
            print(f"The checkpoint directory ({self.checkpoint_dir}) already exists.")

    def get_batch(self):
        """ Get a batch of data from the training dataloader. """
        try:  
            return next(self.iter_train_dataloader)
        except StopIteration: # If the iterator is exhausted, create a new one
            self.iter_train_dataloader = iter(self.train_dataloader)
            return next(self.iter_train_dataloader)
        
    def train_step(self, current_step):
        """ Perform a single training step. """
        loss_accum = 0
        # Zero the gradients
        self.optimizer.zero_grad()

        for _ in range(self.grad_accum_steps):
            X, y = self.get_batch() # X: (batch_size, seq_length)   y: (batch_size, seq_length)
            X, y = X.to(self.device), y.to(self.device)            
            
            # Forward pass
            with torch.autocast(device_type=str(self.device), dtype=torch.bfloat16): # Use bfloat16 for faster computation
                pred = self.model(X)     # (batch_size, seq_length, vocab_size)
            pred = pred.flatten(0,1)     # (batch_size x seq_length, vocab_size)
            y = y.flatten(0,1)           # (batch_size x seq_length)
    
            # Calculate loss
            loss = self.criterion(pred, y)
            loss = loss / self.grad_accum_steps
            loss_accum += loss.item()
            loss.backward()

        # Clip norm of gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
 
        # Update learning rate
        lr = self.get_lr(current_step)
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # Update weights
        self.optimizer.step()

        return loss_accum, lr


    def evaluate(self):
        """ Evaluate the model on the validation set. """
        losses= []
        for X, y in self.val_dataloader:
            X, y = X.to(self.device), y.to(self.device) # X: (bath_size, seq_length)   y: (bath_size, seq_length)
            
            # Forward pass
            with torch.no_grad(): # No need to track the gradients
                with torch.autocast(device_type=str(self.device), dtype=torch.bfloat16): # Use bfloat16 for faster computation
                    pred = self.model(X) # (batch_size, seq_length, vocab_size)
            pred = pred.flatten(0,1)     # (batch_size x seq_length, vocab_size)
            y = y.flatten(0,1)           # (batch_size x seq_length)
            loss = self.criterion(pred, y)
            losses.append(loss.item())
        
        val_loss = np.mean(losses)
        return val_loss


    def log_results(self, step, train_loss, val_loss, training_time, lr):
        """ Log the training results. """
        print(f'Step {step}: train loss {train_loss:.4f}, val loss {val_loss:.4f} | lr {lr:.4f} | {training_time:.2f}s ')

    def save_checkpoint(self, step, val_loss):
        """ Save the model checkpoint if the validation loss has improved. """
        if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_step = step
                model_filename = self.checkpoint_dir / f'best_model.pth'
                torch.save(self.model.state_dict(), model_filename)

    def print_sample_output(self):
        """ Generate and print a sample output from the model. """
        tokens = self.tokenizer.encode(self.sample_context, allowed_special={'<|endoftext|>'}) # list of indexes [3, 2, 1, ... ]
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
        print(f'> {decoded_text}')
    
    def configure_optimizer(self):
        """ Configure the optimizer. """                
        # Get all trainable parameters
        param_dict = {name: p for name, p in self.model.named_parameters() if  p.requires_grad}
        # Apply weight decay to all weights except biases and batch norm layers
        param_groups = [
            {'params': [p for name, p in param_dict.items() if p.dim() >= 2 ], 'weight_decay': self.config.weight_decay},  
            {'params': [p for name, p in param_dict.items() if p.dim() < 2 or not p.requires_grad], 'weight_decay': 0.0}  
        ]

        optmizer = torch.optim.AdamW(param_groups, lr = self.config.max_lr, betas=(0.9, 0.95), eps=1e-8)
        return optmizer
    
    def get_lr(self, current_step):
        """ Get the learning rate for the current step based on the learning rate schedule. """
        # (warup) if the step < warmup_steps then increase the learning rate linearly
        if current_step < self.config.warmup_steps:
            return   ((current_step + 1) / self.config.warmup_steps) * self.config.max_lr
        # (constant min LR) if the step > max_steps then decrease the learning rate linearly
        elif current_step > self.config.max_steps:
            return self.config.min_lr
        # (cosine decay) if the step is between warmup_steps and max_steps then decrease the learning rate using cosine decay
        decay_ratio = (current_step - self.config.warmup_steps)  / (self.config.max_steps - self.config.warmup_steps)
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))

        return self.config.min_lr + coeff * (self.config.max_lr - self.config.min_lr)
    
    def train(self):
        """ Train the model. """
        for step in range(self.config.max_steps):
            start_time = time.time()
            self.model.train()
            train_loss, lr = self.train_step(step)
            if step % 10 == 0: # Log the results every 10 steps
                self.model.eval()
                val_loss = self.evaluate()
                end_time = time.time()
                training_time = (end_time - start_time)
                self.log_results(step, train_loss, val_loss, training_time, lr)
                self.save_checkpoint(step, val_loss)
                self.print_sample_output()