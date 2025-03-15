from dataclasses import dataclass

@dataclass
class GPTConfig:
    """
    Configuration for GPT model architecture.
    
    Attributes:
        vocab_size: Vocabulary size used in the model.
        context_length: The maximum context window length.
        emb_dim: The embedding dimension size.
        n_heads: The number of attention heads in the multi-head attention layer.
        n_layers: The number of transformer layers.
        drop_rate: Dropout rate to prevent overfitting.
        qkv_bias: Whether to use bias in the QKV matrices.
        weight_tying: Whether to tie weights between input and output embedding layers.
    """
    vocab_size = 50257     
    context_length = 1024  
    emb_dim = 768          
    n_heads = 12           
    n_layers = 12          
    drop_rate = 0.1        
    qkv_bias = False       
    weight_tying = True    

@dataclass
class TrainingConfig:
    """
    Configuration for training hyperparameters.
    
    Attributes:
        max_steps: Total number of training steps.
        warmup_steps: Number of warmup steps for learning rate scheduler.
        max_lr: Maximum learning rate for the optimizer.
        min_lr: Minimum learning rate during training.
        weight_decay: Weight decay coefficient for optimizer.
        batch_size: Batch size for each training step.
        max_new_token: The number of new tokens to generate at each step.
        temperature: Sampling temperature for text generation.
        k_top: Top-k sampling to filter the most probable next tokens.
        grad_accum_steps: Number of gradient accumulation steps.
    """
    max_steps = 3000 
    warmup_steps =100 #715  
    max_lr = 6e-4   
    min_lr = 6e-5  # GPT-3 small   (0.1 * 6e-4)
    weight_decay = 0.1  
    batch_size = 64 
    max_new_token = 100
    temperature = 1  
    k_top = 50   
    grad_accum_steps = 8  