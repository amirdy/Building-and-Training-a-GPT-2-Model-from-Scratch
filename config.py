from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size = 50257    # Vocabulary size
    context_length = 1024 # Context length
    emb_dim = 768         # Embedding dimension
    n_heads = 12          # Number of attention heads
    n_layers = 12         # Number of layers
    drop_rate = 0.1       # Dropout rate
    qkv_bias = False      # Query-key-value bias
    weight_tying = False  # Weight tying

@dataclass
class TrainingConfig:
    max_steps = 19073 # Total number of training steps
    warmup_steps = 715 # Warmup steps
    max_lr = 6e-4  # GPT-3 small 
    min_lr = 6e-5  # GPT-3 small   (0.1 * 6e-4)
    weight_decay = 0.1 # Weight decay
    batch_size = 2 
    max_new_token = 50
    temperature = 1
    k_top = 50 
    save_path = "."