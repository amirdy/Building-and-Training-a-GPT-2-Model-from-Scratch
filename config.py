from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size = 50257   # Vocabulary size
    context_length = 256 # Shortened context length (orig: 1024)
    emb_dim = 768        # Embedding dimension
    n_heads = 12         # Number of attention heads
    n_layers = 12        # Number of layers
    drop_rate = 0.1      # Dropout rate
    qkv_bias = False     # Query-key-value bias

@dataclass
class TrainingConfig:
    epochs= 10
    learning_rate = 0.0004  #5e-4,
    weight_decay = 0.1
    batch_size = 2 
    max_new_token = 50
    temperature = 1.1
    k_top = 3 
    save_path = "."