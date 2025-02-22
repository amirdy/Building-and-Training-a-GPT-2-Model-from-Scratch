import torch
import torch.nn as nn
from modules.layer_norm import LayerNorm
from modules.transformer_block import TrnasformerBlock

class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim)
    self.pos_embedding = nn.Embedding(config.context_length, config.emb_dim)
    self.drop_emb = nn.Dropout(config.drop_rate)
    self.trasnformer_blocks = nn.Sequential( *[TrnasformerBlock(config) for _ in range(config.n_layers)] )
    self.final_normalizatoin = LayerNorm(config.emb_dim)
    self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias = False)

  def forward(self, input):
    ## input is indexes of shape (batch_size, context_length)
    
    batch_size, context_length = input.shape

    token_embeds = self.token_embedding(input)
    pos_embeds = self.pos_embedding(torch.arange(context_length, device = input.device))
    x = token_embeds + pos_embeds

    x = self.drop_emb(x) # (batch_size, context_length, embed_dim)

    x = self.trasnformer_blocks(x) # (batch_size, context_length, embed_dim)

    x = self.final_normalizatoin(x) # (batch_size, context_length, embed_dim)
    logits = self.out_head(x) # (batch_size, context_length, vocab_size)
    
    return logits