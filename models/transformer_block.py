import torch.nn as nn
from models.layer_norm import LayerNorm
from models.multi_head_self_attention import MHA
from models.feed_forward import FeedForward

class TransformerBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.drop = nn.Dropout(config.drop_rate)
    self.MHA = MHA(config.emb_dim, config.n_heads, config.drop_rate , mask = True)
    self.layer_nomalization_1 = LayerNorm(config.emb_dim)
    self.layer_nomalization_2 = LayerNorm(config.emb_dim)
    self.feed_forward = FeedForward(config.emb_dim)

  def forward(self, input):
    ## input:  (batch_size, seq_length, embed_dim)
    ## output: (batch_size, seq_length, embed_dim)


    shortcut = input
    output = self.layer_nomalization_1(shortcut) # (batch_size, seq_length, embed_dim)
    output = self.MHA(output)                    # (batch_size, seq_length, embed_dim)
    output = self.drop(output)
    output = output + shortcut

    shortcut = output
    output = self.layer_nomalization_2(shortcut) # (batch_size, seq_length, embed_dim)
    output = self.feed_forward(output)           # (batch_size, seq_length, embed_dim)
    output = self.drop(output)
    output = output + shortcut

    return output