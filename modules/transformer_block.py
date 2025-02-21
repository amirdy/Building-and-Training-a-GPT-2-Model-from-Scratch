import torch
import torch.nn as nn
from modules.layer_norm import LayerNorm
from modules.multi_head_self_attention import MHA
from modules.feed_forward import FeedForward

class transformer_block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.drop = nn.Dropout(config['drop_rate'])
    self.MHA = MHA(config['emb_dim'], config['n_heads'], config['drop_rate'] , mask = True)
    self.layer_nomalization_1 = LayerNorm(config['emb_dim'])
    self.layer_nomalization_2 = LayerNorm(config['emb_dim'])
    self.feed_forward = FeedForward(config)

  def forward(self, input):
    ## input : B x seq_length x embed_dim
    ## output : B x seq_length x embed_dim


    shortcut = input
    output = self.layer_nomalization_1(shortcut) #  output : B x seq_length x embed_dim
    output = self.MHA(output) #  output : B x seq_length x embed_dim
    output = self.drop(output)
    output = output + shortcut

    shortcut = output
    output = self.layer_nomalization_2(shortcut) #  output : B x seq_length x embed_dim
    output = self.feed_forward(output) #  output : B x seq_length x embed_dim
    output = self.drop(output)
    output = output + shortcut

    return output