import torch
import torch.nn as nn
from modules.gelu import GELU

class FeedForward(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.fc1 = nn.Linear(config['emb_dim'], 4 * config['emb_dim'])
    self.GELU = GELU()
    self.fc2 = nn.Linear(4 * config['emb_dim'], config['emb_dim'])

  def forward(self, input):
    ## input : B x seq_length x embed_dim
    ## output : B x seq_length x embed_dim

    output = self.fc1(input)
    output = self.GELU(output)
    output = self.fc2(output)

    return output