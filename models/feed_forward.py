import torch
import torch.nn as nn
from models.gelu import GELU

class FeedForward(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()

    self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
    self.GELU = GELU()
    self.fc2 = nn.Linear(4 * embed_dim, embed_dim)

  def forward(self, input):
    ## input: (batch_size, seq_length, embed_dim)
    ## output : (batch_size, seq_length, embed_dim)

    output = self.fc1(input)
    output = self.GELU(output)
    output = self.fc2(output)

    return output