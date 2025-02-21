import torch
import torch.nn as nn

class LayerNorm(nn.Module):
  def __init__(self, embed_dim,  eps = 1e-5):
    super().__init__()
    self.eps = eps
    self.scale = nn.Parameter(torch.ones(embed_dim))
    self.shift = nn.Parameter(torch.zeros(embed_dim))

    # self.scale = nn.Parameter(torch.ones(1, 1, embed_dim))
    # self.shift = nn.Parameter(torch.zeros(1, 1, embed_dim))
  def forward(self, input):
    ## input : B x seq_length x embed_dim
    ## output : B x seq_length x embed_dim

    mean = input.mean(dim = -1, keepdim = True)
    std = input.std(dim = -1, keepdim = True, unbiased=False)
    normalized = (input - mean) / (std + self.eps)

    return self.scale * normalized   +   self.shift
