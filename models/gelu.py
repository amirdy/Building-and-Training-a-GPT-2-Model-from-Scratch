import torch
import torch.nn as nn

class GELU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, input):
    return 0.5 * input * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * (input + 0.044715 * torch.pow(input, 3))))