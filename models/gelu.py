import torch
import torch.nn as nn

class GELU(nn.Module):
  """ Gaussian Error Linear Unit (GELU) activation function implementation. """

  def __init__(self):
    """ Initializes the GELU activation function. """
    super().__init__()

  def forward(self, input):
    return 0.5 * input * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * (input + 0.044715 * torch.pow(input, 3))))