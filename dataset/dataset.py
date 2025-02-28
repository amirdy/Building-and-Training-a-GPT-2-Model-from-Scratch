import torch

class Dataset(torch.utils.data.Dataset):
  def __init__(self, data, context_length):
    self.tokens = data
    self.context_length = context_length  
    self.context_samples = []
    self.target_samples = []
    for i in range(0, len(self.tokens) - (context_length)  - 1, context_length):
      context = self.tokens[i : i + context_length]
      target = self.tokens[i + 1 : i + context_length + 1]
      self.context_samples.append(context)
      self.target_samples.append(target)
      

  def __getitem__(self,index):
    X = torch.tensor(self.context_samples[index])
    y = torch.tensor(self.target_samples[index])
    return X, y
    # return self.context_samples[index], self.target_samples[index]


  def __len__(self):
    return len(self.context_samples)