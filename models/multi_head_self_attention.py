import torch
import torch.nn as nn

class MHA(nn.Module):
  def __init__(self, embed_dim, num_heads, drop_rate, mask = False):
      super().__init__()
      self.W_query = nn.Linear(embed_dim, embed_dim, bias=False)
      self.W_key = nn.Linear(embed_dim, embed_dim, bias=False)
      self.W_value = nn.Linear(embed_dim, embed_dim, bias=False)
      self.out_proj = nn.Linear(embed_dim, embed_dim)  # Linear layer to combine head outputs
      self.drop = nn.Dropout(drop_rate)
      self.num_heads = num_heads
      self.embed_dim = embed_dim
      self.head_dim = embed_dim // num_heads
      assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
      self.masked = mask

  def forward(self, input):
      ## input: (batch_size, seq_length, embed_dim)
      ## output: (batch_size, seq_length, embed_dim)


      batch_size, seq_length, _ = input.shape

      K = self.W_key(input)    # (batch_size, seq_length, embed_dim)
      Q = self.W_query(input)  # (batch_size, seq_length, embed_dim)
      V = self.W_value(input)  # (batch_size, seq_length, embed_dim)

      K = K.view(batch_size, seq_length, self.num_heads, self.head_dim)  # (batch_size, seq_length, num_heads, head_dim)
      Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)  # (batch_size, seq_length, num_heads, head_dim)
      V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)  # (batch_size, seq_length, num_heads, head_dim)

      K = K.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
      Q = Q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
      V = V.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)

      d_k = torch.tensor(Q.shape[-1], dtype = torch.float32)

      attn_scores = Q@(K.transpose(2,3))  # K.transpose(2,3) : (batch_size, num_heads, seq_length, head_dim) 
      ##  attn_scores: : (batch_size, num_heads, seq_length, seq_length) 

      if self.masked:
        mask = torch.tril(torch.ones((seq_length, seq_length))).to(attn_scores.device)
        
        attn_scores = mask * attn_scores
        attn_scores.masked_fill_(attn_scores == 0, float('-inf')) # equal to attn_scores[ attn_scores== 0] = -torch.inf


      attn_weights  = torch.softmax(attn_scores / torch.sqrt(d_k), dim = -1)
      attn_weights = self.drop(attn_weights)
      output = attn_weights@V # (batch_size, num_heads, seq_length, head_dim) 
      output = output.permute(0, 2, 1, 3) # (batch_size, seq_length, num_heads, head_dim) 
      output = output.contiguous().view(batch_size, seq_length, self.embed_dim) # (batch_size, seq_length, embed_dim) 
      output = self.out_proj(output)  # (batch_size, seq_length, embed_dim) 

      return output