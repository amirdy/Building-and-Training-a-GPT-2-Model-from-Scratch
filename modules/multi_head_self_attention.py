import torch
import torch.nn as nn

class multi_head_self_attention(nn.Module):
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
      ## input : B x seq_length x embed_dim
      ## output: B x seq_length x embed_dim


      B, seq_length, _ = input.shape

      K = self.W_key(input)    # B x seq_length x embed_dim
      Q = self.W_query(input)  # B x seq_length x embed_dim
      V = self.W_value(input)  # B x seq_length x embed_dim
      # print("K", K.shape)
      # print("Q", Q.shape)
      # print("V", V.shape)
      K = K.view(B, seq_length, self.num_heads, self.head_dim)  # B x seq_length x num_heads x head_dim
      Q = Q.view(B, seq_length, self.num_heads, self.head_dim)  # B x seq_length x num_heads x head_dim
      V = V.view(B, seq_length, self.num_heads, self.head_dim)  # B x seq_length x num_heads x head_dim

      K = K.permute(0, 2, 1, 3)    # B x num_heads x seq_length x head_dim
      Q = Q.permute(0, 2, 1, 3)  # B x num_heads x seq_length x head_dim
      V = V.permute(0, 2, 1, 3)  # B x num_heads x seq_length x head_dim

      # print("K", K.shape)

      d_k = torch.tensor(Q.shape[-1], dtype = torch.float32)

      attn_scores = Q@ (K.transpose(2,3)) #   K.transpose(2,3) : B x num_heads x head_dim x seq_length |     attn_scores: : B x num_heads x seq_length x seq_length

      if self.masked:
        mask = torch.tril(torch.ones((seq_length, seq_length))).to(attn_scores.device)
        
        attn_scores = mask * attn_scores
        attn_scores[ attn_scores== 0] = -torch.inf


      attn_weights  = torch.softmax(attn_scores / torch.sqrt(d_k), dim = -1)
      attn_weights = self.drop(attn_weights)
      output = attn_weights@V # B x num_heads x seq_length x head_dim
      output = output.permute(0, 2, 1, 3) # B x seq_length x num_heads x head_dim
      output = output.contiguous().view(B, seq_length, self.embed_dim) # B x seq_length x embed_dim
      output = self.out_proj(output)  # B x seq_length x embed_dim

      return output