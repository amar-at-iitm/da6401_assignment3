# models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        #self.v = nn.Parameter(torch.rand(hidden_size))
        self.v = nn.Parameter(torch.randn(hidden_size) * 0.1)

    def forward(self, hidden, encoder_outputs):
        if hidden.dim() == 3:  
            hidden = hidden[-1]  
        elif hidden.dim() == 1:  
            hidden = hidden.unsqueeze(0)  

        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_size)
        hidden_size = hidden.size(2)
        # Computing energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch, seq_len, hidden_size)

        v = self.v.unsqueeze(0).unsqueeze(1)  # (1, 1, hidden_size)
        v = v.repeat(batch_size, 1, 1)        # (batch, 1, hidden_size)

        #attention_weights = torch.bmm(v, energy.transpose(1, 2)).squeeze(1)  # (batch, seq_len)
        attention_weights = torch.bmm(v, energy.transpose(1, 2)).squeeze(1) / (hidden_size ** 0.5)  # Scale by sqrt(hidden_size)
        # Applying softmax to get normalized weights
        attn_weights = F.softmax(attention_weights, dim=1)  # (batch, seq_len)

        # Computing context vector as weighted sum of encoder_outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, hidden_size)

        return context, attn_weights
    
