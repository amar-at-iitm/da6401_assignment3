# models/attention_decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import Attention

class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, cell_type, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cell_type = cell_type.lower()  
        self.embedding = nn.Embedding(output_dim, emb_dim)
        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[cell_type.lower()]
        self.rnn = rnn_cls(emb_dim + hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.attention = attention
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, input_token, hidden, encoder_outputs):
        # input_token: [batch]
        embedded = self.dropout(self.embedding(input_token)).unsqueeze(1)  # [batch, 1, emb_dim]

        if self.cell_type == "lstm":
            decoder_hidden = hidden[0][-1]  # [batch, hidden]
        else:
            decoder_hidden = hidden[-1]  # [batch, hidden]


        context, attn_weights = self.attention(decoder_hidden, encoder_outputs)  

        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)  # [batch, 1, emb+hidden]
        output, hidden = self.rnn(rnn_input, hidden)

        output = output.squeeze(1)  # [batch, hidden]
        output = self.fc_out(torch.cat((output, context), dim=1))  # [batch, output_dim]

        return output, hidden, attn_weights

