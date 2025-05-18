# models/decoder.py

import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, cell_type='lstm', dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        rnn_cell = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[cell_type.lower()]
        self.rnn = rnn_cell(emb_dim, hidden_dim, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.cell_type = cell_type.lower()
        

    def forward(self, input_token, hidden):
        embedded = self.embedding(input_token.unsqueeze(1))  # (batch, 1, emb_dim)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output[:, -1, :]) 
        return prediction, hidden
