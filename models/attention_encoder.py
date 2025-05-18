# models/attention_encoder.py
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, cell_type='lstm', dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        rnn_cell = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[cell_type.lower()]
        self.rnn = rnn_cell(emb_dim,hidden_dim,num_layers=n_layers,dropout=dropout if n_layers > 1 else 0,batch_first=True)
        self.cell_type = cell_type.lower()
        self.return_outputs = True  

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden 
