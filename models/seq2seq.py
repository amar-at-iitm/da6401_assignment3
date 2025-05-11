# models/seq2seq.py

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, cell_type='lstm', dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        rnn_cell = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[cell_type.lower()]
        self.rnn = rnn_cell(emb_dim, hidden_dim, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.cell_type = cell_type.lower()

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, cell_type='lstm', dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(output_dim, emb_dim)
        rnn_cell = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[cell_type.lower()]
        self.rnn = rnn_cell(emb_dim, hidden_dim, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.cell_type = cell_type.lower()

    def forward(self, input_token, hidden):
        embedded = self.embedding(input_token.unsqueeze(1))  # shape: (batch, 1, emb_dim)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))  # shape: (batch, output_dim)
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.cell_type == decoder.cell_type, "Encoder and Decoder must use the same RNN type"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        output_dim = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)

        hidden = self.encoder(src)
        input_token = trg[:, 0]  # first input is <sos>

        for t in range(1, trg_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input_token = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs
