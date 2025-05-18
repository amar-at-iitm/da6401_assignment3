# models/attention_seq2seq.py
import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, sos_token_id=None, eos_token_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.rnn_type = encoder.cell_type
        
    def _adjust_hidden(self, hidden):
        def pad_or_trim(h, target_layers):
            if h.size(0) == target_layers:
                return h
            elif h.size(0) < target_layers:
                repeat = target_layers - h.size(0)
                return torch.cat([h] + [h[-1:]] * repeat, dim=0)
            else:
                return h[:target_layers]

        if self.rnn_type == 'lstm':
            h_n, c_n = hidden
            h_n = pad_or_trim(h_n, self.decoder.rnn.num_layers)
            c_n = pad_or_trim(c_n, self.decoder.rnn.num_layers)
            return (h_n, c_n)
        else:
            h_n = hidden
            h_n = pad_or_trim(h_n, self.decoder.rnn.num_layers)
            return h_n

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        output_dim = self.decoder.output_dim

        # Store outputs and attention weights
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)
        attentions = torch.zeros(batch_size, trg_len, src.size(1)).to(self.device)

        # Encode input
        encoder_outputs, hidden = self.encoder(src)
        hidden = self._adjust_hidden(hidden)
      

        # Prepare initial decoder input (SOS tokens)
        input_token = torch.full((batch_size,), self.sos_token_id, dtype=torch.long, device=self.device)

        for t in range(trg_len):
            output, hidden, attn_weights = self.decoder(input_token, hidden, encoder_outputs)

            outputs[:, t] = output
            attentions[:, t] = attn_weights  # [batch, src_len]

            # Choose next input
            if trg is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = trg[:, t]
            else:
                input_token = output.argmax(1)

        return outputs, attentions  # Return both predictions and attention matrix

    def predict(self, src, max_len=50, beam_size=1):
        self.eval()
        with torch.no_grad():
            if beam_size > 1:
                return [self.beam_search_decode(src[i:i+1], beam_size=beam_size, max_len=max_len) for i in range(src.size(0))]
            else:
                batch_size = src.size(0)
                hidden = self.encoder(src)
                hidden = self._adjust_hidden(hidden)

                input_token = torch.full((batch_size,), self.sos_token_id, dtype=torch.long, device=src.device)
                outputs = []

                for _ in range(max_len):
                    output, hidden = self.decoder(input_token, hidden)
                    top1 = output.argmax(1)
                    outputs.append(top1)
                    input_token = top1

                    if (top1 == self.eos_token_id).all():
                        break

                outputs = torch.stack(outputs, dim=1)
                return outputs.tolist()

    def greedy_decode(self, src, max_len=30):
        self.eval()
        with torch.no_grad():
            hidden = self.encoder(src)
            hidden = self._adjust_hidden(hidden)

            input_token = torch.tensor([self.sos_token_id]).to(self.device)
            output_tokens = []

            for _ in range(max_len):
                output, hidden = self.decoder(input_token, hidden)
                top1 = output.argmax(1)
                if top1.item() == self.eos_token_id:
                    break
                output_tokens.append(top1.item())
                input_token = top1

        return output_tokens

    def beam_search_decode(self, src, beam_size=3, max_len=30):
        self.eval()
        with torch.no_grad():
            hidden = self.encoder(src)
            hidden = self._adjust_hidden(hidden)

            assert src.size(0) == 1, "Beam search supports batch size 1 only"

            sequences = [[
                [self.sos_token_id], 0.0, hidden
            ]]

            for _ in range(max_len):
                all_candidates = []
                for seq, score, h in sequences:
                    input_token = torch.tensor([seq[-1]]).to(self.device)
                    output, new_hidden = self.decoder(input_token, h)
                    log_probs = torch.log_softmax(output, dim=1).squeeze(0)

                    topk = torch.topk(log_probs, beam_size)

                    for i in range(beam_size):
                        token = topk.indices[i].item()
                        prob = topk.values[i].item()
                        candidate = [seq + [token], score + prob, new_hidden]
                        all_candidates.append(candidate)

                ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
                sequences = ordered[:beam_size]

                if all(seq[0][-1] == self.eos_token_id for seq in sequences):
                    break

            best_sequence = sequences[0][0]
            return best_sequence[1:] 