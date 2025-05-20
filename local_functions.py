# local_functions.py

import random
import numpy as np
import json
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


# ---------- Step: Vocab ----------
def build_vocab(sequences, special_tokens=None):
    sequences = [seq for seq in sequences if isinstance(seq, str)]
    counter = Counter(char for seq in sequences for char in seq)
    chars = sorted(counter)
    if special_tokens:
        chars = list(special_tokens.values()) + chars
    char2idx = {char: idx for idx, char in enumerate(chars)}
    idx2char = {idx: char for char, idx in char2idx.items()}
    return char2idx, idx2char

def save_vocab(char2idx, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(char2idx, f, ensure_ascii=False, indent=2)


def load_vocab(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        char2idx = json.load(f)
    # Ensures all values are integers (keys are already strings like "<pad>")
    char2idx = {char: int(idx) for char, idx in char2idx.items()}
    idx2char = {idx: char for char, idx in char2idx.items()}
    return char2idx, idx2char

# ---------- Step 4: Sequence processing ----------

MAX_LEN = 30  # adjustable
SPECIAL_TOKENS = {"PAD": "<pad>", "SOS": "<sos>", "EOS": "<eos>"}

def encode_sequence(seq, char2idx, add_sos_eos=False):
    if add_sos_eos:
        seq = [SPECIAL_TOKENS["SOS"]] + list(seq) + [SPECIAL_TOKENS["EOS"]]
    else:
        seq = list(seq)
    return [char2idx[char] for char in seq if char in char2idx]


def process_dataset(df, input_char2idx, output_char2idx):
    input_pad_idx = input_char2idx.get(SPECIAL_TOKENS["PAD"], 0)
    output_pad_idx = output_char2idx.get(SPECIAL_TOKENS["PAD"], 0)

    inputs = [torch.tensor(encode_sequence(seq, input_char2idx), dtype=torch.long) for seq in df["roman"]]
    targets = [torch.tensor(encode_sequence(seq, output_char2idx, add_sos_eos=True), dtype=torch.long) for seq in df["devanagari"]]

    x = pad_sequence(inputs, batch_first=True, padding_value=input_pad_idx)
    y = pad_sequence(targets, batch_first=True, padding_value=output_pad_idx)

    # Truncate or pad to fixed MAX_LEN
    if x.size(1) < MAX_LEN:
        x = F.pad(x, (0, MAX_LEN - x.size(1)), value=input_pad_idx)
    else:
        x = x[:, :MAX_LEN]

    if y.size(1) < MAX_LEN + 2:
        y = F.pad(y, (0, MAX_LEN + 2 - y.size(1)), value=output_pad_idx)
    else:
        y = y[:, :MAX_LEN + 2]

    return x, y

#///////////////////////////////////////////////////////////////// 
def idx_to_string(indices, idx2char):
    chars = []
    for idx in indices:
        idx = idx.item() if isinstance(idx, torch.Tensor) else idx
        char = idx2char[idx]
        if char == SPECIAL_TOKENS["EOS"]:
            break
        if char not in [SPECIAL_TOKENS["PAD"], SPECIAL_TOKENS["SOS"]]:
            chars.append(char)
    return "".join(chars)

#/////////////////////////////////////////////////////////////////////
def save_best_model_config(config, model_path="best_model.pt", output_path="best_model.py"):
    with open(output_path, "w") as f:
        f.write("# best_model.py\n\n")
        f.write(f"EMBEDDING_DIM = {config.embedding_dim}\n")
        f.write(f"HIDDEN_DIM = {config.hidden_dim}\n")
        f.write(f"ENCODER_LAYERS = {config.encoder_layers}\n")
        f.write(f"DECODER_LAYERS = {config.decoder_layers}\n")
        f.write(f"RNN_TYPE = \"{config.rnn_type}\"\n")
        f.write(f"DROPOUT = {config.dropout}\n")
        f.write(f"BEAM_SIZE = {getattr(config, 'beam_size', 1)}\n")
        f.write(f"MODEL_PATH = \"{model_path}\"\n")

def save_best_model_attention_config(config, model_path="best_model_attention.pt", output_path="best_model_attention.py"):
    with open(output_path, "w") as f:
        f.write("# best_model_attention.py\n\n")
        f.write(f"EMBEDDING_DIM = {config.embedding_dim}\n")
        f.write(f"HIDDEN_DIM = {config.hidden_dim}\n")
        f.write(f"ENCODER_LAYERS = {config.encoder_layers}\n")
        f.write(f"DECODER_LAYERS = {config.decoder_layers}\n")
        f.write(f"RNN_TYPE = \"{config.rnn_type}\"\n")
        f.write(f"DROPOUT = {config.dropout}\n")
        f.write(f"BEAM_SIZE = {getattr(config, 'beam_size', 1)}\n")
        f.write(f"MODEL_PATH = \"{model_path}\"\n")

#//////////////////////////////////////////////////////////////
def calculate_accuracy(preds, targets, pad_idx):
    _, pred_tokens = preds.max(2)  # shape: (batch, seq_len)
    mask = targets != pad_idx
    correct = ((pred_tokens == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0

#////////////////////////////////////////////////////////////////
def seed_all(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

