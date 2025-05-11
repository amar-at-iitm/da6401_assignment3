# data_preparation.py

import os
import urllib.request
import zipfile
import pandas as pd
import json
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# ---------- Config ----------
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "dakshina_dataset_v1.0.tar")
DATASET_DIR = os.path.join(DATA_DIR, "dakshina_dataset_v1.0")
LANG_CODE = "hi"
MAX_LEN = 30  # adjustable
SPECIAL_TOKENS = {"PAD": "<pad>", "SOS": "<sos>", "EOS": "<eos>"}

VOCAB_INPUT_PATH = os.path.join(DATA_DIR, "vocab_input.json")
VOCAB_OUTPUT_PATH = os.path.join(DATA_DIR, "vocab_output.json")


# ---------- Step 1: Download and unzip ----------
def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print("Downloading Dakshina dataset...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar",
            ZIP_PATH
        )
        print("Download complete.")
    else:
        print("Dataset ZIP already exists.")

    if not os.path.exists(DATASET_DIR):
        print("Unzipping dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Unzipping complete.")
    else:
        print("Dataset already extracted.")


# ---------- Step 2: Load .tsv ----------
# def read_tsv(split="train"):
#     path = os.path.join(DATASET_DIR, LANG_CODE, "lexicons", f"{LANG_CODE}.translit.sampled.{split}.tsv")
#     df = pd.read_csv(path, sep="\t", header=None, names=["roman", "devanagari"])
#     return df

def read_tsv(split="train"):
    path = os.path.join(DATASET_DIR, LANG_CODE, "lexicons", f"{LANG_CODE}.translit.sampled.{split}.tsv")
    df = pd.read_csv(path, sep="\t", header=None, names=["devanagari", "roman", "label"])
    df = df[["devanagari", "roman"]]
    df.dropna(subset=["roman", "devanagari"], inplace=True)
    df["roman"] = df["roman"].astype(str)
    df["devanagari"] = df["devanagari"].astype(str)
    return df


# ---------- Step 3: Vocab ----------
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
    idx2char = {int(idx): char for char, idx in char2idx.items()}
    char2idx = {char: int(idx) for char, idx in char2idx.items()}
    return char2idx, idx2char


# ---------- Step 4: Sequence processing ----------
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


# ---------- Step 5: Load all ----------
def load_data():
    download_and_extract()

    train_df = read_tsv("train")
    dev_df = read_tsv("dev")
    test_df = read_tsv("test")

    # Input vocab
    if os.path.exists(VOCAB_INPUT_PATH):
        input_char2idx, input_idx2char = load_vocab(VOCAB_INPUT_PATH)
    else:
        input_char2idx, input_idx2char = build_vocab(train_df["roman"])
        save_vocab(input_char2idx, VOCAB_INPUT_PATH)

    # Output vocab
    if os.path.exists(VOCAB_OUTPUT_PATH):
        output_char2idx, output_idx2char = load_vocab(VOCAB_OUTPUT_PATH)
    else:
        output_char2idx, output_idx2char = build_vocab(train_df["devanagari"], special_tokens=SPECIAL_TOKENS)
        save_vocab(output_char2idx, VOCAB_OUTPUT_PATH)

    # Process datasets
    x_train, y_train = process_dataset(train_df, input_char2idx, output_char2idx)
    x_val, y_val = process_dataset(dev_df, input_char2idx, output_char2idx)
    x_test, y_test = process_dataset(test_df, input_char2idx, output_char2idx)

    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
        "input_char2idx": input_char2idx,
        "input_idx2char": input_idx2char,
        "output_char2idx": output_char2idx,
        "output_idx2char": output_idx2char
    }

if __name__ == "__main__":
    data = load_data()
    print("Data loaded successfully.")
    print(f"Input vocab size: {len(data['input_char2idx'])}")
    print(f"Output vocab size: {len(data['output_char2idx'])}")
    print(f"Training set shape: {data['x_train'].shape}, {data['y_train'].shape}")
    print(f"Validation set shape: {data['x_val'].shape}, {data['y_val'].shape}")
    print(f"Test set shape: {data['x_test'].shape}, {data['y_test'].shape}")