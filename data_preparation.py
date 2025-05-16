# data_preparation.py
import os
import urllib.request
import zipfile
import pandas as pd
from local_functions import build_vocab, save_vocab, load_vocab, SPECIAL_TOKENS, process_dataset

# ---------- Config ----------
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "dakshina_dataset_v1.0.tar")
DATASET_DIR = os.path.join(DATA_DIR, "dakshina_dataset_v1.0")
LANG_CODE = "hi"

VOCAB_INPUT_PATH = os.path.join(DATA_DIR, "vocab_input.json")
VOCAB_OUTPUT_PATH = os.path.join(DATA_DIR, "vocab_output.json")


# ---------- Step: Download and unzip ----------
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


# ---------- Step: Load .tsv ----------   

def read_tsv(split="train"):
    path = os.path.join(DATASET_DIR, LANG_CODE, "lexicons", f"{LANG_CODE}.translit.sampled.{split}.tsv")
    df = pd.read_csv(path, sep="\t", header=None, names=["devanagari", "roman", "label"])
    df = df[["devanagari", "roman"]]
    df.dropna(subset=["roman", "devanagari"], inplace=True)
    df["roman"] = df["roman"].astype(str)
    df["devanagari"] = df["devanagari"].astype(str)
    return df


# ---------- Step: Load all ----------
def load_data():

    train_df = read_tsv("train")
    dev_df = read_tsv("dev")
    test_df = read_tsv("test")

    # Input vocab
    if os.path.exists(VOCAB_INPUT_PATH):
        input_char2idx, input_idx2char = load_vocab(VOCAB_INPUT_PATH)
    else:
        input_char2idx, input_idx2char = build_vocab(train_df["roman"], special_tokens=SPECIAL_TOKENS)
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
    download_and_extract()
    data = load_data()
    print("Data loaded successfully.")
    print(f"Input vocab size: {len(data['input_char2idx'])}")
    print(f"Output vocab size: {len(data['output_char2idx'])}")
    print(f"Training set shape: {data['x_train'].shape}, {data['y_train'].shape}")
    print(f"Validation set shape: {data['x_val'].shape}, {data['y_val'].shape}")
    print(f"Test set shape: {data['x_test'].shape}, {data['y_test'].shape}")