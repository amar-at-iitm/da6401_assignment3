# test_vanilla.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from data_preparation import load_data
from local_functions import idx_to_string, SPECIAL_TOKENS
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from best_model import EMBEDDING_DIM, HIDDEN_DIM, ENCODER_LAYERS, DECODER_LAYERS, RNN_TYPE, DROPOUT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_CONFIG = {
    "embedding_dim": EMBEDDING_DIM,
    "hidden_dim": HIDDEN_DIM,
    "encoder_layers": ENCODER_LAYERS,
    "decoder_layers": DECODER_LAYERS,
    "dropout": DROPOUT,
    "rnn_type": RNN_TYPE, 
}

def load_best_model():
    data = load_data()
    input_dim = len(data["input_char2idx"])
    output_dim = len(data["output_char2idx"])
    pad_idx = data["output_char2idx"][SPECIAL_TOKENS["PAD"]]
    sos_idx = data["output_char2idx"][SPECIAL_TOKENS["SOS"]]
    eos_idx = data["output_char2idx"][SPECIAL_TOKENS["EOS"]]

    encoder = Encoder(
        input_dim=input_dim,
        emb_dim=BEST_CONFIG["embedding_dim"],
        hidden_dim=BEST_CONFIG["hidden_dim"],
        n_layers=BEST_CONFIG["encoder_layers"],
        cell_type=BEST_CONFIG["rnn_type"],
        dropout=BEST_CONFIG["dropout"]
    ).to(DEVICE)

    decoder = Decoder(
        output_dim=output_dim,
        emb_dim=BEST_CONFIG["embedding_dim"],
        hidden_dim=BEST_CONFIG["hidden_dim"],
        n_layers=BEST_CONFIG["decoder_layers"],
        cell_type=BEST_CONFIG["rnn_type"],
        dropout=BEST_CONFIG["dropout"]
    ).to(DEVICE)

    model = Seq2Seq(encoder, decoder, DEVICE, sos_token_id=sos_idx, eos_token_id=eos_idx).to(DEVICE)
    model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
    model.eval()
    return model, data

def test_model():
    model, data = load_best_model()
    x_test, y_test = data["x_test"].to(DEVICE), data["y_test"].to(DEVICE)

    input_vocab = data["input_idx2char"]
    output_vocab = data["output_idx2char"]
    pad_idx = data["output_char2idx"][SPECIAL_TOKENS["PAD"]]

    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64)

    all_predictions = []
    total_correct_exact = 0
    total_exact_total = 0
    total_correct_chars = 0
    total_char_total = 0

    # Confusion matrix tracking
    confusion_counts_char = defaultdict(lambda: defaultdict(int))
    confusion_counts_seq = {"correct": 0, "incorrect": 0}
    all_chars = set(output_vocab.values())

    os.makedirs("predictions_vanilla", exist_ok=True)
    output_file = open("predictions_vanilla/all_predictions.tsv", "w", encoding="utf-8")
    output_file.write("Input\tTarget\tPrediction\n")

    with torch.no_grad():
        for batch_id, (src, trg) in enumerate(tqdm(test_loader, desc="Testing")):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg=trg, teacher_forcing_ratio=0.0)
            pred_tokens = output.argmax(2)
            batch_size = src.size(0)

            for i in range(batch_size):
                input_str = idx_to_string(src[i], input_vocab)
                target_str_raw = idx_to_string(trg[i], output_vocab)
                pred_str_raw = idx_to_string(pred_tokens[i], output_vocab)

                target_str = target_str_raw.replace(SPECIAL_TOKENS["SOS"], "").replace(SPECIAL_TOKENS["EOS"], "").replace(SPECIAL_TOKENS["PAD"], "")
                pred_str = pred_str_raw.replace(SPECIAL_TOKENS["SOS"], "").replace(SPECIAL_TOKENS["EOS"], "").replace(SPECIAL_TOKENS["PAD"], "")

                all_predictions.append((input_str, target_str, pred_str))
                output_file.write(f"{input_str}\t{target_str}\t{pred_str}\n")

                # Sequence-level accuracy
                if pred_str == target_str:
                    total_correct_exact += 1
                    confusion_counts_seq["correct"] += 1
                else:
                    confusion_counts_seq["incorrect"] += 1
                total_exact_total += 1

                # Character-level confusion
                for p_char, t_char in zip(pred_str, target_str):
                    if p_char == t_char:
                        total_correct_chars += 1
                    confusion_counts_char[t_char][p_char] += 1
                total_char_total += len(target_str)

    output_file.close()

    # Final accuracy
    exact_accuracy = total_correct_exact / total_exact_total if total_exact_total > 0 else 0.0
    char_accuracy = total_correct_chars / total_char_total if total_char_total > 0 else 0.0

    print(f"\nExact Match Accuracy:     {exact_accuracy:.4f} (full-sequence match)")
    print(f"Character-Level Accuracy: {char_accuracy:.4f} (token-wise match)\n")

    # Show sample predictions
    print("Sample Predictions:\n" + "-" * 50)
    for i in range(min(10, len(all_predictions))):
        print(f"Input    : {all_predictions[i][0]}")
        print(f"Target   : {all_predictions[i][1]}")
        print(f"Predicted: {all_predictions[i][2]}")
        print("-" * 50)

    # Character-level confusion matrix
    sorted_chars = sorted(list(all_chars))
    matrix = np.zeros((len(sorted_chars), len(sorted_chars)), dtype=int)
    char_to_idx = {c: i for i, c in enumerate(sorted_chars)}
    for true_char, preds in confusion_counts_char.items():
        for pred_char, count in preds.items():
            i = char_to_idx[true_char]
            j = char_to_idx[pred_char]
            matrix[i][j] = count

    df_cm_char = pd.DataFrame(matrix, index=sorted_chars, columns=sorted_chars)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_cm_char, cmap="YlGnBu", annot=False, fmt="d")
    plt.title("Character-Level Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("predictions_vanilla/char_confusion_matrix.png")
    plt.close()
    print("Character confusion matrix saved to predictions_vanilla/char_confusion_matrix.png")

    # Sequence-level confusion matrix (bar plot)
    labels = ["Correct", "Incorrect"]
    values = [confusion_counts_seq["correct"], confusion_counts_seq["incorrect"]]
    plt.figure(figsize=(6, 5))
    sns.barplot(x=labels, y=values, palette="Set2")
    plt.title("Sequence-Level (Exact Match) Confusion")
    plt.ylabel("Number of Sequences")
    plt.tight_layout()
    plt.savefig("predictions_vanilla/sequence_confusion.png")
    plt.close()
    print("Sequence-level confusion matrix saved to predictions_vanilla/sequence_confusion.png")


if __name__ == "__main__":
    test_model()
