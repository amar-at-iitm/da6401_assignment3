# test_attention.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# wandb integration
import wandb
wandb.init(project="DA6401_assign_3", name="Test_Attention")

#/////////////////////////////////////////////////////////////
# Devnagri Font setup
import matplotlib.font_manager as fm
font_path = "font/NotoSansDevanagari-VariableFont_wdth,wght.ttf"
devanagari_font = fm.FontProperties(fname=font_path)
#//////////////////////////////////////////////////////////////
from data_preparation import load_data
from local_functions import idx_to_string, SPECIAL_TOKENS
from models.attention import Attention
from models.attention_encoder import Encoder
from models.attention_decoder import AttentionDecoder
from models.attention_seq2seq import Seq2Seq
from best_model_attention import EMBEDDING_DIM, HIDDEN_DIM, ENCODER_LAYERS, DECODER_LAYERS, RNN_TYPE, DROPOUT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_CONFIG = {
    "embedding_dim": EMBEDDING_DIM,
    "hidden_dim": HIDDEN_DIM,
    "encoder_layers": ENCODER_LAYERS,
    "decoder_layers": DECODER_LAYERS,
    "dropout": DROPOUT,
    "rnn_type": RNN_TYPE, 
}

def clean_string(s):
    for tok in SPECIAL_TOKENS.values():
        s = s.replace(tok, "")
    return s


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
    attention = Attention(BEST_CONFIG["hidden_dim"]).to(DEVICE)
    decoder = AttentionDecoder(
        output_dim=output_dim,
        emb_dim=BEST_CONFIG["embedding_dim"],
        hidden_dim=BEST_CONFIG["hidden_dim"],
        n_layers=BEST_CONFIG["decoder_layers"],
        cell_type=BEST_CONFIG["rnn_type"],
        attention=attention,
        dropout=BEST_CONFIG["dropout"]
    ).to(DEVICE)

    model = Seq2Seq(encoder, decoder, DEVICE, sos_token_id=sos_idx, eos_token_id=eos_idx).to(DEVICE)
    model.load_state_dict(torch.load("best_model_attention.pt", map_location=DEVICE))
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
    attention_samples = 0
    corrected_samples = set()

    confusion_counts_char = defaultdict(lambda: defaultdict(int))
    confusion_counts_seq = {"correct": 0, "incorrect": 0}

    os.makedirs("predictions_attention", exist_ok=True)
    output_file = open("predictions_attention/all_predictions.tsv", "w", encoding="utf-8")
    output_file.write("Input\tTarget\tPrediction\n")

    with torch.no_grad():
        for batch_id, (src, trg) in enumerate(tqdm(test_loader, desc="Testing")):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            pred_list, attn_batch, _ = model.predict(src, beam_size=1)
            batch_size = src.size(0)
            pred_tokens = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(seq, device=DEVICE) for seq in pred_list],
                batch_first=True,
                padding_value=pad_idx
            )

            for i in range(batch_size):
                input_str = idx_to_string(src[i], input_vocab)
                target_str = clean_string(idx_to_string(trg[i], output_vocab))
                pred_str = clean_string(idx_to_string(pred_tokens[i], output_vocab))

                all_predictions.append((input_str, target_str, pred_str))
                output_file.write(f"{input_str}\t{target_str}\t{pred_str}\n")

                if pred_str == target_str:
                    total_correct_exact += 1
                    confusion_counts_seq["correct"] += 1
                    if (input_str, target_str, pred_str) not in corrected_samples:
                        with open("predictions_attention/corrected_cases.txt", "a", encoding="utf-8") as f_corr:
                            f_corr.write(f"Input   : {input_str}\nTarget  : {target_str}\nPredicted: {pred_str}\n\n")
                        corrected_samples.add((input_str, target_str, pred_str))
                else:
                    confusion_counts_seq["incorrect"] += 1
                total_exact_total += 1

                correct_char_count = sum(p == t for p, t in zip(pred_str, target_str))
                total_correct_chars += correct_char_count
                total_char_total += len(target_str)

                for t_char, p_char in zip(target_str, pred_str):
                    confusion_counts_char[t_char][p_char] += 1

                if attention_samples < 10:
                    attention_matrix = attn_batch[i][:len(pred_str), :len(input_str)].cpu().numpy()
                    plt.figure(figsize=(8, 6))
                    ax = sns.heatmap(attention_matrix, xticklabels=list(input_str), yticklabels=list(pred_str), cmap="viridis")
                    ax.set_xlabel("Input", fontproperties=devanagari_font)
                    ax.set_ylabel("Prediction", fontproperties=devanagari_font)
                    ax.set_title(f"Attention Heatmap {attention_samples + 1}", fontproperties=devanagari_font)
                    for label in ax.get_xticklabels():
                        label.set_fontproperties(devanagari_font)
                    for label in ax.get_yticklabels():
                        label.set_fontproperties(devanagari_font)
                    plt.tight_layout()
                    filename = f"predictions_attention/heatmap_{attention_samples + 1}.png"
                    plt.savefig(filename)
                    wandb.log({f"Attention Heatmap {attention_samples + 1}": wandb.Image(filename)})
                    plt.close()
                    attention_samples += 1

    output_file.close()
    exact_accuracy = total_correct_exact / total_exact_total if total_exact_total > 0 else 0.0
    char_accuracy = total_correct_chars / total_char_total if total_char_total > 0 else 0.0

    with open("predictions_attention/accuracy.txt", "w") as acc_file:
        acc_file.write(f"Exact Match Accuracy: {exact_accuracy:.4f}\n")
        acc_file.write(f"Character-Level Accuracy: {char_accuracy:.4f}\n")

    print(f"\nExact Match Accuracy:     {exact_accuracy:.4f} (full-sequence match)")
    print(f"Character-Level Accuracy: {char_accuracy:.4f} (token-wise match)\n")

    wandb.log({
        "Exact Match Accuracy": exact_accuracy,
        "Character-Level Accuracy": char_accuracy,
        "Sample Predictions": wandb.Table(
            columns=["Input", "Target", "Prediction"],
            data=all_predictions[:10]
        )
    })

    print("Sample Predictions:\n" + "-" * 50)
    for i in range(min(10, len(all_predictions))):
        print(f"Input    : {all_predictions[i][0]}")
        print(f"Target   : {all_predictions[i][1]}")
        print(f"Predicted: {all_predictions[i][2]}")
        print("-" * 50)

    grid_image = Image.new("RGB", (3 * 400, 3 * 300))
    for idx in range(9):
        path = f"predictions_attention/heatmap_{idx + 1}.png"
        if os.path.exists(path):
            img = Image.open(path).resize((400, 300))
            x = (idx % 3) * 400
            y = (idx // 3) * 300
            grid_image.paste(img, (x, y))
    grid_path = "predictions_attention/attention_grid.png"
    grid_image.save(grid_path)
    wandb.log({"Attention Grid": wandb.Image(grid_path)})

    wandb.finish()


if __name__ == "__main__":
    test_model()
