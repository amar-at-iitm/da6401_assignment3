# test_vanilla.py
import os
import torch
import wandb  

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
    wandb.init(project="vanilla-seq2seq", name="vanilla-test")  # <<< WandB init

    model, data = load_best_model()
    x_test, y_test = data["x_test"].to(DEVICE), data["y_test"].to(DEVICE)

    input_vocab = data["input_idx2char"]
    output_vocab = data["output_idx2char"]

    with torch.no_grad():
        val_output = model(x_test, y_test, teacher_forcing_ratio=0.0)

        val_pred_flat = val_output[:, 1:].reshape(-1, val_output.size(-1))  # [batch * (seq_len - 1), output_dim]
        val_target_flat = y_test[:, 1:].reshape(-1)                         # [batch * (seq_len - 1)]

        pad_idx = data["output_char2idx"][SPECIAL_TOKENS["PAD"]]
        mask = val_target_flat != pad_idx
        sos_idx = data["output_char2idx"][SPECIAL_TOKENS["SOS"]]

        correct_preds = (val_pred_flat.argmax(1)[mask] == val_target_flat[mask]).sum().item()
        total_tokens = mask.sum().item()
        val_acc = correct_preds / total_tokens if total_tokens > 0 else 0.0

        print(f"Character-Level Test Accuracy: {val_acc:.4f}")
        wandb.log({"char_level_accuracy": val_acc})  

        # Character-wise Accuracy
        os.makedirs("predictions_vanilla", exist_ok=True)
        predictions_path = "predictions_vanilla/character_predictions.tsv"
        with open(predictions_path, "w", encoding="utf-8") as f_out:
            f_out.write("Input\tTarget\tPredicted\n")
            preds = val_output.argmax(2)  # shape: [batch, seq_len]
            for i in range(x_test.size(0)):
                input_str = idx_to_string(x_test[i], input_vocab)
                target_str = clean_string(idx_to_string(y_test[i], output_vocab))
                pred_str = clean_string(idx_to_string(preds[i], output_vocab))
                f_out.write(f"{input_str}\t{target_str}\t{pred_str}\n")

        print(f"Saved character-wise predictions to: {predictions_path}")

    # Exact Match Accuracy 
    exact_matches = 0
    all_outputs = []
    for i in range(x_test.size(0)):
        pred_str = clean_string(idx_to_string(preds[i], output_vocab))
        target_str = clean_string(idx_to_string(y_test[i], output_vocab))
        input_str = idx_to_string(x_test[i], input_vocab)

        if pred_str == target_str:
            exact_matches += 1

        all_outputs.append((input_str, target_str, pred_str))

    exact_accuracy_raw = exact_matches / x_test.size(0)

    # Saving predictions to file
    os.makedirs("predictions_vanilla", exist_ok=True)
    with open("predictions_vanilla/all_predictions.tsv", "w", encoding="utf-8") as f:
        f.write("Input\tTarget\tPredicted\n")
        for inp, tgt, pred in all_outputs:
            f.write(f"{inp}\t{tgt}\t{pred}\n")

    
    with open("predictions_vanilla/accuracy.txt", "w") as f:
        f.write(f"Character-Level Accuracy: {val_acc:.4f}\n")
        f.write(f"Exact Match Accuracy : {exact_accuracy_raw:.4f}\n")

    # WandB logging
    wandb.log({
        "exact_match_accuracy": exact_accuracy_raw
    })  

  
    print(f"Total Words in Test: {x_test.size(0)}")
    print(f"Exact Match Accuracy:     {exact_accuracy_raw:.4f}")

    wandb.finish()  

if __name__ == "__main__":
    test_model()
