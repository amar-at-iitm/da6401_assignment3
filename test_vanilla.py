# test_vanilla.py
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from data_preparation import load_data
from local_functions import  idx_to_string, SPECIAL_TOKENS
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from best_model import EMBEDDING_DIM, HIDDEN_DIM, ENCODER_LAYERS, DECODER_LAYERS, RNN_TYPE, DROPOUT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config — match the best model config (adjust these if needed)
BEST_CONFIG = {
    "embedding_dim": EMBEDDING_DIM,
    "hidden_dim": HIDDEN_DIM,
    "encoder_layers": ENCODER_LAYERS,
    "decoder_layers": DECODER_LAYERS,
    "dropout": DROPOUT,
    "rnn_type": RNN_TYPE,  
}

def load_best_model():
    #data = load_data(split ="test")
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
    # x_test = data["x_test"].to(DEVICE)
    # y_test = data["y_test"].to(DEVICE)

    input_vocab = data["input_idx2char"]
    output_vocab = data["output_idx2char"]
    pad_idx = data["output_char2idx"][SPECIAL_TOKENS["PAD"]]

    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64)

    all_predictions = []
    total_correct = 0
    total_tokens = 0
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    print("First sample x_test:", x_test[0])
    print("First sample y_test:", y_test[0])

    os.makedirs("predictions_vanilla", exist_ok=True)

    with torch.no_grad():
        for batch_id, (src, trg) in enumerate(tqdm(test_loader, desc="Testing")):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg=trg, teacher_forcing_ratio=0.0)  # Greedy decode

            pred_tokens = output.argmax(2)

            # Calculate accuracy
            total_correct += ((pred_tokens[:, 1:] == trg[:, 1:]) & (trg[:, 1:] != pad_idx)).sum().item()
            total_tokens += (trg[:, 1:] != pad_idx).sum().item()

            for i in range(src.size(0)):
                input_str = idx_to_string(src[i], input_vocab)
                target_str = idx_to_string(trg[i], output_vocab)
                pred_str = idx_to_string(pred_tokens[i], output_vocab)

                all_predictions.append((input_str, target_str, pred_str))

                # Save individual predictions to file (optional)
                with open(f"predictions_vanilla/pred_{batch_id}_{i}.txt", "w", encoding="utf-8") as f:
                    f.write(f"Input   : {input_str}\n")
                    f.write(f"Target  : {target_str}\n")
                    f.write(f"Predicted: {pred_str}\n")

    test_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    print(f"\nTest Accuracy: {test_accuracy:.4f} (exact character match)\n")

    # Show 10 samples
    print("Sample Predictions:\n" + "-" * 50)
    for i in range(10):
        print(f"Input    : {all_predictions[i][0]}")
        print(f"Target   : {all_predictions[i][1]}")
        print(f"Predicted: {all_predictions[i][2]}")
        print("-" * 50)

if __name__ == "__main__":
    test_model()
