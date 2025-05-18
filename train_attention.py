# train_attention.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
import os
from tqdm import tqdm

from data_preparation import load_data, SPECIAL_TOKENS
from local_functions import seed_all, calculate_accuracy, save_best_model_attention_config
from models.attention_encoder import Encoder
from models.attention import Attention
from models.attention_decoder import AttentionDecoder
from models.attention_seq2seq import Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_global_best(path="best_accuracy_attention.txt"):
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return float(f.read().strip())
            except ValueError:
                return 0.0
    return 0.0

def save_global_best(val_acc, path="best_accuracy_attention.txt"):
    with open(path, "w") as f:
        f.write(str(val_acc))


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        run_name = f"attn_{config.rnn_type}_emb-{config.embedding_dim}_bs-{config.batch_size}_el-{config.encoder_layers}_dl-{config.decoder_layers}_hd-{config.hidden_dim}_dropout-{config.dropout}"
        wandb.run.name = run_name
        wandb.run.save()

        seed_all()

        data = load_data()
        x_train, y_train = data["x_train"].to(DEVICE), data["y_train"].to(DEVICE)
        x_val, y_val = data["x_val"].to(DEVICE), data["y_val"].to(DEVICE)

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True)

        input_dim = len(data["input_char2idx"])
        output_dim = len(data["output_char2idx"])
        pad_idx = data["output_char2idx"][SPECIAL_TOKENS["PAD"]]
        sos_idx = data["output_char2idx"][SPECIAL_TOKENS["SOS"]]
        eos_idx = data["output_char2idx"][SPECIAL_TOKENS["EOS"]]

        encoder = Encoder(
            input_dim=input_dim,
            emb_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.encoder_layers,
            cell_type=config.rnn_type,
            dropout=config.dropout
        ).to(DEVICE)

        attention = Attention(config.hidden_dim).to(DEVICE)

        decoder = AttentionDecoder(
            output_dim=output_dim,
            emb_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.decoder_layers,
            cell_type=config.rnn_type,
            dropout=config.dropout,
            attention=attention
        ).to(DEVICE)

        model = Seq2Seq(encoder, decoder, DEVICE, sos_token_id=sos_idx, eos_token_id=eos_idx).to(DEVICE)

        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        wandb.watch(model, log="gradients", log_freq=100)

        no_improve = 0
        patience = config.get("patience", 3)

        for epoch in range(config.epochs):
            model.train()
            epoch_loss = 0
            correct_preds = 0
            total_tokens = 0

            for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}", ncols=100, colour="cyan"):
                optimizer.zero_grad()

                # Forward pass with attention
                output, attentions = model(batch_x, batch_y, teacher_forcing_ratio=config.get("teacher_forcing_ratio", 0.5))

                output_dim = output.shape[-1]
                output_flat = output[:, 1:].reshape(-1, output_dim)
                target_flat = batch_y[:, 1:].reshape(-1)

                # Cross-entropy loss
                loss = criterion(output_flat, target_flat)

                # Optional: attention regularization
                attn_entropy = -torch.sum(attentions * torch.log(attentions + 1e-8), dim=-1).mean()
                loss += 0.01 * attn_entropy

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                with torch.no_grad():
                    correct_preds += ((output.argmax(2)[:, 1:] == batch_y[:, 1:]) & (batch_y[:, 1:] != pad_idx)).sum().item()
                    total_tokens += (batch_y[:, 1:] != pad_idx).sum().item()

            train_acc = correct_preds / total_tokens if total_tokens > 0 else 0.0

            # Evaluate on validation set as usual
            model.eval()
            with torch.no_grad():
                val_output, _ = model(x_val, y_val, teacher_forcing_ratio=0.0)
                val_loss = criterion(
                    val_output[:, 1:].reshape(-1, output_dim),
                    y_val[:, 1:].reshape(-1)
                )
                val_acc = calculate_accuracy(val_output[:, 1:], y_val[:, 1:], pad_idx)

            # Log to wandb or print
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_loss / len(train_loader),
                "train_acc": train_acc,
                "val_loss": val_loss.item(),
                "val_acc": val_acc,
                "attention_entropy": attn_entropy.item()
            })


            print(f" Train Loss: {epoch_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f}")

            global_best_path = "best_accuracy_attention.txt"
            current_best = load_global_best(global_best_path)

            if val_acc > current_best:
                torch.save(model.state_dict(), "best_model_attention.pt")
                save_global_best(val_acc, global_best_path)
                save_best_model_attention_config(config, model_path="best_model_attention.pt")
                print(f"New global best model saved with val_acc: {val_acc:.4f}")
                no_improve = 0
            else:
                no_improve += 1
                print(f"No improvement. Counter: {no_improve}/{patience}")
                if no_improve >= patience:
                    print("Early stopping triggered.")
                    break

if __name__ == "__main__":
    import sweep_config
    sweep_id = wandb.sweep(sweep_config.sweep_config_attention, project="DA6401_assign_3")
    wandb.agent(sweep_id, function=train, count=30)
    wandb.finish()
    print("Sweep complete.")
