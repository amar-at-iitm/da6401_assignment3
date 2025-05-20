# visualize_attention.py
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from test_attention import load_best_model
from local_functions import SPECIAL_TOKENS
import matplotlib.image as mpimg
from torch.utils.data import TensorDataset, DataLoader
from IPython.display import display, HTML
from matplotlib import font_manager, rcParams
import wandb

#/////////////////////////////////////////////////////////////
# Devnagri Font setup
import matplotlib.font_manager as fm
font_path = "font/NotoSansDevanagari-VariableFont_wdth,wght.ttf"
devanagari_font = fm.FontProperties(fname=font_path)
font_manager.fontManager.addfont(font_path)
rcParams['font.family'] = devanagari_font.get_name()
#//////////////////////////////////////////////////////////////

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wandb.init(project="DA6401_assign_3", name="Attention_vizualization")

# LOADING BEST MODEL

model, data = load_best_model()
x_test, y_test = data["x_test"].to(DEVICE), data["y_test"].to(DEVICE)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64)
input_vocab = data["input_idx2char"]
output_vocab = data["output_idx2char"]
input_eos_idx = data["input_char2idx"][SPECIAL_TOKENS["EOS"]]
eos_idx = data["output_char2idx"][SPECIAL_TOKENS["EOS"]]
inv_input_vocab = {v: k for k, v in input_vocab.items()}
inv_output_vocab = {v: k for k, v in output_vocab.items()}
model.eval()


correct_words, total_words = 0, 0
total_correct_chars, total_chars = 0, 0
predictions = []
viz_count, viz_samples = 0, 10

with torch.no_grad():
    for batch_idx, (src, tgt) in enumerate(test_loader):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        for i in range(src.size(0)):
            src_single = src[i].unsqueeze(0)
            tgt_single = tgt[i].unsqueeze(0)
           
            output = model.predict(src_single, max_len=30, beam_size=5)
            if isinstance(output, list):  # beam search case
                pred_seq, attn_weights, _ = output[0]
            else:  # greedy decode
                pred_seq, attn_weights, _ = output

            pred_indices = pred_seq
            if eos_idx in pred_indices:
                pred_indices = pred_indices[:pred_indices.index(eos_idx)]
            pred_str = ''.join([inv_output_vocab.get(idx, '?') for idx in pred_indices])

            tgt_indices = tgt_single[0, 1:].tolist()
            if eos_idx in tgt_indices:
                tgt_indices = tgt_indices[:tgt_indices.index(eos_idx)]
            tgt_str = ''.join([inv_output_vocab.get(idx, '?') for idx in tgt_indices])

            input_indices = src_single[0].tolist()
            
            
            input_str = ''.join([inv_input_vocab.get(idx, '?') for idx in input_indices if idx not in [0, input_eos_idx]])

            correct_word = pred_str == tgt_str
            correct_chars = sum(1 for p, t in zip(pred_str, tgt_str) if p == t)

            correct_words += correct_word
            total_words += 1
            total_correct_chars += correct_chars
            total_chars += len(tgt_str)

            predictions.append({
                'input': input_str,
                'target': tgt_str,
                'prediction': pred_str,
                'correct_word': correct_word,
                'correct_chars': correct_chars,
                'total_chars': len(tgt_str)
            })

            if viz_count < viz_samples:
                input_chars = [inv_input_vocab.get(idx, '?') for idx in input_indices if idx not in [0, input_eos_idx]]
          
                pred_chars = [inv_output_vocab.get(idx, '?') for idx in pred_indices]
                #attn_weights = [attn.cpu().numpy()[0] for attn in best_attn_list]

                fig, axes = plt.subplots(len(pred_chars), 1, figsize=(max(6, 0.7 * len(input_chars)), 1.5 * len(pred_chars) + 1))
                if len(pred_chars) == 1:
                    axes = [axes]

                fig.suptitle(f"Input: {input_str} -> Predicted: {pred_str}", fontsize=14, fontproperties=devanagari_font, y=1.05)

                for j, ax in enumerate(axes):
                    ax.set_title(f"Output char: {pred_chars[j]}", fontproperties=devanagari_font, fontsize=10)
                    for k, char in enumerate(input_chars):
                        weight = attn_weights[j][k]
                        ax.text(k * 0.7, 0, char, fontsize=14, ha='center', va='center',
                                bbox=dict(facecolor=plt.cm.Greens(weight), alpha=1, edgecolor='none', boxstyle='round,pad=0.2'),
                                fontproperties=devanagari_font)
                    ax.set_xlim(-0.5, len(input_chars) * 0.7)
                    ax.set_ylim(-0.5, 0.5)
                    ax.axis('off')

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                viz_path = f'predictions_attention/attent_@sample{viz_count}.png'
                plt.savefig(viz_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
                wandb.log({f"attent_@sample{viz_count}": wandb.Image(viz_path)})
                viz_count += 1


if viz_count > 0:
    grid_cols = 2
    grid_rows = (viz_count + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 1.2 * grid_rows * 5))

    for i in range(grid_rows * grid_cols):
        ax = axes.flat[i]
        if i < viz_count:
            img = mpimg.imread(f'predictions_attention/attent_@sample{i}.png')
            ax.imshow(img)
            ax.set_title(f"Sample {i}", fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    combined_path = 'predictions_attention/attent_@_samples_combined.png'
    plt.savefig(combined_path, dpi=150)
    plt.close(fig)
    wandb.log({"attent_@_samples_combined": wandb.Image(combined_path)})
    print(f"Saved combined attention image: {combined_path}")


word_accuracy = 100 * correct_words / total_words
char_accuracy = 100 * total_correct_chars / total_chars
wandb.log({"test_word_accuracy": word_accuracy, "test_char_accuracy": char_accuracy})
print(f"\nWord Accuracy: {word_accuracy:.2f}%")
print(f"Char Accuracy: {char_accuracy:.2f}%")


df_predictions = pd.DataFrame(predictions)
csv_path = 'predictions_attention/pred_attention.csv'
html_plain_path = 'predictions_attention/predictions_all_plain_attention.html'
html_colored_path = 'predictions_attention/predictions_sample_colored_attention.html'

df_predictions.to_csv(csv_path, index=False)
df_predictions.to_html(html_plain_path, index=False)

def highlight_row(row):
    return ['background-color: #d4edda;' if row['correct_word'] else 'background-color: #f8d7da;'] * len(row)

sample_df = pd.DataFrame(predictions[:viz_count])

styled_sample = sample_df.style.apply(highlight_row, axis=1)\
    .set_table_styles([
        {"selector": "th", "props": [("font-size", "110%"), ("text-align", "center")]},
        {"selector": "td", "props": [("text-align", "center")]}
    ])\
    .set_properties(**{'border': '1px solid black', 'padding': '5px'})


with open(html_colored_path, 'w', encoding='utf-8') as f:
    f.write(f"<h3>Sample Predictions (Color-Coded)</h3>\n{styled_sample.to_html()}")

display(HTML("<h3>Sample Predictions (Color-Coded)</h3>"))
display(styled_sample)


artifact = wandb.Artifact('predictions_attention', type='predictions')
artifact.add_file(csv_path)
artifact.add_file(html_plain_path)
artifact.add_file(html_colored_path)
for i in range(viz_count):
    artifact.add_file(f'predictions_attention/attent_@sample{i}.png')
if viz_count > 0:
    artifact.add_file(combined_path)
wandb.log_artifact(artifact)

wandb.log({
    "sample_predictions_table": wandb.Html(html_colored_path),
    "attention_samples_combined": wandb.Image(combined_path)
})

wandb.finish()