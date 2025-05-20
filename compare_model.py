import os
import pandas as pd
import wandb

# File paths
VANILLA_PATH = "predictions_vanilla/all_predictions.tsv"
ATTENTION_PATH = "predictions_attention/all_predictions.tsv"

# Devanagari vowel sets
INDEPENDENT_VOWELS = {"अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ॠ", "ऌ", "ॡ", "ए", "ऐ", "ओ", "औ"}
DEPENDENT_VOWELS = {"ा", "ि", "ी", "ु", "ू", "ृ", "ॄ", "ॢ", "ॣ", "े", "ै", "ो", "ौ", "ं", "ः", "ँ"}
DEVANAGARI_VOWELS = INDEPENDENT_VOWELS | DEPENDENT_VOWELS

# Load predictions
vanilla_df = pd.read_csv(VANILLA_PATH, sep="\t")
attention_df = pd.read_csv(ATTENTION_PATH, sep="\t")

# Sanity check
assert all(vanilla_df["Input"] == attention_df["Input"]), "Mismatch in input data!"

# Prepare results
results = []
corrected_by_attention = []
regressed_by_attention = []

for i in range(len(vanilla_df)):
    input_str = vanilla_df.loc[i, "Input"]
    target_str = vanilla_df.loc[i, "Target"]
    vanilla_pred = vanilla_df.loc[i, "Predicted"]
    attention_pred = attention_df.loc[i, "Predicted"]

    is_vanilla_correct = vanilla_pred == target_str
    is_attention_correct = attention_pred == target_str

    if not is_vanilla_correct and is_attention_correct:
        corrected_by_attention.append((input_str, target_str, vanilla_pred, attention_pred))
    if is_vanilla_correct and not is_attention_correct:
        regressed_by_attention.append((input_str, target_str, vanilla_pred, attention_pred))

    results.append({
        "Input": input_str,
        "Target": target_str,
        "Vanilla": vanilla_pred,
        "Attention": attention_pred,
        "Vanilla_Correct": is_vanilla_correct,
        "Attention_Correct": is_attention_correct,
        "Length": len(target_str)
    })

df = pd.DataFrame(results)

# Accuracy
vanilla_acc = df["Vanilla_Correct"].mean()
attention_acc = df["Attention_Correct"].mean()

# Vowel/Consonant Error Analysis
vowel_errors = {"Vanilla": 0, "Attention": 0}
consonant_errors = {"Vanilla": 0, "Attention": 0}
total_vowels = 0
total_consonants = 0

for _, row in df.iterrows():
    tgt, van, att = row["Target"], row["Vanilla"], row["Attention"]
    for t_char, v_char, a_char in zip(tgt, van, att):
        if t_char in DEVANAGARI_VOWELS:
            total_vowels += 1
            if v_char != t_char:
                vowel_errors["Vanilla"] += 1
            if a_char != t_char:
                vowel_errors["Attention"] += 1
        else:
            total_consonants += 1
            if v_char != t_char:
                consonant_errors["Vanilla"] += 1
            if a_char != t_char:
                consonant_errors["Attention"] += 1

# Error by specific word lengths
length_error_stats = df.groupby("Length").agg({
    "Vanilla_Correct": lambda x: 1 - x.mean(),
    "Attention_Correct": lambda x: 1 - x.mean()
}).rename(columns={
    "Vanilla_Correct": "Vanilla_Error",
    "Attention_Correct": "Attention_Error"
}).sort_index()

for length in [1, 3, 6, 10]:
    if length not in length_error_stats.index:
        length_error_stats.loc[length] = {"Vanilla_Error": float('nan'), "Attention_Error": float('nan')}
length_error_stats = length_error_stats.sort_index()

# Print outputs
print("\nOverall Accuracy:")
print(f"  Vanilla   : {vanilla_acc:.4f}")
print(f"  Attention : {attention_acc:.4f}")

print("\nVowel Errors:")
print(f"  Vanilla   : {vowel_errors['Vanilla']} / {total_vowels}")
print(f"  Attention : {vowel_errors['Attention']} / {total_vowels}")

print("\nConsonant Errors:")
print(f"  Vanilla   : {consonant_errors['Vanilla']} / {total_consonants}")
print(f"  Attention : {consonant_errors['Attention']} / {total_consonants}")

print("\nError by Specific Sequence Lengths:")
for length in [1, 3, 6, 10]:
    v_err = length_error_stats.loc[length, "Vanilla_Error"]
    a_err = length_error_stats.loc[length, "Attention_Error"]
    print(f"  Length {length:2} → Vanilla: {v_err:.2%} | Attention: {a_err:.2%}")

print(f"\nAttention fixed {len(corrected_by_attention)} mistakes made by vanilla.")
print(f"Attention introduced {len(regressed_by_attention)} new errors.")

# Save outputs
os.makedirs("analysis", exist_ok=True)
df.to_csv("analysis/model_comparison.tsv", sep="\t", index=False)

def save_examples(examples, path, label):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Input\tTarget\tVanilla\tAttention\n")
        for inp, tgt, van, att in examples:
            f.write(f"{inp}\t{tgt}\t{van}\t{att}\n")
    print(f"Saved {label} cases to {path}")

save_examples(corrected_by_attention, "analysis/corrected_by_attention.tsv", "corrected")
save_examples(regressed_by_attention, "analysis/regressed_by_attention.tsv", "regressed")

print("\nAnalysis complete and saved in 'analysis/' folder.")


# -------------------
# WandB Logging
# -------------------

def log_to_wandb():
    wandb.init(project="DA6401_assign_3", name="Model_Comparison")
    
    log_data = {
        "Accuracy/Vanilla": vanilla_acc,
        "Accuracy/Attention": attention_acc,
        "Vowel Errors/Vanilla": vowel_errors["Vanilla"],
        "Vowel Errors/Attention": vowel_errors["Attention"],
        "Total Vowels": total_vowels,
        "Consonant Errors/Vanilla": consonant_errors["Vanilla"],
        "Consonant Errors/Attention": consonant_errors["Attention"],
        "Total Consonants": total_consonants,
        "Corrections by Attention": len(corrected_by_attention),
        "New Error by Attention": len(regressed_by_attention)
    }

    for length in [1, 3, 6, 10]:
        v_err = length_error_stats.loc[length, "Vanilla_Error"]
        a_err = length_error_stats.loc[length, "Attention_Error"]
        log_data[f"Length_{length}/Vanilla_Error"] = v_err
        log_data[f"Length_{length}/Attention_Error"] = a_err

    wandb.log(log_data)
    wandb.finish()

log_to_wandb()
