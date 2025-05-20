# da6401_assignment3

---

#### `Amar Kumar`  `(MA24M002)`

#### `M.Tech (Industrial Mathematics and Scientific Computing)` `IIT Madras`

##### For more detail go to [wandb project report](https://wandb.ai/amar74384-iit-madras/DA6401_assign_3/reports/Amar-s-DA6401-Assignment-3--VmlldzoxMjY2NzE3Nw)

---

## Dakshina Transliteration: Seq2Seq and Attention-Based Models

### Overview

This assignment contains the problem of transliteration using the [Dakshina dataset](https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar). For a word in Latin script (romanized), the task is to generate the corresponding word in Devanagari script using neural sequence-to-sequence models(RNN, LSTM, GRU).

The task is to implement and compare two models:

1. A vanilla seq2seq model
2. An attention-enhanced seq2seq model

The project involves training, hyperparameter tuning using Weights & Biases (wandb), evaluation on a test set, error analysis, comparing both models, and visualizing attention.

### Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/amar-at-iitm/da6401_assignment3
   cd da6401_assignment3
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure WandB:

   ```bash
   wandb login
   ```

### Project Structure  

```structure
.

├── data/
│   ├── dakshina_dataset_v1.0.zip               # Original dataset archive
│   ├── vocab_input.json                        # Character-to-index mapping for input (Latin)
│   ├── vocab_output.json                       # Character-to-index mapping for output (Devanagari)
│   └── dakshina_dataset_v1.0/hi/lexicons/      # Hindi lexicons used for training/testing
│
├── font/
│   ├── NotoSansDevanagari-VariableFont_wdth,wght.ttf
|
├── models/                                     # Model architectures
│   ├── encoder.py                              # Encoder for vanilla model
│   ├── decoder.py                              # Decoder for vanilla model
│   ├── seq2seq.py                              # Seq2seq class integrating encoder and decoder
│   ├── attention.py                            # Attention mechanism
│   ├── attention_encoder.py                    # Encoder for attention model
│   ├── attention_decoder.py                    # Decoder for attention model
│   └── attention_seq2seq.py                    # Seq2seq class with attention
│
├── predictions_vanilla/                        # Model outputs on test set (vanilla)
├── predictions_attention/                      # Model outputs on test set (attention)
│
├── best_model.pt                               # Best vanilla model checkpoint
├── best_model.py                               # Script to load & use best vanilla model
├── best_model_attention.pt                     # Best attention model checkpoint
├── best_model_attention.py                     # Script to load & use best attention model
│
├── train_vanilla.py                            # Training script for vanilla model
├── test_vanilla.py                             # Testing script for vanilla model
├── train_attention.py                          # Training script for attention model
├── test_attention.py                           # Testing script for attention model
│
├── visualize_attention.py                      # Script to generate attention heatmaps
├── sweep_config.py                             # Configuration for wandb hyperparameter sweeps
├── data_preparation.py                         # Data preprocessing, vocab creation, padding
│
├── requirements.txt                            # Python dependencies
└── README.md                                   # Project overview
                   
```


## Prepare the Dataset

```bash
python data_preparation.py
```

- This downloads the data in `.tar` format from the ([original source](https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar)).
- Extracts the dataset into the `data/dakshina_dataset_v1.0/ directory`.
- Generates `vocab_input.json` and `vocab_output.json` and saves them in the `data/` folder.


## Trainning

### Vanilla Seq2Seq Model

```bash
python train_vanilla.py
```

- Trains a standard Seq2Seq model using a wandb sweep (Bayesian optimization; maximizing validation accuracy).
- Uses:
   - Train: data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv
   - Validation: data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv
- Logs metrics like training/validation loss and accuracy to wandb.

### Attention-based Seq2Seq Model

```bash
python train_attention.py
```

- Trains a Seq2Seq model with attention using a wandb sweep (Bayesian optimization; maximizing validation accuracy).
- Same data and logging setup as vanilla.

### Hyperparameter Tuning with WandB

`sweep_config.py` contains the hyperparameter for both vanilla and attention training.

- `sweep_config`: configuration for vanilla model
- `sweep_config_attention`: configuration for attention model


## Evaluation

### Vanilla Model

```bash
python test_vanilla.py
```

- Evaluates the vanilla model on the test set:
   - `data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv`
- Loads the model architecture from `best_model.py` and weights from `best_model.pt`.
- Saves predictions to `predictions_vanilla/`.

### Attention Model

```bash
python test_attention.py
```

- Evaluates the vanilla model on the same test set:
- Loads the model architecture from `best_model_attention.py` and weights from `best_model_attention.pt`.
- Saves predictions to `predictions_attention/`.

### Comparing Models

```bash
python compare_model.py
```

- Compares predictions from the vanilla and attention models.
- Logs comparison results (e.g., accuracy differences, corrected errors, consonent and vowel errors, error on sequence lenght) to wandb.


## Attention Visualizations

```bash
python visualize_attention.py
```

- Generates attention for select test samples to visualize alignment between input and output characters.


## References

* [Dakshina Dataset](https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar)
* [Blog1](https://google-research.github.io/seq2seq)
* [Blog2](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
* [Blog3](https://medium.com/data-science/visualising-lstm-activations-in-keras-b50206da96ff)
* [Article](https://distill.pub/2019/memorization-in-rnns/#appendix-autocomplete)


