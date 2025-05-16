# da6401_assignment3

---

#### `Amar Kumar`  `(MA24M002)`

#### `M.Tech (Industrial Mathematics and Scientific Computing)` `IIT Madras`

##### For more detail go to [wandb project report](https://wandb.ai/amar74384-iit-madras/DA6401_assign_3/reports/Amar-s-DA6401-Assignment-3--VmlldzoxMjY2NzE3Nw)

---

### Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/amar-at-iitm/da6401_assignment2
   cd da6401_assignment2
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
│   ├── dakshina_dataset_v1.0.zip
|   ├── vocab_input.json
|   ├── vocab_output.json
|   ├── dakshina_dataset_v1.0/      # Original dataset
|        └── hi/
|            └── lexicons/
|
├── models/
│   ├── encoder.py
│   ├── decoder.py
│   ├── seq2seq.py    
|
├── predictions_vanilla/
├── predictions_attention/
├── best_model.pt
├── best_model.py
├── local_functions.py
├── train_vanilla.py
├── test_vanilla.py
├── sweep_config.py                  # Sweep configuration file
├── data_preparation.py              # Handles downloading, loading, vocab, padding
├── requirements.txt                 # List of Python dependencies
└── README.md                        # Root README with project overview
                   
```
