# sweep_config.py
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'name': 'vanilla-training',
    'description': 'Hyperparameter tuning for Seq2Seq model without attention',
    'program': 'train_vanilla.py',
    'parameters': {
        'embedding_dim': {"values": [16, 32, 64, 128, 256]},
        'hidden_dim': {'values': [32, 64, 128, 256]},
        'dropout': {'values': [0.1, 0.2, 0.3]},
        'rnn_type': {"values": ["RNN", "GRU", "LSTM"]},
        'encoder_layers': {'values': [1, 2, 3]},
        'decoder_layers': {'values': [1, 2, 3]},
        'batch_size': {'values': [32, 64]},
        'beam_size': {'values': [1, 3, 5]},
        'epochs': {'value': 10},
        'patience': {'value': 4}
    }
}
