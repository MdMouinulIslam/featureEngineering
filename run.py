from train import train





hyperparams = {
    'batch_size' : 1,
    'save_loss_interval' : 5,
    'print_interval' : 20,
    'save_model_interval' : 20,
    'n_epochs' : 200,
    'learning_rate' : 0.01
}

train(hyperparams)
