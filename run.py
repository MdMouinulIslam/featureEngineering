from train import train,predict


hyperparams = {
    'batch_size' : 20,
    'save_loss_interval' : 5,
    'print_interval' : 2,
    'save_model_interval' : 20,
    'n_epochs' : 10,
    'learning_rate' : 0.1
}

model, dataLoader = train(hyperparams)
predict(model,dataLoader)
