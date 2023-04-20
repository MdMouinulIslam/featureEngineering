from train import train,predict
from dataloader import createDataLoader
from dataCleanup import cleanData
import uci_dataset
from sklearn.preprocessing import LabelEncoder

hyperparams = {
    'batch_size' : 20,
    'save_loss_interval' : 5,
    'print_interval' : 2,
    'save_model_interval' : 20,
    'n_epochs' : 10,
    'learning_rate' : 0.00001,
    'trainTestSplit':0.7
}

fullDataset = uci_dataset.load_audiology()
fullDataset = fullDataset.apply(LabelEncoder().fit_transform).dropna()
cleanData(fullDataset,"train")
data_loader, NUM_VAL, NUM_TRAIN = createDataLoader(hyperparams['trainTestSplit'])
model = train(data_loader, NUM_VAL, NUM_TRAIN,hyperparams)
predict(model,data_loader, NUM_VAL, NUM_TRAIN )
