from train import train,predict
from dataloader import createDataLoader
from dataCleanup import cleanData
import uci_dataset
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import openml
import pandas as pd
from gnnModel import FEModel
from datasetGen import createDataset

hyperparams = {
    'batch_size' : 32,
    'save_loss_interval' : 5,
    'print_interval' : 2,
    'save_model_interval' : 20,
    'n_epochs' : 60,
    'learning_rate' : 0.0005,
    'numfeature_use':100,
    'dataset_name':'madelon',
    'num_intent':5
}
#'learning_rate' : 0.0005,
#'batch_size' : 32,


#https://openml.github.io/openml-python/develop/examples/30_extended/datasets_tutorial.html
openml_list = openml.datasets.list_datasets()  # returns a dict

# Show a nice table with some key data properties
datalist = pd.DataFrame.from_dict(openml_list, orient="index")
datalist = datalist[["did", "name", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"]]

#print(f"First 10 of {len(datalist)} datasets...")
#datalist.head(n=10)

# The same can be done with lesser lines of code
openml_df = openml.datasets.list_datasets(output_format="dataframe")

datasetName = hyperparams['dataset_name'] #'gisette'
numfeature_use = hyperparams['numfeature_use']
did = list(openml_df[openml_df['name'] == datasetName]['did'])[-1]

dataset = openml.datasets.get_dataset(did)

# Get the dataset features and target variable
X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

# Print a summary
print(
    f"This is dataset '{dataset.name}', the target feature is "
    f"'{dataset.default_target_attribute}'"
)
print(f"URL: {dataset.url}")
print(dataset.description[:500])


fullDataset = pd.DataFrame(X, columns=attribute_names)
nfeature = len(list(fullDataset.columns))
#test purpose drop last n column
fullDataset.drop(columns=fullDataset.columns[numfeature_use - nfeature:], axis=1,  inplace=True)



fullDataset["class"] = y

numOfIntent = hyperparams['num_intent']
batchSize = hyperparams['batch_size']

#fullDataset = fullDataset.apply(LabelEncoder().fit_transform).dropna()
#cleanData(fullDataset,datasetName,numOfIntent)
dataset,numRecords, train_mask_df = createDataset(datasetName)
data_loader, NUM_VAL, NUM_TRAIN = createDataLoader(dataset,numRecords,train_mask_df,batchSize)
model = FEModel(numfeature_use)
model = train(data_loader, model,NUM_VAL, NUM_TRAIN,hyperparams)
predict(model,data_loader, NUM_VAL, NUM_TRAIN )
