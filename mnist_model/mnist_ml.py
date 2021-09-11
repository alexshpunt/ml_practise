from mnist_model.mnist_data import *
from sklearn.neighbors import KNeighborsClassifier
from shared.data_utility import * 
import pandas as pd 
import numpy as np

data = pd.read_csv("data/mnist_small.csv")

def prepare_data(data):
    return data 

dataTrain, dataTest = build_train_test_data(data) 
featuresTrain, featuresTest = build_train_test_features(dataTrain, dataTest, prepare_data)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(dataTrain.drop('label', axis=1), dataTrain['label'])
#TODO: Try it with a real image
knn.score(dataTest.drop('label', axis=1), dataTest['label'])

