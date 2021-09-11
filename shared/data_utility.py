import pandas as pd 
import numpy as np

def cat_to_num(data):
    categories = set(data) 
    features = {} 
    for cat in categories: 
        features[f"{data.name}={cat}"] = (data == cat).astype("int")
    return pd.DataFrame(features)

def build_train_test_data(data, factor = 0.8):
    trainRate = int(factor * len(data))
    trainData = data[:trainRate]
    testData = data[trainRate:]
    return trainData, testData

def build_train_test_features(trainData, testData, prepareFunc):
    return prepareFunc(trainData), prepareFunc(testData)