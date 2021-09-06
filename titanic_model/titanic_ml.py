from titanic_model.titanic_data import *
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 

data = pd.read_csv("data/titanic.csv")

def prepare_data(data):
    features = pd.DataFrame()
    features["Pclass"] = data["Pclass"]
    features["Age"] = data["Age"].fillna(-1)
    features["SibSp"] = data["SibSp"]
    features["Parch"] = data["Parch"]
    features["Sqrt(Fare)"] = np.sqrt(data["Fare"])
    features = features.join(cat_to_num(data['Sex']))
    features = features.join(cat_to_num(data['Embarked']))
    return features

def cat_to_num(data):
    categories = set(data) 
    features = {} 
    for cat in categories: 
        features[f"{data.name}={cat}"] = (data == cat).astype("int")
    return pd.DataFrame(features)

def build_train_test_data(data):
    trainRate = int(0.8 * len(data))
    trainData = data[:trainRate]
    testData = data[trainRate:]
    return trainData, testData

def build_train_test_features(trainData, testData):
    return prepare_data(trainData), prepare_data(testData)

def build_linear_model(data):
    trainData, testData = build_train_test_data(data)    
    trainFeatures, testFeatures = build_train_test_features(trainData, testData)

    model = LogisticRegression(solver='liblinear')
    model.fit(trainFeatures, trainData['Survived'])
    return model.score(testFeatures, testData["Survived"])

def build_SVC_model(data):
    trainData, testData = build_train_test_data(data)    
    trainFeatures, testFeatures = build_train_test_features(trainData, testData)

    model = SVC(verbose=True)
    model.fit(trainFeatures, trainData['Survived'])
    return model.score(testFeatures, testData["Survived"])

build_linear_model(data)