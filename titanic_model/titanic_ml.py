from titanic_model.titanic_data import *
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.metrics import roc_auc_score

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

def build_linear_model(data):
    trainData, testData = build_train_test_data(data)    
    trainFeatures, testFeatures = build_train_test_features(trainData, testData, prepare_data)
 
    model = LogisticRegression(solver='liblinear')
    model.fit(trainFeatures, trainData['Survived'])
    return model.score(testFeatures, testData["Survived"])

def build_SVC_model(data):
    trainData, testData = build_train_test_data(data)    
    trainFeatures, testFeatures = build_train_test_features(trainData, testData, prepare_data)

    model = SVC()
    model.fit(trainFeatures, trainData['Survived'])
    return model.score(testFeatures, testData["Survived"])

def grid_search(data):
    y = data['Survived']
    x = data.drop(['Survived', 'PassengerId', 'Cabin', 'Ticket', 'Name', 'Fare'], axis=1)
    x['Sex'] = x['Sex'].map(lambda x: 1 if x == 'male' else 0)
    x = x.join(cat_to_num(x.Embarked))
    x = x.drop(['Embarked', 'Embarked=nan'], axis=1)
    x = x.fillna(-1)
    
    gamVec, costVec = np.meshgrid(
        np.linspace(0.01, 10, 11),
        np.linspace(0.01, 10, 11))

    AUC = [] 
    N = len(y)
    K = 10

    folds = np.random.randint(0, K, size=N)
    for iParam in np.arange(len(gamVec.ravel())):
        yCVPrediction = np.empty(N)

        for i in np.arange(K):
            xTrain = x.iloc[folds != i,:]
            yTrain = y.iloc[folds != i]
            xTest = x.iloc[folds == i,:]
            model = SVC(gamma=gamVec.ravel()[iParam], C=costVec.ravel()[iParam])
            model.fit(xTrain, yTrain)
            yCVPrediction[folds == i] = model.predict(xTest)
        
        AUC.append(roc_auc_score(y, yCVPrediction))

    AUCGrid = np.array(AUC).reshape(gamVec.shape)

    indmax = np.argmax(AUC)
    return ("Max", np.max(AUC)), ("Gamma", gamVec.ravel()[indmax]), ("Cost", costVec.ravel()[indmax])

print(build_linear_model(data))
print(build_SVC_model(data))
# grid_search(data)
data.plot(kind="bar")
