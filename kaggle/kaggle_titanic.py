import numpy as np
import pandas as pd 

from os.path import join
data_path = 'data/kaggle/titanic/'
data = pd.read_csv(join(data_path, 'train.csv'))

def prepare_data(data):
    import math 
    def cat_to_num(data, cat_name):
        return pd.get_dummies(data, prefix=[cat_name], columns=[cat_name])
    data = cat_to_num(data, 'Sex')
    data = cat_to_num(data, 'Embarked')
    data.Fare = data.Fare.fillna(0)
    data['Sqr_Fare'] = data['Fare'].apply(math.sqrt)
    data = data.drop(['Fare', 'Name', 'Cabin', 'Ticket'], axis=1)

    import statistics
    mean = math.floor(statistics.mean(data['Age'].dropna()))
    data['Age'] = data['Age'].fillna(mean).apply(math.floor)
    return data 

from sklearn.model_selection import train_test_split
data = prepare_data(data).drop('PassengerId', axis=1)
split_data = train_test_split(data.drop('Survived', axis=1), data.Survived)
train_data, test_data, target_train_data, target_test_data = split_data

from utilities import measure_model
from sklearn.ensemble import GradientBoostingClassifier
m = GradientBoostingClassifier(n_estimators=100)
m.fit(train_data, target_train_data)
print(measure_model(m, test_data, target_test_data, color='b'))

submission_data = pd.read_csv(join(data_path, 'test.csv'))
passenger_ids = submission_data.PassengerId
submission_data = prepare_data(submission_data)
survived = m.predict(submission_data.drop('PassengerId', axis=1))

submission_data = pd.DataFrame()
submission_data['PassengerId'] = passenger_ids
submission_data['Survived'] = survived
submission_data.to_csv(join(data_path, 'submission.csv'), index=False)

