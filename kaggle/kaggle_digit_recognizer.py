import numpy as np
import pandas as pd 

from os.path import join
data_path = 'data/kaggle/digit-recognizer/'
data = pd.read_csv(join(data_path, 'train.csv'))

from sklearn.model_selection import train_test_split
split_data = train_test_split(data.drop('label', axis=1), data.label)
train_data, test_data, target_train_data, target_test_data = split_data

from sklearn.neighbors import KNeighborsClassifier
m = KNeighborsClassifier()
m.fit(train_data, target_train_data)

from utilities import measure_model
# print(measure_model(m, test_data, target_test_data, multiclass=True))

data = pd.read_csv(join(data_path, 'test.csv'))
pred = m.predict(data)
df = pd.DataFrame()
df['ImageId'] = range(1, len(pred)+1)
df['Label'] = pred 
df.to_csv(join(data_path, 'submission.csv'), index=False)