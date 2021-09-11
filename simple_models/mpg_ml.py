from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd 
from matplotlib.pyplot import plot
from shared.data_utility import *

data = pd.read_csv("data/auto-mpg.csv")
data = data.join(cat_to_num(data.origin))
data = data.drop('origin', axis=1)

dataTrain, dataTest = build_train_test_data(data, 0.8)
model = LinearRegression()
model.fit(dataTrain.drop('mpg', axis=1), dataTrain.mpg)
print(model.score(dataTest.drop('mpg', axis=1), dataTest.mpg))
prediction = model.predict(dataTest.drop('mpg', axis=1))

# plot(data.weight, data.mpg, 'o')
# plot(data.horsepower, data.mpg, 'o')
# plot(data.displacement, data.mpg, 'o')

plot(dataTest.mpg, prediction, 'o')
x = np.linspace(10, 40, 5)
plot(x, x, '-')
