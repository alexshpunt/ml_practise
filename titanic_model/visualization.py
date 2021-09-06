from operator import index
from pandas.core.frame import DataFrame
from titanic_model.titanic_data import *
import matplotlib as plt
from statsmodels.graphics.mosaicplot import mosaic 
import statsmodels.api as sm 

vData = DataFrame(data) 
# vData['Survived'].replace({0:'No', 1:'Yes'}, inplace = True)

def t(x):
    return f"xx{x}"
vData["PassengerId"].apply(np.sqrt)
print(vData)

def visualize_sex_survived(data):
    crosstable = pd.crosstab(data['Sex'], data['Survived'])
    return mosaic(data, ['Sex', 'Survived'])

def visualize_cabin_survived(data):
    crosstable = pd.crosstab(data['Pclass'], data['Survived'], normalize=True)
    print(crosstable)
    # print(crosstable.loc['1', "No"])
    def get_labels(key): 
        print(key)
        print(crosstable.loc[key])
        pass 

    return mosaic(data, ['Pclass', 'Survived'])

# visualize_sex_survived(vData)
# visualize_cabin_survived(vData)