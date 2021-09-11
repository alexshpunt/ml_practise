from operator import index
from numpy.core.fromnumeric import std
from pandas.core.frame import DataFrame
from pandas.core.reshape.pivot import crosstab
from titanic_model.titanic_data import *
import matplotlib as plt
from statsmodels.graphics.mosaicplot import mosaic 
import statsmodels.api as sm 
import numpy as np 

# vData = DataFrame(data) 
vData = data 
# vData['Survived'].replace({0:'No', 1:'Yes'}, inplace = True)
# vData['Pclass'] = vData['Pclass'].apply(str)

def visualize_sex_survived(data):
    crosstable = pd.crosstab(data['Sex'], data['Survived'])
    return mosaic(data, ['Sex', 'Survived'])

def visualize_cabin_survived(data):
    y,x = data['Survived'], data['Pclass']
    x = sm.add_constant(x) 
    model = sm.OLS(y, x).fit()

    influence = model.get_influence() 
    stdResiduals = influence.resid_studentized_internal

    print(x,y)

    crosstable = pd.crosstab(data['Pclass'],stdResiduals)

    df = DataFrame({'Pclass':data['Pclass'], 'Residuals':stdResiduals})

    return crosstable, pd.crosstab(data['Pclass'], data['Survived'])
    return mosaic(df, ['Pclass', 'Residuals'])


    # crosstable = pd.crosstab(data['Pclass'], data['Survived'], normalize=True)
    # return mosaic(data, ['Pclass', 'Survived'], labelizer = lambda k: f"{np.round(crosstable.loc[k] * 100.0)}%")

# visualize_sex_survived(vData)
visualize_cabin_survived(vData)