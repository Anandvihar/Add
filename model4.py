# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 17:28:57 2021

@author: Anandvihar
"""

import numpy as np
import pandas as pd
import pickle as p
#import sklearn
from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import fit_grid_point

Addition = {
    "a":['2','3','4'],
    "b":['1','4','7'],
    "Add":['3','7','11']    
    }

df = pd.DataFrame(Addition)
x = df.iloc[:,:2]
print(x)
y = df.iloc[:,-1]
print(y)

regressor = LinearRegression(())
regressor.fit(x,y)
#sklearn.model_selection.fit_grid_point(x,y)
pickle.dump(regressor,open('model4.pkl','wb'))
model = pickle.load(open('model4.pkl','rb'))



print(model.predict([[21,20]]))