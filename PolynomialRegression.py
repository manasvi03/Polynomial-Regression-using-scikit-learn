# -*- coding: utf-8 -*-
"""
@author: MANASVI

"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Position_Salaries.csv")

#plotting the curve
plt.scatter(data["Level"], data["Salary"])
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Level vs Salary")
#plt.show()          #it has a polynomial relation


X = data["Level"]       #Feature
Y = data["Salary"]      #Label

#Data Augmentation
pf = PolynomialFeatures(degree = 4)
X_poly = pf.fit_transform(X.values.reshape(-1,1))

#fitting the model
lr = LinearRegression()
lr.fit(X_poly, Y)

print(lr.score(X_poly, Y))



