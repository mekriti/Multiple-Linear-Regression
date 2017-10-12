#Denver Neighborhoods

#The data (X1, X2, X3, X4, X5, X6, X7) are for each neighborhood
#X1 = total population (in thousands)
#X2 = % change in population over past several years
#X3 = % of children (under 18) in population
#X4 = % free school lunch participation
#X5 = % change in household income over past several years
#X6 = crime rate (per 1000 population)
#X7 = % change in crime rate over past several years
#Reference: The Piton Foundation, Denver, Colorado

def lin(X,y):
    y = np.array(y)
    X = np.array(X)
    y_train, y_test , X_train, X_test = train_test_split(y,X)
    print X_train.shape, X_test.shape,y_train.shape, y_test.shape
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    print lr.score(X_test,y_test)
    lr.coef_
    lr.intercept_
    pl.scatter(y_test,y_pred)
    pl.xlabel('Y_test')
    pl.ylabel('y_pred')
    pl.show()

    
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_excel("Denver.xls",header = 0)
# (df.columns)
df.describe()
df.corr()
#clearly,x2&x5,x3&x4,x4&x6,x5&x6 are highly correlated
#excluding one out of each pair.
X = df[['X2','X3', 'X6']]
X.shape
y = df[['X7']]
y.shape
lin (X,y)
X = df[['X2','X3', 'X4','X5']] #1
X.shape
lin(X,y)
X = df[['X1','X3', 'X5','X6']] #2
X.shape
lin(X,y)
X = df[['X1','X2','X3','X4','X5','X6']] #3
lin(X,y)
X = df[['X2','X4','X5']] #4
lin(X,y)

