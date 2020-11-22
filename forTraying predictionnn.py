import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
df=pd.read_csv("col.csv")
imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
df.iloc[:, 3:] = imputer.fit_transform(df.iloc[:, 3:])
df
df.isnull().sum()
X = df.iloc[:,3:-2].values
X
y = df.iloc[:,-2:].values
y.reshape(-1,1)
y = df.iloc[:,12]
X = df.iloc[:,3:12] 
regr=LinearRegression(fit_intercept=True, normalize=True,copy_X=True,n_jobs=-1).fit(X,y)
regr.predict(X.iloc[100:,:])
round(regr.score(X,y),4)
SVM = svm.LinearSVC().fit(X,y)
SVM.predict(X.iloc[100:,:])
round(SVM.score(X,y), 4)
LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')
LR.fit(X,y)
LR.predict(X.iloc[100:,:]) # prediction value
round(LR.score(X,y),4) # accuracy to first 4 number
NN = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(1, 5), random_state=0)
NN.fit(X, y)
NN.predict(X.iloc[100:,:]) # prediction value
round(NN.score(X,y), 4)
LR = LogisticRegression(random_state=5, solver='lbfgs', multi_class='ovr')
LR.fit(X, y)
LR.predict(X.iloc[100:,:])
round(LR.score(X,y),4) 
RF = RandomForestClassifier(n_estimators=60, max_depth=25 , random_state=3)
RF.fit(x,y)
RF.predict(X_test)
round(RF.score(x,y),4)