import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

user= pd.read_csv('signatureData.csv', sep=',',header=0) # read data
print('headar : ',user. head())
h_train=user[:75] # train
k_test=user[:15] # test
print('h_train : ',h_train)
print('k_test : ',k_test)

# test
k1_test = k_test.iloc[:,0]
k2_test = k_test.iloc[:,1:]

# train
h1_tr = h_train.iloc[:,0] 
h2_tr = h_train.iloc[:,1:] 

#use Logistic Regression to prediction
LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(h1_tr, h2_tr)
LR.predict(k_test) # predict value
round(LR.score(k_test,k_test), 4) # accuracy
SVM = svm.SVC(decision_function_shape="ovo").fit(h1_tr, h2_tr)
SVM.predict(k_test) # predict value
round(SVM.score(k_test, k_test), 4) # accuracy

# use Random Forests to prediction
RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(h1_tr, h2_tr)
RF.predict(k_test)# predict value
round(RF.score(k_test, k_test), 4) # accuracy

# use Neural Networks to prediction
NN = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(250, 80), random_state=0).fit(h1_tr, h2_tr)
NN.predict(k_test)# predict value
round(NN.score(k_test, k_test), 4) # accuracy

