import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

X = np.genfromtxt('data/X_train.txt');
Y = np.genfromtxt('data/Y_train.txt');
Xte = np.genfromtxt('data/X_test.txt');

X, Y = ml.shuffleData(X, Y)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
Xte = scaler.transform(Xte)

Xtr = X[0:120000]
Ytr = Y[0:120000]
Xtest = X[130000:]
Ytest = Y[130000:]

hide = [(10,),(20,),(30,),(40,),(50,),(60,),(70,),(80,),(90,),(100,)]
test_err = []
train_err = []

for i,k in enumerate(hide):
     gbm0 = MLPClassifier(activation='tanh', hidden_layer_sizes = k)
     gbm0.fit(Xtr, Ytr)
     Yhat = gbm0.predict_proba(Xtr);
     temp = roc_auc_score(Ytr, Yhat[:, 1])
     train_err.append(temp);
     Yhat = gbm0.predict_proba(Xtest)
     temp1 = roc_auc_score(Ytest, Yhat[:, 1])
     test_err.append(temp1)
     print "hidden neurons =",k," testerr = ",temp, " trainerr = ",temp1

print "Score of train data", test_err;
print "Score of test data", train_err;
plt.semilogy(hide, train_err, marker = 'o', color = 'b');
plt.semilogy(hide, test_err, marker = 'o', color = 'r');
plt.xlabel('Hidden neurons')
plt.ylabel('Area under curve')
plt.show()



