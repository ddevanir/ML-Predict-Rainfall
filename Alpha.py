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

alpha = [1e-05, 0.0001, 0.001, 0.01, 0.1]
test_err = []
train_err = []

for i,k in enumerate(alpha):
     gbm0 = MLPClassifier(activation='tanh', alpha = k, hidden_layer_sizes = (80,))
     gbm0.fit(Xtr, Ytr)
     Yhat = gbm0.predict_proba(Xtr);
     temp = roc_auc_score(Ytr, Yhat[:, 1])
     train_err.append(temp);
     Yhat = gbm0.predict_proba(Xtr)
     temp1 = roc_auc_score(Ytr, Yhat[:, 1])
     test_err.append(temp1)
     print "Alpha =",k," testerr = ",temp, " trainerr = ",temp1;


print "Score for testing", test_err
print "Score for training", train_err
print "Alpha", alpha
plt.semilogy(alpha, train_err, marker = 'o', color = 'b')
plt.semilogy(alpha, test_err, marker = 'o', color = 'r')
plt.xlabel('Alpha')
plt.ylabel('Area under curve')
plt.show()
