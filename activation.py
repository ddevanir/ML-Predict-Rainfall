import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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

activation = ['identity', 'tanh', 'logistic','relu']
test_err = []
train_err = []

for i,k in enumerate(activation):
     gbm0 = MLPClassifier(activation= k, alpha = 0.1, hidden_layer_sizes = (80,))
     gbm0.fit(Xtr, Ytr)
     train_acc = gbm0.score(Xtr, Ytr)
     train_err.append(1 - train_acc)
     test_acc = gbm0.score(Xtest, Ytest)
     test_err.append(1 - test_acc)
     print "activation =",k," testerr = ",(1 - test_acc), " trainerr = ",(1 - train_acc)

plt.semilogy(activation, train_err, marker = 'o', color = 'b')
plt.semilogy(activation, test_err, marker = 'o', color = 'r')
plt.xlabel('Activation')
plt.ylabel('Validation Error')
plt.show()


