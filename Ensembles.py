import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as NN
from sklearn.model_selection import StratifiedShuffleSplit as SS
from sklearn.model_selection import GridSearchCV as GS
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

Y = np.genfromtxt('data/Y_train.txt');
X = np.genfromtxt('data/X_train.txt');
Xte = np.genfromtxt('data/X_test.txt');

X, Y = ml.shuffleData(X, Y)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
Xte = scaler.transform(Xte)

Xtr = X[0:40000]
Ytr = Y[0:40000]
Xtest = X[180000:]
Ytest = Y[180000:]
clf = NN(activation='tanh', alpha = 0.1, hidden_layer_sizes=(80,))
gbm0 = RandomForestClassifier(n_estimators=300, max_depth=15, max_features=3)
clf.fit(Xtr, Ytr)
gbm0.fit(Xtr, Ytr)
YteNN = clf.predict_proba(Xtest)[:, 1];
YteRF = gbm0.predict_proba(Xtest)[:, 1];

wt = [0.6,0.65,0.7,0.75, 0.8]
auc=[]
for i,k in enumerate(wt):
    Yfinal = k * YteNN + (1-k) * YteRF;
    temp = roc_auc_score(Ytest, Yfinal)
    auc.append(temp)

print auc
plt.plot(wt,auc,marker='o');
plt.show()
#
# Yfinal = 0.6 * YteNN + 0.4 * YteRF;
#
# np.savetxt('ensembles2.txt',
#  np.vstack( (np.arange(len(Yfinal)), Yfinal)).T,
#  '%d, %.2f',header='ID,Prob1',comments='',delimiter=',');
