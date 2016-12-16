import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as NN
from sklearn.model_selection import StratifiedShuffleSplit as SS
from sklearn.model_selection import GridSearchCV as GS
from sklearn.preprocessing import StandardScaler

Y = np.genfromtxt('data/Y_train.txt');
X = np.genfromtxt('data/X_train.txt');
Xte = np.genfromtxt('data/X_test.txt');

X, Y = ml.shuffleData(X, Y)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
Xte = scaler.transform(Xte)

clf = NN(activation='tanh', alpha = 0.1, hidden_layer_sizes=(80,))

# alpha_range = np.logspace(-5, 5, 11)
# activation_range = ['logistic', 'tanh','identity','relu']
#hidden_range = [(),()]

# param_grid = dict(alpha=alpha_range, activation=activation_range)
# cv = SS(n_splits=5, test_size=0.2, random_state=42)
# grid = GS(NN(), param_grid=param_grid, cv=cv)
#grid.fit(X, Y)
# print("The best parameters are %s with a score of %0.2f"
#      % (grid.best_params_, grid.best_score_))


# clf = NN(activation='logistic', alpha = 0.001, hidden_layer_sizes=(80, 80))
clf.fit(X, Y)
# = clf.score(X[180000:], Y[180000:])
Yte = clf.predict_proba(Xte)[:, 1];
#print score

np.savetxt('neural_networks11.txt',
 np.vstack( (np.arange(len(Yte)), Yte)).T,
 '%d, %.2f',header='ID,Prob1',comments='',delimiter=',');

