import numpy as np
import mltools as ml
import mltools.dtree as mldt
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

Ytr = np.genfromtxt('data/Y_train.txt');
Xtr = np.genfromtxt('data/X_train.txt');
Xtest = np.genfromtxt('data/X_test.txt')
nFolds = 5;
depth = [1,7,8,10,14,15,20];
n_learners = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
testerr=[]
trainerr=[]

err = np.zeros(len(d))
for i,k in enumerate(n_learners):
    J = np.zeros(nFolds)
    for iFold in range(nFolds):
        Xti, Xvi, Yti, Yvi = ml.crossValidate(Xtr, Ytr, nFolds, iFold);
        gbm0 = GradientBoostingClassifier(n_estimators=k)
        gbm0.fit(Xti, Yti)
        mean_acc = gbm0.score(Xti, Yti)
        trainerr.append(1 - mean_acc);
        mean_acc = gbm0.score(Xvi, Yvi)
        J[iFold] = 1 - mean_acc
        testerr.append(1 - mean_acc);
plt.semilogy(depth,testerr,marker='o',color = 'r');
plt.semilogy(depth,trainerr,marker='o',color = 'b');
plt.xlabel(Number of learners)
plt.ylabel('MSE')
plt.show()

testAUC = []
trainAUC = []
err = np.zeros(len(d))
for i,k in enumerate(depth):
    J = np.zeros(nFolds)
    for iFold in range(nFolds):
        Xti, Xvi, Yti, Yvi = ml.crossValidate(Xtr, Ytr, nFolds, iFold);
        gbm0 = GradientBoostingClassifier(n_estimators=1000,max_depth=k)
        gbm0.fit(Xti, Yti)
        mean_acc = gbm0.score(Xti, Yti)
        trainerr.append(1 - mean_acc);
        mean_acc = gbm0.score(Xvi, Yvi)
        J[iFold] = 1 - mean_acc
        testerr.append(1 - mean_acc);
	     Yhat = gbm0.predict_proba(Xti)
        temp = roc_auc_score(Yti, Yhat[:, 1])
         trainAUC.append(temp);
         Yhat = gbm0.predict_proba(Xvi)
         temp = roc_auc_score(Yvi, Yhat[:, 1])
        testAUC.append(temp)

    testerr.append(np.mean(J))
    
plt.semilogy(depth,testerr,marker='o',color = 'r');
plt.semilogy(depth,trainerr,marker='o',color = 'b');
plt.xlabel('Depth')
plt.ylabel('MSE')
plt.show()

plt.semilogy(depth,testAUC,marker='o',color = 'r');
plt.semilogy(depth,trainAUC,marker='o',color = 'b');
plt.xlabel(AUC)
plt.ylabel('MSE')
plt.show()




