import numpy as np
import mltools as ml
import mltools.dtree as mldt
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

n_learners = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
testerr=[]
trainerr=[]
for i,k in enumerate(n_learners):
    J = np.zeros(nFolds)
    for iFold in range(nFolds):
      Xti, Xvi, Yti, Yvi = ml.crossValidate(Xtr, Ytr, nFolds, iFold);
      gbm0 = RandomForestClassifier(n_estimators=k, max_features=3)
      gbm0.fit(Xtri ,Ytri)
      Yhat = gbm0.predict(Xtri)
      temp = gbm0.score(Ytri, Yhat)
      trainerr.append(temp);
      Yhat = gbm0.predict(Xvi)
      temp = roc_auc_score(Yvi, Yhat
      testerr.append(temp)

plt.plot(depth,testerr,label="Test MSE",color='r')
plt.plot(depth,trainerr,label="Train MSE",color='b')
plt.xlabel("Depth")
plt.ylabel("AUC")




depth = [1,5,10,15,20,25,30];
testerr=[]
trainerr=[]
for i,k in enumerate(depth):
    J = np.zeros(nFolds)
    for iFold in range(nFolds):
      Xti, Xvi, Yti, Yvi = ml.crossValidate(Xtr, Ytr, nFolds, iFold);
      gbm0 = RandomForestClassifier(n_estimators=300,max_depth=k,max_features=3)
      gbm0.fit(Xtri ,Ytri)
      Yhat = gbm0.predict_proba(Xtri)
      temp = roc_auc_score(Ytri, Yhat[:, 1])
      trainerr.append(temp);
      Yhat = gbm0.predict_proba(Xvi)
      temp = roc_auc_score(Yvi, Yhat[:, 1]
      testerr.append(temp)
      Yhat = gbm0.predict(Xtri)
      temp = gbm0.score(Ytri, Yhat)
      trainerr.append(temp);
      Yhat = gbm0.predict(Xvi)
      temp = roc_auc_score(Yvi, Yhat
      testerr.append(temp)

plt.plot(depth,testerr,label="Test MSE",color='r')
plt.plot(depth,trainerr,label="Train MSE",color='b')
plt.xlabel("Depth")
plt.ylabel("AUC")

plt.plot(depth,testerr,label="Test AUC",color='r')
plt.plot(depth,trainerr,label="Train AUC",color='b')
plt.xlabel("Depth")
plt.ylabel("AUC")