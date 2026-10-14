import pandas as pd
import os
import numpy as np
from sklearn.metrics import f1_score


dir=os.path.join(os.path.dirname(__file__),"..","data")
df=pd.read_csv(os.path.join(dir,"dataAfterPrep.csv"))
X=df.drop("Survived",axis=1)
y=df["Survived"]
X[['Fare','familySize','Age']]=(X[['Fare','familySize','Age']]-X[['Fare','familySize','Age']].mean())/(X[['Fare','familySize','Age']]).std()
X = X.to_numpy(dtype=float)
y = y.to_numpy(dtype=float).reshape(-1,1)

X_ones=np.ones((X.shape[0],1))
X=np.hstack((X_ones,X))
print(X)


m=X.shape[0]
alpha=0.1
theta_Vector=np.zeros((X.shape[1],1))
for i in range(1000):
    lr_h=X@theta_Vector
    p_hat=1/((1+np.exp(-(lr_h))))  #h(x) step2
    p_hat=np.clip(p_hat, 1e-15, 1 - 1e-15)
    e=p_hat-y #error
    cost= (-1/m)*np.sum((y*np.log(p_hat))+((1-y)*np.log(1-p_hat)))#1/-m * sum(y(i)*log(p^)+(1-y(i))*log(1-p^))  step3
    grad_vector=(1/m)*(X.T@e) #step 4
    theta_Vector=theta_Vector-(alpha*grad_vector)


lr_h=X@theta_Vector #last t
p_hat=1/((1+np.exp(-(lr_h)))) #last p_hat
y_hat=(p_hat>=0.5).astype(int)

f1=f1_score(y,y_hat)
print(f1)