from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,classification_report
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
import os
import pandas as pd
import numpy as np

dir=os.path.join(os.path.abspath(os.path.dirname(__file__)),"..","Data")
data=pd.read_csv(os.path.join(dir,"dataAfterPrep.csv"))
X=(data.drop(['Survived'],axis=1)).to_numpy()
y=(data['Survived']).to_numpy()


# X_ones=np.ones((X.shape[0],1))

# X=np.hstack((X_ones,X))   ---->mistake
#split --> poly --->scaling--->model
#______________________________Split_______________________________________

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)
X_train,X_valid,y_train,y_valid=train_test_split(X_train,y_train,test_size=0.1,random_state=42)

#______________________________Pipeline_______________________________________

pipeline=Pipeline(
    [('poly',PolynomialFeatures(degree=2,include_bias=False)),
     ("scaler",RobustScaler()),
     ("model",RidgeClassifier(10,max_iter=1000))]
)

pipeline.fit(X_train,y_train)
y_test_predict=pipeline.predict(X_test)
y_valid_predict=pipeline.predict(X_valid)

f1_test=f1_score(y_test,y_test_predict)
f1_valid=f1_score(y_valid,y_valid_predict)
print(f"classification report test ::- {classification_report(y_test,y_test_predict)}")
print(f"classification report Valid::- {classification_report(y_valid,y_valid_predict)}")
print(f"f1 score from test {f1_test}")
print(f"f1 score from valid {f1_valid}")

#try model with and without age

#try different combinations of fare Pclass and HasCabin

#-----------------------with age--------------------------
# f1 score from test 0.8181818181818182
# f1 score from valid 0.78125

#-------------------------without-------------------------------------
# f1 score from test 0.7761194029850746
# f1 score from valid 0.7692307692307693 -->age is slightly useful feature


#-------------------with fare and without pclass and hasCabin--------------------
# f1 score from test 0.8059701492537313
# f1 score from valid 0.6875


#-------------------with pclass and without fare and hasCabin--------------------
# f1 score from test 0.8181818181818182
# f1 score from valid 0.75

#-------------------with hasCabin and without fare and Pclass--------------------
# f1 score from test 0.7941176470588235
# f1 score from valid 0.7384615384615385  

# ------------------------- Conclusion of the ablation experiments -------------------------
# Compared: with/without Age, and different combinations of Fare / Pclass / hasCabin.
#
# Result: the differences between the experiments are small (about 1-4 F1 points)
# and they are not consistent between the test set and the validation set
# (e.g. Age: test diff ~0.04 vs valid diff ~0.01).
#
# Reason: the dataset is small, so test and valid have only ~40 samples each.
# Changing the prediction of 1-2 passengers is enough to move F1 by a few points,
# so these differences are within random noise.
#
# Decision: no feature is dropped or selected based on these numbers.
# These experiments were only for exploration/understanding, not for model selection.
#
# Possible future improvement (when I reach this topic in my roadmap):
# use Cross-validation instead of a single split to get more reliable comparisons.