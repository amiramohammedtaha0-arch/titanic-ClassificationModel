import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

dir=os.path.join(os.path.abspath(os.path.dirname(__file__)),"..","Data")
df=pd.read_csv(os.path.join(dir,"train.csv"))
#-------------------------------data quality----------------------------------
#1Completeness
#2-uniquness
#3-timelines
#4-Accurecy
#5-Consistency
#6-validation

#-----------------------------------Completeness---------------------------------
print(df.isna().sum())
print(df.info()) #most capin is nan so i



#relationship between cabin and surviving

# df['hasCabin']=df['Cabin'].na.astype(int)



# 1) عمود يوضح هل الراكب عنده كابينة ولا لأ
df['hasCabin'] = df['Cabin'].notna().astype(int)

# 2) نسبة النجاة لكل مجموعة
survival_rate = df.groupby('hasCabin')['Survived'].mean()
print(survival_rate)

# 3) الرسم
survival_rate.index = ['No Cabin', 'Has Cabin']
survival_rate.plot(kind='bar', color=['tomato', 'seagreen'])
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Cabin Availability')
plt.xticks(rotation=0)
plt.show()



relation_between_Pclass_and_cabin=df.groupby("Pclass")['hasCabin'].mean()  #is that reducndant information

print(relation_between_Pclass_and_cabin)


#2 cat 1-2 1-3 means chi2 test
contingencyTable=pd.crosstab(df["hasCabin"],df["Pclass"])
chi2_cabin_pclass=stats.chi2_contingency(contingencyTable)
print(chi2_cabin_pclass)
# Chi2ContingencyResult(statistic=np.float64(238.40460966993172), pvalue=np.float64(1.7025378668863945e-52), dof=2, expected_freq=array([[ 83.70574163,  72.75358852, 170.54066986],
#        [ 23.29425837,  20.24641148,  47.45933014]]))

#there is signigicant relationship between Cabin and Pclass


df=df.drop(["Cabin"],axis=1)
print(df[df['Fare'].isna()]) # index 152
# df.drop(152,inplace=True)

sns.histplot(df['Age'],kde=True) #right skewed use median instead of mean

plt.show()

df['Age']=df['Age'].fillna(df['Age'].median())
print(df.isna().sum())


#-------------------------------------------------------------------------

df['Sex']=(df['Sex']=="female").astype(int)

df['familySize']=df['SibSp']+df['Parch']+1

df=df.drop(["Name","PassengerId","Ticket","SibSp","Parch"],axis=1)  #not important

df=pd.get_dummies(df,columns=["Embarked"],drop_first=True)

df.to_csv(os.path.join(dir,"dataAfterPrep.csv"),index=False)

df['Embarked_S']=df['Embarked_S'].astype(int)
df['Embarked_Q']=df['Embarked_Q'].astype(int)
print(df.info())
# sns.heatmap(df.corr(),annot=True)
# plt.show()

#try model with and without age

#try different combinations of fare Pclass and HasCabin

