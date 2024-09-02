#!/usr/bin/env python
# coding: utf-8

# In[334]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier


# In[335]:


dataset=pd.read_csv("titanic.csv")


# In[336]:


dataa=pd.read_csv("titanic.csv")


# In[337]:


dataa.head(1)


# In[ ]:





# In[338]:


pd.set_option('display.max_rows', None)

dataset.head(None)


# In[339]:


dataset=dataset.drop(columns=["Name","PassengerId","Cabin","Ticket"])


# In[340]:


dataset=dataset[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Survived"]]


# In[ ]:





# In[341]:


le=LabelEncoder()
le.fit(dataset["Sex"])
dataset["Sex"]=le.transform(dataset["Sex"])


# In[342]:


le1=LabelEncoder()
le1.fit(dataset["Embarked"])
dataset["Embarked"]=le1.transform(dataset["Embarked"])


# In[343]:


dataset.info()


# In[344]:


dataset.isnull().sum()


# In[345]:


dataset.shape


# In[346]:


dataset["Age"]=dataset["Age"].fillna(dataset["Age"].mean())


# In[347]:


dataset["Age"]=dataset["Age"].astype("int32")


# In[ ]:





# In[348]:


plt.figure(figsize=[10,40])
numeric_columns = dataset.select_dtypes(include='number').columns.tolist()
sns.boxplot(data=dataset[numeric_columns])
plt.show()


# In[349]:


sns.scatterplot(data=dataset,x="Age",y="Sex",hue="Survived")
plt.show()


# In[350]:


sns.lineplot(data=dataset,y="Age",x="Pclass")
plt.show()


# In[ ]:





# In[ ]:





# In[351]:


sns.heatmap(dataset.corr(),annot=True,)
plt.show()


# In[352]:


sns.kdeplot(dataset["Age"])
plt.show()


# In[353]:


sns.boxplot(x="Age",data=dataset)
plt.show()


# In[354]:


q1=np.percentile(dataset["Age"],25)
q3=np.percentile(dataset["Age"],75)
q1,q3


# In[355]:


iqr=q3-q1


# In[356]:


max_r=q3+(1.5*iqr)
min_r=q1-(1.5*iqr)
max_r,min_r


# In[357]:


dataset.loc[dataset["Age"]>max_r ,"Age"] =max_r


# In[358]:


sns.kdeplot(dataset["Fare"])
plt.show()


# In[359]:


sns.boxplot(x="Fare",data=dataset)
plt.show()


# In[360]:


q1=np.percentile(dataset["Age"],25)
q3=np.percentile(dataset["Age"],75)
q1,q3


# In[361]:


iqr=q3-q1


# In[362]:


dataset.loc[dataset["Fare"]>max_r ,"Fare"] =max_r


# In[ ]:





# In[ ]:





# In[363]:


sns.kdeplot(dataset["Parch"])
plt.show()


# In[364]:


sns.boxplot(x="Parch",data=dataset)
plt.show()


# In[365]:


q1=np.percentile(dataset["Parch"],25)
q3=np.percentile(dataset["Parch"],75)
q1,q3


# In[366]:


iqr=q3-q1
dataset.loc[dataset["Parch"]>max_r ,"Parch"] =max_r


# In[ ]:





# In[367]:


sns.kdeplot(dataset["SibSp"])
plt.show()


# In[368]:


sns.boxplot(x="SibSp",data=dataset)
plt.show()


# In[369]:


q1=np.percentile(dataset["SibSp"],25)
q3=np.percentile(dataset["SibSp"],75)
q1,q3


# In[370]:


iqr=q3-q1
dataset.loc[dataset["SibSp"]>max_r ,"SibSp"] =max_r


# In[371]:


dataset.head(1)


# In[372]:


scp=StandardScaler()
scp.fit(dataset[["Pclass"]])
# dataset["Pclass"]=scp.transform(dataset[["Pclass"]])


# In[373]:


scp=StandardScaler()
scp.fit(dataset[["Sex"]])
dataset["Sex"]=scp.transform(dataset[["Sex"]])


# In[374]:


scp=StandardScaler()
scp.fit(dataset[["Age"]])
dataset["Age"]=scp.transform(dataset[["Age"]])


# In[375]:


scp=StandardScaler()
scp.fit(dataset[["SibSp"]])
dataset["SibSp"]=scp.transform(dataset[["SibSp"]])


# In[376]:


scp=StandardScaler()
scp.fit(dataset[["Parch"]])
dataset["Parch"]=scp.transform(dataset[["Parch"]])


# In[377]:


scp=StandardScaler()
scp.fit(dataset[["Fare"]])
dataset["Fare"]=scp.transform(dataset[["Fare"]])


# In[378]:


scp=StandardScaler()
scp.fit(dataset[["Embarked"]])
dataset["Embarked"]=scp.transform(dataset[["Embarked"]])


# In[379]:


x=dataset.iloc[:,:-1]


# In[380]:


y=dataset["Survived"]


# In[381]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.3)


# In[382]:


lr=LinearRegression()
lr.fit(x_train,y_train)


# In[383]:


lr.score(x_test,y_test)*100,lr.score(x_train,y_train)*100


# In[384]:


lrl={"penelty":['l1', 'l2', 'elasticnet']}


# In[385]:


# gdl=GridSearchCV(LogisticRegression(),lrl)
# gdl.fit(x_train,y_train)


# In[386]:


# x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)


# In[387]:


lr1=LogisticRegression()
lr1.fit(x_train,y_train)


# In[388]:


lr1.score(x_test,y_test)*100,lr1.score(x_train,y_train)*100


# In[ ]:





# In[389]:


dataset["lr1"]=lr1.predict(x)
# lr1.predict([[0.827377,-1.355574,-0.262975,-0.474545,-0.473674,-0.833816,0.581114]])


# In[390]:


dtl={"criterion":["gini", "entropy", "log_loss"],"splitter":["best", "random"],"max_depth":[i for i in range(1,20)]}


# In[391]:


gdd=GridSearchCV(DecisionTreeClassifier(),dtl)
gdd.fit(x_train,y_train)


# In[392]:


gdd.best_estimator_


# In[393]:


dt=DecisionTreeClassifier(criterion="entropy",max_depth=3)
dt.fit(x_train,y_train)


# In[394]:


dt.score(x_test,y_test)*100,dt.score(x_train,y_train)*100


# In[395]:


knl={"weights":['uniform', 'distance'],"algorithm":['auto', 'ball_tree', 'kd_tree', 'brute'],"n_neighbors":[i for i in range(1,20)]}


# In[396]:


knd=GridSearchCV(KNeighborsClassifier(),knl)
knd.fit(x_train,y_train)
    


# In[397]:


knd.best_estimator_


# In[398]:


kn=KNeighborsClassifier(n_neighbors=14)
kn.fit(x_train,y_train)
kn.score(x_test,y_test)*100,kn.score(x_train,y_train)*100


# In[399]:


nl=[]
for i in range(1,20):
    kn=KNeighborsClassifier(n_neighbors=i)
    kn.fit(x_train,y_train)
    print(i,kn.score(x_test,y_test)*100,kn.score(x_train,y_train)*100)
    nl.append((kn.score(x_train,y_train)*100)-(kn.score(x_test,y_test)*100))
    
    


# In[400]:


(nl)


# In[ ]:





# In[401]:


kn.score(x_test,y_test)*100,kn.score(x_train,y_train)*100


# In[402]:


kn.predict([[0.827377,-1.355574,-0.262975,-0.474545,-0.473674,-0.833816,0.581114]])


# In[403]:


# dataset["new_predict"]=kn.predict(x)


# In[ ]:





# In[404]:


rf1={"criterion":["gini", "entropy", "log_loss"]}


# In[405]:


gdr=GridSearchCV(RandomForestClassifier(),rf1)
gdr.fit(x_train,y_train)


# In[406]:


gdr.best_estimator_


# In[ ]:





# In[407]:


rf=RandomForestClassifier(criterion="log_loss")
rf.fit(x_train,y_train)


# In[408]:


rf.score(x_test,y_test)*100,rf.score(x_train,y_train)*100


# In[409]:


d={"kernel":['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],"gamma":['scale']}


# In[ ]:





# In[410]:


# gd=GridSearchCV(SVC(),d)
# gd.fit(x_train,y_train)


# In[411]:


sv=SVC()
sv.fit(x_train,y_train)


# In[412]:


sv.score(x_test,y_test)*100,sv.score(x_train,y_train)*100


# In[413]:


gn=GaussianNB()
gn.fit(x_train,y_train)


# In[414]:


gn.score(x_test,y_test)*100,gn.score(x_train,y_train)*100


# In[415]:


gb=GradientBoostingClassifier()
gb.fit(x_train,y_train)


# In[416]:


gb.score(x_test,y_test)*100,gb.score(x_train,y_train)*100


# In[ ]:





# In[ ]:





# In[417]:


base_model = KNeighborsClassifier()

bagging_model = BaggingClassifier(estimator=base_model, n_estimators=50, random_state=42)
bagging_model.fit(x_train, y_train)


# In[418]:


bagging_model.score(x_test,y_test)*100,bagging_model.score(x_train,y_train)*100


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




