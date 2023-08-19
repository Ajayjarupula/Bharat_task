#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Prediction

# # Import libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("C:/Users/admin/Desktop/QualityPrediction.csv")
df.head()


# In[3]:


df.info()


# In[5]:


df.isnull().sum()


# In[7]:


df.describe()


# In[9]:


df.quality.value_counts()


# In[10]:


df.quality.value_counts()/len(df)*100


# In[11]:


def get_distplot(col):
    ax = sns.distplot(df[col], bins = 6)
    Q1 = np.percentile(df[col],25)
    Q3 = np.percentile(df[col],75)
    IQR=Q3-Q1    
    lower_threshold = Q1 - 1.5*IQR
    upper_threshold = Q3 + 1.5*IQR
    
    ax.axvline(Q1, color='red', linestyle='-', label="Q1")
    ax.axvline(Q3, color='blue', linestyle='-', label="Q3")
    ax.axvline(lower_threshold, color='yellow', linestyle='-', label="Lower threshold")
    ax.axvline(upper_threshold, color='green', linestyle='-', label="Upper threshold")
    ax.legend()


# In[13]:


fig = plt.figure(figsize=(18,6))

ax1 = fig.add_subplot(121)
ax1 = sns.countplot(df['quality'])
ax2 = fig.add_subplot(122)
ax2 = get_distplot('quality')
plt.show()


# # Univariate Exploaration

# In[15]:


for i in df.columns:
    f, (ax1) = plt.subplots(1,1,figsize=(5,5))
    ax1 = get_distplot(i)


# In[16]:


# create box plots
fig, ax = plt.subplots(ncols=4, nrows=3, figsize=(12,12))
index = 0
ax = ax.flatten()

for col in df.columns:
    sns.boxplot(y=col, data=df, ax=ax[index], color='r')
    plt.subplots_adjust(wspace = .5)
    index += 1


# # Bivariate Exploration

# In[17]:


x = df.drop(columns="quality")
fig, axs = plt.subplots(4,3, figsize=(20,20))
fig.patch.set_facecolor('white')
attributes = x.columns
att = 0
for i in range(4):
    for j in range(3):
        try:
            sns.barplot(x="quality", y=attributes[att], data=df, estimator=np.mean, ax=axs[i][j])
        except: #to handle index value 11 
            print()
        att += 1


# # Correlation Coefficient

# In[20]:


c=df.corr()
c


# In[21]:


plt.figure(figsize=(10, 8))
sns.heatmap(c, cmap = 'coolwarm', annot = True)


# In[22]:


dfML = df.copy()
dfML.head()


# In[23]:


bins = [0, 6, 10]
labels = ["Poor","Good"]
dfML['quality'] = pd.cut(dfML['quality'], bins=bins, labels=labels)


# In[24]:


label_quality = LabelEncoder()
dfML['quality'] = label_quality.fit_transform(dfML['quality'])
dfML['quality'].value_counts()


# In[25]:


X = dfML.drop('quality', axis = 1)
y = dfML['quality']


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[27]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# # Apply ML Models

# In[28]:


lst_model = []
lst_accuracy  = []
lst_accuracy_train = []
lst_accuracy_test = [] 
lst_cv_score = []
lst_TP = []
lst_TN = []
lst_FP = []
lst_FN = []

def applyMLmodel(model):
    # train the model
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test) * 100
    lst_accuracy.append(accuracy)
    print("Accuracy :", accuracy)
    
    # cross-validation , y_train.ravel() is similar to y_train.reshape(-1)
    cv = cross_val_score(estimator = model, X = X_train, y = y_train.ravel(), cv = 10)
    lst_cv_score.append(cv.mean())
    print("CV Score :", cv.mean())
    
    # predicting accuracy for training data set
    y_pred_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    lst_accuracy_train.append(accuracy_train)
    print("Accuracy(Training) :", accuracy_train)

    # predicting accuracy for test data set
    y_pred_test = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    lst_accuracy_test.append(accuracy_test)
    print("Accuracy(Test) :", accuracy_test)
    
     # confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print("Confusion Matrix :")
    print(cm)

    # storing TN,TP,FN and FP as a part of list
    lst_TN.append(cm[0,0])
    lst_FP.append(cm[0,1])
    lst_FN.append(cm[1,0])
    lst_TP.append(cm[1,1])


# In[29]:


model = LogisticRegression()
applyMLmodel(model) 
lst_model.append("LogisticRegression")


# In[30]:


model = DecisionTreeClassifier()
applyMLmodel(model)
lst_model.append("DecisionTreeClassifier")


# In[31]:


model = DecisionTreeClassifier()
applyMLmodel(model)
lst_model.append("DecisionTreeClassifier")


# In[32]:


model =KNeighborsClassifier()  
applyMLmodel(model)
lst_model.append("KNeighborsClassifier")


# In[33]:


model = GaussianNB()
applyMLmodel(model)
lst_model.append("GaussianNB")


# In[34]:


predictiondf = pd.DataFrame({'Model': np.array(lst_model),
                             'Accuracy': np.array(lst_accuracy),
                             'Accuracy(Training)' : np.array(lst_accuracy_train),
                             'Accuracy(Test)' : np.array(lst_accuracy_test),
                             'CV Score' : np.array(lst_cv_score),
                             'True Positive' : np.array(lst_TP),
                             'True Negative' : np.array(lst_TN),
                             'False Positive' : np.array(lst_FP),
                             'False Negative' : np.array(lst_FN)
                            })
predictiondf


# In[35]:


fig, ax = plt.subplots(2,2, figsize=(24,14))
plt.subplots_adjust(wspace = .25, hspace = .25)
#comparing CV score
predictiondf.sort_values(by=['CV Score'], ascending=False, inplace=True)

sns.barplot(x='CV Score', y='Model', data = predictiondf, ax = ax[0][0])
ax[0][0].set_xlabel('Cross-Validaton Score', size=16)
ax[0][0].set_ylabel('Model')
ax[0][0].set_xlim(0,1.0)
ax[0][0].set_xticks(np.arange(0, 1.1, 0.1))

#comparing accuracy
predictiondf.sort_values(by=['Accuracy'], ascending=False, inplace=True)

sns.barplot(x='Accuracy', y='Model', data = predictiondf, ax = ax[0][1])
ax[0][1].set_xlabel('Accuracy', size=16)
ax[0][1].set_ylabel('Model')
#comparing accuracy(training)
predictiondf.sort_values(by=['Accuracy(Training)'], ascending=False, inplace=True)

sns.barplot(x='Accuracy(Training)', y='Model', data = predictiondf, palette='Blues_d', ax = ax[1][0])
ax[1][0].set_xlabel('Accuracy(Training)', size=16)
ax[1][0].set_ylabel('Model')
ax[1][0].set_xlim(0,1.0)
ax[1][0].set_xticks(np.arange(0, 1.1, 0.1))

#comparing accuracy(testing)
predictiondf.sort_values(by=['Accuracy(Test)'], ascending=False, inplace=True)

sns.barplot(x='Accuracy(Test)', y='Model', data = predictiondf, palette='Reds_d', ax = ax[1][1])
ax[1][1].set_xlabel('Accuracy(Test)', size=16)
ax[1][1].set_ylabel('Model')
ax[1][1].set_xlim(0,1.0)
ax[1][1].set_xticks(np.arange(0, 1.1, 0.1))

plt.show()


# In[40]:


# Comparing how many false prediction made by each model
predictiondf.sort_values(by=(['Accuracy(Test)']), ascending=True, inplace=True)

f, axe = plt.subplots(1,1, figsize=(5,5))
sns.barplot(x = predictiondf['Model'], y=predictiondf['False Positive'] + predictiondf['False Negative'], ax = axe)
axe.set_xlabel('Model', size=10)
axe.set_ylabel('False Observations', size=10)

plt.show()

