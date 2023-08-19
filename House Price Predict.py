#!/usr/bin/env python
# coding: utf-8

# # Housing Price Prediction

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


# In[2]:


df = pd.read_csv('C:/Users/admin/Desktop/House price.csv')
print(df.shape)
df.sample(5)


# In[3]:


print(df.columns) 


# In[4]:


pd.options.display.max_columns=None
df.info()


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.info


# In[40]:


df = df.drop_duplicates()


# In[8]:


df.plot('area', 'price', kind='scatter')


# In[9]:


Category_features = ['mainroad','guestroom', 'basement', 'hotwaterheating', 
            'airconditioning', 'prefarea', 'furnishingstatus']
Numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']


# In[10]:


for feature in df.columns:
    print(feature, " : ", df[feature].unique())


# In[11]:


fig, axes = plt.subplots(2, 4, figsize=(20,10))
plt.title("Category_feature")
for k in range(len(Category_features)):
  num = []
  for t in df[Category_features[k]].unique():
    num.append(df[Category_features[k]].tolist().count(t))
  axes[k//4][k%4].pie(num, labels=df[Category_features[k]].unique(), autopct="%.2f%%", labeldistance=1.15, 
            wedgeprops = {'linewidth':1, 'edgecolor':'white'}, textprops={'color':'lightgreen', 'fontsize':15}, 
            colors=sns.color_palette('Blues_r'))
  axes[k//4][k%4].set_title(Category_features[k])
axes[-1][-1].axis('off')
plt.tight_layout()
plt.show()


# In[12]:


import random

random_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(5)]


# In[13]:


fig = plt.figure(figsize=(20, 10))
gs = GridSpec(nrows=2, ncols=3)
for k in range(len(Numeric_features)-1):
  ax=fig.add_subplot(gs[k//3, k%3])
  sns.countplot(ax=ax, data=df, x=Numeric_features[k+1], palette=sns.color_palette('pastel'))
k+=1
sns.countplot(ax=fig.add_subplot(gs[k//3, k%3:]), data=df, x=Numeric_features[0], palette=sns.color_palette('pastel'))
plt.tight_layout()
plt.show()


# In[14]:


fig = plt.figure(figsize=(20, 10))
gs = GridSpec(nrows=2, ncols=3)
for k in range(len(Numeric_features)-1):
  ax=fig.add_subplot(gs[k//3, k%3])
  sns.countplot(ax=ax, data=df, x=Numeric_features[k+1], palette=sns.color_palette('pastel'))
  for label in ax.containers:
    ax.bar_label(label)
k+=1
ax = sns.countplot(ax=fig.add_subplot(gs[k//3, k%3:]), data=df, x=Numeric_features[0], palette=sns.color_palette('pastel'))
for label in ax.containers:
    ax.bar_label(label)
plt.tight_layout()


# In[17]:


sns.boxplot(data=df, y="area")
plt.show()


# In[24]:


sns.set_style("darkgrid")


# In[23]:


plt.figure(figsize=(5,8))
plt.boxplot(x=df['price'], notch=True)
plt.ylabel('price')


# In[25]:


fig, axes = plt.subplots(1, 3, figsize=(5,7))
sns.boxplot(ax=axes[0], data=df, y="price")
sns.boxplot(ax=axes[1], data=df, y="price", showcaps=False, 
        whiskerprops={"linestyle": 'dashed', "lw":4})
sns.boxplot(ax=axes[2], data=df, y="price", notch=True, showmeans=True, meanline=True, 
        meanprops={"color": "r", "lw":2}, medianprops={"color": "c", "lw":3})
plt.tight_layout()
plt.show()


# In[26]:


fig, axes = plt.subplots(2, 2, figsize=(10,10))
fig.suptitle('Histogram')
sns.histplot(ax=axes[0][0], data=df, x='price')
axes[0][0].set_title('Default (bins="auto")')
sns.histplot(ax=axes[0][1], data=df, x='price', binwidth=1e6)
axes[0][1].set_title('Set binwidth to 1e6')
sns.histplot(ax=axes[1][0], data=df, x='price', bins=5, element="bars", shrink=0.8)
axes[1][0].set_title('Set number of bins to 5')
sns.histplot(ax=axes[1][1], data=df, x='price', bins=5, element="step", stat='percent')
axes[1][1].set_title('percentage')
plt.tight_layout()
plt.show()


# In[27]:


fig, axes = plt.subplots(3,1, figsize=(5,15))
fig.suptitle('Histogram  (continue data)')
sns.kdeplot(ax=axes[0], data=df, x='price')
axes[0].set_title('Default')
sns.kdeplot(ax=axes[1], data=df, x='price', bw_adjust=.2)
axes[1].set_title('less smooth')
sns.kdeplot(ax=axes[2], data=df, x='price', bw_adjust=1.5, cut=0)
axes[2].set_title('more smoothing')
plt.tight_layout()
plt.show()


# In[28]:


sns.displot(data=df, x='price', kde=True)


# In[29]:


fig, axes = plt.subplots(1, 2, figsize=(10,5))
sns.scatterplot(ax=axes[0], data=df, x='area', y='price', hue='mainroad', size="stories", sizes=(10, 200), alpha=0.2)
axes[0].legend(loc='center left', prop={'size': 8})
markers = {'no':'X', 'yes':'o'}
sns.scatterplot(ax=axes[1], data=df, x='area', y='price', style='mainroad', markers=markers, alpha=0.2)
plt.tight_layout()
plt.show()


# In[30]:


fig, axes = plt.subplots(1, 3, figsize=(15,5))
sns.histplot(ax=axes[0], data=df, x="price", hue="mainroad")
sns.histplot(ax=axes[1], data=df, x="price", hue="guestroom")
sns.histplot(ax=axes[2], data=df, x="price", hue="basement")
plt.tight_layout()
for i in range(3):
    axes[i].set_ylim(0, 80)
    axes[i].set_xlim(0.1e7, 1.3e7)
plt.show()


# In[31]:


sns.displot(data=df, x="price", hue="mainroad", row="stories", col="furnishingstatus", kde=True)


# In[32]:


sns.jointplot(data=df, x="area", y="price", hue="prefarea", marker="*", s=200, alpha=0.3, marginal_ticks=True)


# In[33]:


sns.jointplot(data=df, x="area", y="price", kind="kde", hue="basement", marginal_ticks=True)


# In[34]:


print(df.cov())


# In[41]:


Q1, Q3 = df['area'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR


# In[42]:


print(df.shape)
df = df[ (df['area'] >= lower_limit) & (df['area'] <= upper_limit) ]
print(df.shape)


# In[43]:


r = 3
lower_limit = df['area'].mean() - r*df['area'].std()
upper_limit = df['area'].mean() + r*df['area'].std()  


# In[38]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), cmap="Blues", annot=True)
plt.show()


# In[44]:


print(df.shape)
df = df[ (df['area'] >= lower_limit) & (df['area'] <= upper_limit) ]
print(df.shape)


# In[45]:


import numpy as np
import pandas as pd

ex_df = pd.DataFrame({
    'feature_name' : ['square', np.nan, 'oval', 'square', 'circle', np.nan, 'triangle'],
    'feature_name2' : [1, np.nan, 3, 4, 5, np.nan, 7],
    'feature_name3' : ['squares', 'triangles', np.nan, 'circles', 'ovals', np.nan, 'squares'],
})
ex_df


# In[46]:


ex_df.isnull().sum()


# In[48]:


ex_df = ex_df.dropna(axis=0)
ex_df


# In[49]:


ex_df['feature_name2'] = ex_df['feature_name2'].fillna(ex_df['feature_name2'].median())
ex_df


# In[50]:


from sklearn.impute import SimpleImputer
imr = SimpleImputer(strategy='most_frequent')
ex_df = pd.DataFrame( imr.fit_transform(ex_df), columns = ex_df.columns)
imr = imr.fit(ex_df)
ex_df = imr.transform(ex_df)

ex_df = pd.DataFrame(ex_df)
ex_df


# In[51]:


df[Category_features].head(5)


# In[52]:


df['furnishingstatus'].unique()


# In[53]:


df['furnishingstatus'].head(8)


# In[57]:


type_dict = {'unfurnished':1, 'semi-furnished': 2, 'furnished': 3}
df['furnishingstatus'] = df['furnishingstatus'].map(type_dict)

df['furnishingstatus'].head(8)


# In[58]:


df.head(5)


# In[66]:


df[Category_features].head(5)


# In[67]:


for k in Category_features:
  num = df[k].value_counts()  
  for i in range(len(num.index)):
    df[k] = df[k].replace(num.index[i], num[num.index[i]])

df[Category_features].head(5)


# In[89]:


corr = df.corr()
corr

