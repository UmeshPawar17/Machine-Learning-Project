#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Prediction

# Name- Umesh Sunil Pawar

# In[1]:


# Import Required Libraries.
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[2]:


# Read input Data
df= pd.read_csv("wine.csv")
df


# In[3]:


df.shape


# In[4]:


# Take info from the data
df.info()


# In[5]:


# Find the null Value.
df.isnull()


# In[6]:


df=df.fillna(df.mean())
df.isnull().sum()


# In[7]:


x=df[['fixed acidity', 'volatile acidity','citric acid', 'residual sugar',
     'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
     'pH', 'sulphates', 'alcohol']]
y=df['quality']
print(x)
print(y)


# In[8]:


# Here Quality is highly correlated with alcohol
sns.heatmap(df.corr())


# In[ ]:


# Plot out the Data

fig = plt.figure(figsize=(15,10))
plt.subplot(3,4,1)
sns.barplot(x='quality', y='fixed acidity', data=df)
plt.subplot(3,4,2)
sns.barplot(x='quality', y='volatile acidity', data=df)
plt.subplot(3,4,3)
sns.barplot(x='quality', y='citric acid', data=df)
plt.subplot(3,4,4)
sns.barplot(x='quality', y='residual sugar', data=df)
plt.subplot(3,4,5)
sns.barplot(x='quality', y='chlorides', data=df)
plt.subplot(3,4,6)
sns.barplot(x='quality', y='free sulfur dioxide', data=df)
plt.subplot(3,4,7)
sns.barplot(x='quality', y='total sulfur dioxide', data=df)
plt.subplot(3,4,8)
sns.barplot(x='quality', y='density', data=df)
plt.subplot(3,4,9)
sns.barplot(x='quality', y='pH', data=df)
plt.subplot(3,4,10)
sns.barplot(x='quality', y='sulphates', data=df)
plt.subplot(3,4,11)
sns.barplot(x='quality', y='alcohol', data=df)



# In[ ]:


# Count the instances of Each rows.
df['quality'].value_counts()


# In[ ]:


# Make Just 2 categories as good had
ranges = (2,5,8)
groups = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins=ranges, labels=groups)


# In[ ]:


# Alloting 0 to Bad & 1 to Good.
le = LabelEncoder()
df['quality'] = le.fit_transform(df['quality'])
df


# In[ ]:


# Again check counts.
df['quality'].value_counts()


# In[ ]:


# Balancing our Dataset, & make new dataset.
good_quality = df[df['quality']==1]
bad_quality = df[df['quality']==0]
bad_quality = bad_quality.sample(frac=1)
bad_quality = bad_quality[:217]
new_df = pd.concat([good_quality, bad_quality])
new_df = new_df.sample(frac=1)
new_df


# In[ ]:


new_df['quality'].value_counts()


# In[ ]:


#Checking the correlationn between columns
new_df.corr()['quality'].sort_values(ascending=False)


# In[ ]:


# Spliting the data into train & test.
from sklearn.model_selection import train_test_split
x = new_df.drop('quality', axis=1)
y = new_df['quality']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)


# In[ ]:


le = LabelEncoder()
df['quality'] = le.fit_transform(df['quality'])
df['type'] = encoder.fit_transform(df['type'])
df.head()


# In[ ]:


# finally training our wine quality prediction model.
param = {'max_depth': [100,200,300,400,500,600,700,800,900,1000]}
#grid_rf = GridSearchCV(RandomForestClassifier(), param, scoring='accuracy', cv=10)
grid_rf = GridSearchCV(RandomForestClassifier(), param, scoring='accuracy', cv=10, error_score='raise')

grid_rf.fit(x_train, y_train)
print('Best parameters -->', grid_rf.best_params_)

# Wine Quality Prediction
pred = grid_rf.predict(x_test)
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))
print('\n')
print(accuracy_score(y_test, pred))


# In[ ]:


df.info()


# In[ ]:





# In[ ]:




