#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importation des modules nécessaires
import pandas as pd
import numpy as np


# In[2]:


import sys
get_ipython().system('{sys.executable} -m pip install seaborn')


# In[3]:


import seaborn as sb


# In[4]:


#charger les données dans un pandas dataframe
df = pd.read_csv('titanic.csv', sep='\t', engine='python')


# In[5]:


#visualiser les 10 premiéres lignes de notre dataframe 
df.head(10)


# In[6]:


#supprmier les colonnes Name, Ticket , Cabin qui ne sont pas numeriques
cols_to_drop = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols_to_drop, axis=1)


# In[7]:


df.head(3)


# In[8]:


#afficher les informations sur notre dataset
df.info()
sb.heatmap(df.isnull())


# In[9]:


#interpolation des valeurs manquantes
df['Age'] = df['Age'].interpolate()


# In[10]:


sb.heatmap(df.isnull())


# In[11]:


#supprmier tous les lignes avec des données manquantes
df = df.dropna()


# In[12]:


df.head()


# In[13]:


#convertir les données catégoriques en des données numériques
EmbarkedColumnDummy = pd.get_dummies(df['Embarked'])
SexColumnDummy = pd.get_dummies(df['Sex'])


# In[14]:


df = pd.concat((df, EmbarkedColumnDummy, SexColumnDummy), axis=1)


# In[15]:


df = df.drop(['Sex','Embarked'],axis=1)


# In[16]:


#séparation des données en X et Y
X = df.values
y = df['Survived'].values


# In[17]:


X = np.delete(X,1,axis=1)


# In[18]:


#séparation du données du test et donées du training 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[19]:


#Classification avec la regression logistique 
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression ()
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test)


# In[20]:


#classification avec Gradient Boost
from sklearn import ensemble
gb_clf = ensemble.GradientBoostingClassifier()
gb_clf.fit(X_train, y_train)
gb_clf.score(X_test, y_test)


# In[21]:


#Tuning
gb_clf = ensemble.GradientBoostingClassifier(n_estimators=50)
gb_clf.fit(X_train,y_train)
gb_clf.score(X_test, y_test)


# In[ ]:




