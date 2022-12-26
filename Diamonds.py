#!/usr/bin/env python
# coding: utf-8

# Analysis diamonds by their cut,color,clarity ,price and other attributes

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn .preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso


# In[2]:


data=pd.read_csv('diamonds.csv')
data


# In[3]:


data.drop(data.columns[0],axis=1,inplace=True)


# In[4]:


y=data['price']
X=data.drop('price',axis=1)


# In[14]:


data.nunique(axis = 0, dropna = True)
#or 
#print(f"Cuts:{len(data['cut'].unique())}")
#print(f"Clarity:{len(data['clarity'].unique())})
#print(f"Color : {len(data['color'].unique())})


# In[20]:


encoder =LabelEncoder()
X['cut']=encoder.fit_transform(X['cut'])
cut_mapping={index: label for index , label in enumerate(encoder.classes_)}
cut_mapping

X['color']=encoder.fit_transform(X['color'])
color_mapping={index: label for index , label in enumerate(encoder.classes_)}
color_mapping

X['clarity']=encoder.fit_transform(X['clarity'])
clarity_mapping={index: label for index , label in enumerate(encoder.classes_)}
clarity_mapping


# In[21]:


print(cut_mapping)
print(color_mapping)
print(clarity_mapping)


# In[22]:


X


# In[23]:


scaler =MinMaxScaler()
X=scaler.fit_transform(X)


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=0.8)


# In[29]:


std_model=LinearRegression()
l1_model=Lasso(alpha=1)
l2_model=Ridge(alpha=1)

std_model.fit(X_train,y_train)
l1_model.fit(X_train,y_train)
l2_model.fit(X_train,y_train)


# In[30]:


print(f"---Without regularization: {std_model.score(X_test,y_test)}")
print(f"Lass0(l1) regularization: {l1_model.score(X_test,y_test)}")
print(f"Ridge(l2) regularization: {l2_model.score(X_test,y_test)}")


# In[35]:


l2_model=Ridge(alpha=0.001)
l2_model.fit(X_train,y_train)
print(f"Ridge(l2) regularization: {l2_model.score(X_test,y_test)}")


# In[ ]:




