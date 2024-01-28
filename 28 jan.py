#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('', 'matplotilo as mp')


# In[9]:


data=pd.read_csv('Advertising.csv')


# In[10]:


data.tail


# In[14]:


data.info()


# In[13]:


data.describe()


# In[15]:


data.head()


# In[16]:


#analyzing Tv
sns.displot(x=data.TV,kde=True)


# In[17]:


#analyzing radio
sns.histplot(x=data.Radio,kde=True)


# In[19]:


sns.histplot(x=data.Newspaper,kde=True)


# In[20]:


data.head()


# In[22]:


#scatter to know the realtion
#analyze tv and sales
sns.relplot(x="TV",y="Sales",data=data)


# In[24]:


sns.relplot(x='Radio',y='Sales',data=data)


# In[25]:


sns.relplot(x='Newspaper',y='Sales',data=data)


# In[26]:


sns.pairplot(data)


# In[27]:


#1.finding ,missing values
data.isnull().sum()


# In[28]:


# 2.categorical


# In[29]:


#check outlier 
sns.boxplot(x='TV',data=data)


# In[30]:


sns.boxplot(x='Radio',data=data)


# In[31]:


sns.boxplot(x='Newspaper',data=data)


# In[32]:


data.head()


# In[34]:


l1=['Unnamed: 0']
data.drop(l1,axis=1,inplace=True)


# In[35]:


data.head()


# In[36]:


data.head()


# In[37]:


#check correlation
sns.heatmap(data.drop('Sales',axis=1).corr(),annot=True)


# In[38]:


data.corr()


# In[39]:


#data modeling
data.head()


# In[40]:


#1 independent and dependent
X=data[['TV','Radio','Newspaper']]
Y=data.Sales


# In[41]:


X


# In[42]:


Y


# In[43]:


#train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=4)


# In[44]:


X_train


# In[48]:


# creating model
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_train,Y_train)
Y_predict=LR.predict(X_test)


# In[50]:


Y_test


# In[51]:


Y_predict


# In[52]:


from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
r2score=r2_score(Y_test,Y_predict)


# In[53]:


r2score


# In[54]:


X_test.shape


# In[55]:


#calculationn of adjsyted r2 score
adjusted_r2 = 1-(1-0.90)*(40-3)/(40-3-1)


# In[56]:


adjusted_r2


# In[57]:


import math
print(mean_squared_error(Y_test,Y_predict))
print(math.sqrt(mean_squared_error(Y_test,Y_predict)))


# In[60]:


print(mean_absolute_error(Y_test,Y_predict))#value of abso


# In[ ]:




