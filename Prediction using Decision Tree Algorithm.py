#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[17]:


ir=pd.read_csv('Iris2.csv')
ir


# In[18]:


ir = ir.drop('Id', axis = 1)
ir


# In[19]:


ir.describe()


# In[20]:


ir.shape


# In[6]:


ir.columns


# In[21]:


ir.Species.value_counts()


# In[22]:


sns.countplot(y= ir.Species, palette = "mako")


# In[24]:


fig = ir[ir.Species == 'Iris-setosa'].plot(kind = 'scatter', x= 'SepalLengthCm',y = 'SepalWidthCm',color = 'green', 
                                               label= 'Setosa')
ir[ir.Species == 'Iris-versicolor'].plot(kind = 'scatter', x= 'SepalLengthCm', y = 'SepalWidthCm',color = 'blue',
                                             label= 'Versicolor',ax = fig)
ir[ir.Species == 'Iris-virginica'].plot(kind = 'scatter', x= 'SepalLengthCm',y = 'SepalWidthCm',color = 'orange', 
                                            label= 'Virginica',ax = fig)
fig.set_xlabel("sepal length", fontsize =15)
fig.set_ylabel("sepal Width", fontsize = 15)
fig.set_title("sepal length Vs sepal width", fontsize= 20)
plt.show()


# In[25]:


fig = ir[ir.Species == 'Iris-setosa'].plot.scatter(x= 'PetalLengthCm',y = 'PetalWidthCm', color = 'green', label= 'Setosa')
ir[ir.Species == 'Iris-versicolor'].plot.scatter(x= 'PetalLengthCm', y = 'PetalWidthCm', color = 'blue',
                                                     label= 'Versicolor',ax = fig)
ir[ir.Species == 'Iris-virginica'].plot.scatter(x= 'PetalLengthCm',y = 'PetalWidthCm', color = 'orange', 
                                                    label= 'Virginica',ax = fig)
fig.set_xlabel("petal Length", fontsize = 15)
fig.set_ylabel("Petal Width", fontsize = 15)
fig.set_title("Petal Lenghth Vs Petal Width", fontsize = 20)
plt.show()


# In[26]:


plt.figure(figsize = (8,6))
sns.heatmap(ir.corr(),annot = True, cmap = 'PuBu')
plt.show()


# In[27]:


from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer


# In[28]:


re = LabelEncoder()


# In[29]:


ir.iloc[:,-1] =  re.fit_transform(ir.iloc[:,-1])
ir


# In[31]:


x =  ir.iloc[:,:-1]
x.head()


# In[33]:


y = ir.iloc[:,-1]
y.head()


# In[34]:


from sklearn.model_selection import train_test_split
trx, tex, tray, tey = train_test_split(x,y,test_size = 0.20, random_state = 50)
trx.head()


# In[35]:


print(trx.shape, tex.shape)


# In[36]:


print(trx.shape, tey.shape)


# In[37]:


from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier()


# In[38]:


dc.fit(trx,tray)


# In[40]:


ypred = dc.predict(tex)
ypred


# In[41]:


tey = np.array(tey)
tey


# In[42]:


fpred = pd.DataFrame( { 'Actual':  tey,'Predicted': dc.predict( tex) } )
fpred.sample(n = 10)


# In[43]:


from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
metrics.accuracy_score( fpred.Actual, fpred.Predicted )


# In[44]:


from sklearn.metrics import accuracy_score
accuracy_score(ypred, tey)


# In[47]:


from sklearn import tree
from matplotlib import pyplot as plt
plt.figure(figsize = (25,15))
t = tree.plot_tree(dc, feature_names = x.columns, filled = True,fontsize = 20)


# In[ ]:




