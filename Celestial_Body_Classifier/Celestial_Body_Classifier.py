#!/usr/bin/env python
# coding: utf-8

# In[1]:


#LIBRARIES USED


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE


# In[3]:


#UNDERSTANDING OUR DATA


# In[4]:


df = pd.read_csv('/Users/maxwell/Downloads/star_classification.csv')
df.head()


# In[5]:


df.info()


# In[6]:


df['class'].value_counts()


# In[7]:


df['class']=[0 if i == 'GALAXY' else 1 if i == 'STAR' else 2 for i in df['class']]


# In[8]:


sns.countplot(df['class'], palette='plasma')
plt.title("Class ",fontsize=10)
plt.show()


# In[9]:


#OUTLIER DETECTION


# In[10]:


clf = LocalOutlierFactor()
y_pred = clf.fit_predict(df) 


# In[11]:


x_score = clf.negative_outlier_factor_
#create a repository for the outlier datapoints
outlier_score = pd.DataFrame()
outlier_score["score"] = x_score

#threshold
threshold = -1.5                                            
filter = outlier_score["score"] < threshold
outlier_index = outlier_score[filter].index.tolist()
df.drop(outlier_index, inplace=True)


# In[12]:


#FEATURE SELECTION


# In[13]:


#show correlation of features with target class using heatmap
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, linewidths=0.1, fmt='.3f', ax=ax)
plt.show()


# In[14]:


corr = df.corr()
corr['class'].sort_values()


# In[15]:


#remove noise
df.drop(['obj_ID', 'alpha', 'delta', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'fiber_ID'], axis=1)


# In[16]:


#HANDLING THE INBALANCE IN THE CLASSES


# In[17]:


#we will now synthesize new data by duplicating existing elements within the classes to balance them out


# In[18]:


X = df.drop(['class'], axis = 1)
y = df.loc[:,'class'].values


# In[19]:


sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)


# In[20]:


sns.countplot(y, palette='plasma')
plt.title("Class ",fontsize=10)
plt.show()


# In[21]:


#DATA SCALING


# In[22]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# In[23]:


#MODEL CONSTRUCTION


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from yellowbrick.classifier import ConfusionMatrix


# In[25]:


#Splitting data into training and testing batches


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=42)


# In[27]:


#define model
svm_clf = svm.SVC(random_state=42)
svm_clf.fit(X_train, y_train)
#prediction
preds = svm_clf.predict(X_test)


# In[28]:


#EVALUATION


# In[29]:


score = svm_clf.score(X_test, y_test)
svm_score = np.mean(score)
print('Accuracy measures to', svm_score)


# In[30]:


#create confusion matrix


# In[34]:


classes = ['GALAXY','STAR','QUASAR']

svm_cm = ConfusionMatrix(svm_clf, classes=classes, cmap="BuPu")

svm_cm.fit(X_train, y_train)
svm_cm.score(X_test, y_test)
svm_cm.show()


# In[32]:


#generate classification report


# In[33]:


print(classification_report(y_test, preds))

