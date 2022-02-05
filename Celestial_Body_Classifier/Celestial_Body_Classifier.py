
#LIBRARIES USED

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE


#UNDERSTANDING OUR DATA


df = pd.read_csv('/Users/maxwell/Downloads/star_classification.csv')
df.head()


df.info()


df['class'].value_counts()



df['class']=[0 if i == 'GALAXY' else 1 if i == 'STAR' else 2 for i in df['class']]



sns.countplot(df['class'], palette='plasma')
plt.title("Class ",fontsize=10)
plt.show()


#OUTLIER DETECTION


clf = LocalOutlierFactor()
y_pred = clf.fit_predict(df) 


x_score = clf.negative_outlier_factor_
#create a repository for the outlier datapoints
outlier_score = pd.DataFrame()
outlier_score["score"] = x_score

#threshold
threshold = -1.5                                            
filter = outlier_score["score"] < threshold
outlier_index = outlier_score[filter].index.tolist()
df.drop(outlier_index, inplace=True)


#FEATURE SELECTION


#show correlation of features with target class using heatmap
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, linewidths=0.1, fmt='.3f', ax=ax)
plt.show()


corr = df.corr()
corr['class'].sort_values()


#remove noise
df.drop(['obj_ID', 'alpha', 'delta', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'fiber_ID'], axis=1)


#HANDLING THE INBALANCE IN THE CLASSES


#we will now synthesize new data by duplicating existing elements within the classes to balance them out

X = df.drop(['class'], axis = 1)
y = df.loc[:,'class'].values


sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)


sns.countplot(y, palette='plasma')
plt.title("Class ",fontsize=10)
plt.show()


#DATA SCALING

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)



#MODEL CONSTRUCTION

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from yellowbrick.classifier import ConfusionMatrix


#Splitting data into training and testing batches

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=42)



#define model
svm_clf = svm.SVC(random_state=42)
svm_clf.fit(X_train, y_train)
#prediction
preds = svm_clf.predict(X_test)


#EVALUATION

score = svm_clf.score(X_test, y_test)
svm_score = np.mean(score)
print('Accuracy measures to', svm_score)


#create confusion matrix

classes = ['GALAXY','STAR','QUASAR']

svm_cm = ConfusionMatrix(svm_clf, classes=classes, cmap="BuPu")

svm_cm.fit(X_train, y_train)
svm_cm.score(X_test, y_test)
svm_cm.show()


#generate classification report

print(classification_report(y_test, preds))

