# Breast-Cancer-Detection
# import libraries

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import accuracy_score

#Load breast cancer dataset
from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()

type(cancer_dataset)

#Data Manipulation
cancer_dataset

#keys in dataset
cancer_dataset.keys()

# Features of each cells in numeric format
cancer_dataset['data']

# Description of data
print(cancer_dataset['DESCR'])

# Name of features
print(cancer_dataset['feature_names'])

# Location/path of data file
print(cancer_dataset['filename'])

# Create datafrmae
cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
             columns = np.append(cancer_dataset['feature_names'], ['target']))

# Head of cancer DataFrame
cancer_df.head(6)

# Tail of cancer DataFrame
cancer_df.tail(6)

# Information of cancer Dataframe
cancer_df.info()

# Numerical distribution of data
cancer_df.describe()

#DATA VISUALIZATION
#PAIR PLOT OF BREAST CANCER DATA
# Pairplot of cancer dataframe
sns.pairplot(cancer_df, hue = 'target')

# pair plot of sample feature
sns.pairplot(cancer_df, hue = 'target', 
             vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'] )

import seaborn as sns
print(cancer_df["target"])

#COUNTERPLOT

# Count the target class
ax=sns.countplot(x="target",data=cancer_df,hue = 'target')

custom_palette = sns.color_palette("husl", len(cancer_df['mean radius'].unique()))

plt.figure(figsize=(20, 8))
a = sns.countplot(x='mean radius', data=cancer_df, palette=custom_palette)


a.set_xlabel('Mean Radius', fontsize=14)  # Set x-axis label with fontsize
a.set_ylabel('Count', fontsize=14)  # Set y-axis label with fontsize
a.set_title('Count of Mean Radius', fontsize=16)  # Set plot title with fontsize
a.tick_params(axis='x', labelrotation=45, labelsize=12)  # Rotate and adjust fontsize of x-axis labels
a.tick_params(axis='y', labelsize=12)  # Adjust fontsize of y-axis labels

plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()

#HEATMAP

# heatmap of DataFrame
plt.figure(figsize=(16,9))
sns.heatmap(cancer_df)

# Heatmap of Correlation matrix of breast cancer DataFrame
plt.figure(figsize=(20,20))
sns.heatmap(cancer_df.corr(), annot = True, cmap ='coolwarm', linewidths=2)

#Correlation barplot

# create second DataFrame by droping target
cancer_df2 = cancer_df.drop(['target'], axis = 1)
print("The shape of 'cancer_df2' is : ", cancer_df2.shape)

colors = sns.color_palette('husl', len(cancer_df2))

plt.figure(figsize=(16, 5))
ax = sns.barplot(x=cancer_df2.corrwith(cancer_df['target']).index, y=cancer_df2.corrwith(cancer_df['target']), palette=colors)


ax.set_xlabel('Features', fontsize=14)  # Set x-axis label with fontsize
ax.set_ylabel('Correlation', fontsize=14)  # Set y-axis label with fontsize
ax.set_title('Correlation with Target', fontsize=16)  # Set plot title with fontsize
ax.tick_params(axis='x', labelrotation=45, labelsize=12)  # Rotate and adjust fontsize of x-axis labels
ax.tick_params(axis='y', labelsize=12)  # Adjust fontsize of y-axis labels

plt.tight_layout()
plt.show()

#DATA PREPROCESSING
# input variable
X = cancer_df.drop(['target'], axis = 1)
X.head(6)

# output variable
y = cancer_df['target']
y.head(6)

#split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 5)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# Train with Standard scaled Data
svc_classifier2 = SVC()
svc_classifier2.fit(X_train_sc, y_train)
y_pred_svc_sc = svc_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_svc_sc)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming X_train, X_test, y_train, y_test are already defined

lr_classifier = LogisticRegression(random_state=51, solver='lbfgs')  # Using default penalty ('l2')
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_lr)
print("Accuracy:", accuracy)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming X_train_sc, X_test_sc, y_train, y_test are already defined and X_train_sc and X_test_sc are scaled versions of X_train and X_test respectively

lr_classifier2 = LogisticRegression(random_state=51, solver='lbfgs')  # Using default penalty ('l2')
lr_classifier2.fit(X_train_sc, y_train)
y_pred_lr_sc = lr_classifier2.predict(X_test_sc)  # Use lr_classifier2 instead of lr_classifier
accuracy = accuracy_score(y_test, y_pred_lr_sc)
print("Accuracy:", accuracy)

#K-NEAREST NEIGHBOR CLASSIFIER


from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_score(y_test, y_pred_knn)

# Train with Standard scaled Data
knn_classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier2.fit(X_train_sc, y_train)
y_pred_knn_sc = knn_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_knn_sc)

#NAIVFE BAYES CLASSIFIER
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_nb)

# Train with Standard scaled Data
nb_classifier2 = GaussianNB()
nb_classifier2.fit(X_train_sc, y_train)
y_pred_nb_sc = nb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_nb_sc)

#DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuracy_score(y_test, y_pred_dt)

#DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuracy_score(y_test, y_pred_dt)

#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
accuracy_score(y_test, y_pred_rf)

# Train with Standard scaled Data
rf_classifier2 = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier2.fit(X_train_sc, y_train)
y_pred_rf_sc = rf_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_rf_sc)

#ADABOOST CLASSIFIER
from sklearn.ensemble import AdaBoostClassifier
adb_classifier = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    algorithm='SAMME.R',
                                    random_state=1,)
adb_classifier.fit(X_train, y_train)
y_pred_adb = adb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_adb)

# Train with Standard scaled Data
adb_classifier2 = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    algorithm='SAMME.R',
                                    random_state=1,)
adb_classifier2.fit(X_train_sc, y_train)
y_pred_adb_sc = adb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_adb_sc)

#XGBOOST CLASSIFIER
import xgboost
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_xgb)

# Train with Standard scaled Data
xgb_classifier2 = XGBClassifier()
xgb_classifier2.fit(X_train_sc, y_train)
y_pred_xgb_sc = xgb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_xgb_sc)

#XGBoost Parameter Tuning Randomized Search
params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] 
}

# Randomized Search
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(xgb_classifier, param_distributions=params, scoring= 'roc_auc', n_jobs= -1, verbose= 3)
random_search.fit(X_train, y_train)

{'min_child_weight': 1,
 'max_depth': 3,
 'learning_rate': 0.3,
 'gamma': 0.4,
 'colsample_bytree': 0.3}

random_search.best_estimator_

# training XGBoost classifier with best parameters
xgb_classifier_pt = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.4, gamma=0.2,
       learning_rate=0.1, max_delta_step=0, max_depth=15,
       min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
       nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)

import xgboost as xgb
from sklearn.metrics import accuracy_score

# Assuming X_train, X_test, y_train, and y_test are already defined

xgb_classifier_pt = xgb.XGBClassifier()  # Assuming you've initialized your XGBoost classifier
xgb_classifier_pt.fit(X_train, y_train)
y_pred_xgb_pt = xgb_classifier_pt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_xgb_pt)
print("Accuracy:", accuracy)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred_xgb_pt)
plt.title('Heatmap of Confusion Matrix', fontsize=15)
sns.heatmap(cm, annot=True)
plt.show()

#CLASSIFICATION REPORT OF MODEL
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_xgb_pt))

#Cross-Validation of the ML Model
import xgboost as xgb


xgb_model_pt2 = xgb.XGBClassifier()

# Now you can perform cross-validation
cross_validation = cross_val_score(estimator=xgb_model_pt2, X=X_train_sc, y=y_train, cv=10)
print("Cross validation of XGBoost model =", cross_validation)
print("Cross validation of XGBoost model (in mean) =", cross_validation.mean())

#SAVING THE MACHINE LEARNING MODEL
## Pickle
import pickle
 
# save model
pickle.dump(xgb_classifier_pt, open('breast_cancer_detector.pickle', 'wb'))
 
# load model
breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))
 
# predict the output
y_pred = breast_cancer_detector_model.predict(X_test)
 
# confusion matrix
print('Confusion matrix of XGBoost model: \n',confusion_matrix(y_test, y_pred),'\n')
 
# show the accuracy
print('Accuracy of XGBoost model = ',accuracy_score(y_test, y_pred))




