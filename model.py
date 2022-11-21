import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, plot_confusion_matrix, plot_roc_curve
import pickle

# Upload dataset
df = pd.read_csv('dataR2.csv')

#List the columns with outlier
outlier_col = ['Glucose','Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']

# Transform features using Log transformation 
np.seterr(divide = 'ignore')
for i in outlier_col:
    df['Log_'+i] = np.log(df[i])
    
#  Log transformated dataset
df2 = df.loc[:, ['Age', 'BMI', 'Log_Glucose', 'Log_Insulin', 'Log_HOMA', 'Log_Leptin', 'Log_Adiponectin', 'Log_Resistin','Log_MCP.1', 'Classification']]

#Detection and Removal of Outliers using IQR
Q1 = df2.quantile(0.25)
Q3 = df2.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_out = df2[~((df2 < lower_bound) | (df2 > upper_bound)).any(axis=1)]    # Dataset without outlier

# Extract features and target variables
X = df_out.drop(['Classification'], axis=1)
y = df_out['Classification']

# Save the feature name and target variables
feature_names = X.columns
labels = y.unique()

# Split data into 75% training set and 25% testing set
X_train, X_test, y_train, y_test = train_test_split(X, y , random_state= 55, test_size=0.3, shuffle=True)

# Standardizing data with StandardScaler() function
sc = StandardScaler()
X_train =  sc.fit_transform(X_train)
X_test =  sc.fit_transform(X_test)

rfc = RandomForestClassifier(random_state= 1234)
rfc.fit(X_train, y_train)

y_prediction = rfc.predict(X_test) 
test_accuracy= accuracy_score(y_test, y_prediction)*100

importances = rfc.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
feat_labels = df.columns[1:]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[sorted_indices[f]],
                            importances[sorted_indices[f]]))


print(test_accuracy)

# pickling the model
pickle_out = open("rfc.pkl", "wb")
pickle.dump(rfc, pickle_out)
pickle_out.close()