#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from joblib import dump


# # Dataset

# In[3]:


df = pd.read_csv('data/augmented_dataset.csv', index_col='id')

features = [i for i in df.columns if i != 'class']
target = 'class'

X, y = df.loc[:, features], df.loc[:, target]


# In[4]:


df.head(5)


# # Training

# ## Configurations
# 
# All the configurations have the form:
# 
# $$ \text{scaler} \mapsto \text{reducer} \mapsto \text{classifier}$$

# In[9]:


steps = [
    ('scaler', StandardScaler()),
    ('reducer', PCA(random_state=1)),
    ('classifier', 'passthrough'),  # Will be populated by param_grid.
]

param_grid = [
    {
        'reducer__n_components': [20, 50, 100, 300],
        'classifier': [RandomForestClassifier(random_state=1)],
        'classifier__n_estimators': [100, 200, 500],
        'classifier__min_samples_split': [2, 5, 10],
    },
    {
        'reducer__n_components': [20, 50, 100, 300],
        'classifier': [XGBClassifier(random_state=1)],
        'classifier__n_estimators': [50, 100, 500, 1000, 2000],
        'classifier__learning_rate': [0.1, 0.5, 1.],
        'classifier__max_depth': [2, 4, 6, 8],
    },
]


# ## CV configuration

# In[10]:


# Create the pipeline: scaler => reducer => classifier.
pipeline = Pipeline(steps=steps)

# Spli the data into train and test with stratification.
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, train_size=0.9, random_state=1)

# 5-fold cross validation for tuning hyperparameters.
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Use AUC as validation metric.
grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=kf, scoring='roc_auc', n_jobs=-1, refit=True)


# ## Tuning hyperparameters and refitting on the whole training set

# In[ ]:


grid.fit(X_train, y_train)


# # Save the best model and the grid

# In[ ]:


best_model = grid.best_estimator_
best_model


# In[ ]:


dump(best_model, 'saved_models/best_model.joblib')
dump(grid, 'saved_models/gridsearch_cv.joblib')

