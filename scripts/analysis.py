#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, LearningCurveDisplay, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import RocCurveDisplay, roc_auc_score, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.base import clone

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
    
pd.options.mode.copy_on_write = True


# # Load data and model

# In[2]:


df = pd.read_csv('data/augmented_dataset.csv', index_col='id')

features = [i for i in df.columns if i != 'class']
target = 'class'

X, y = df.loc[:, features], df.loc[:, target]


# In[3]:


# Use the random state as in ``training.ipynb``.
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, train_size=0.9, random_state=1)


# ### Build a dummy classifier

# In[4]:


# Dummy classifier that predics the most frequent class.
dummy = DummyClassifier(strategy='most_frequent')


# In[5]:


dummy.fit(X_train, y_train)


# ### Load the model

# In[6]:


model = joblib.load('saved_models/best_model.joblib')
model


# # Visualize the results

# In[7]:


fig, ax = plt.subplots()

for clf in [dummy, model]:
    RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax)
ax.set_title('ROC curve of the best pipeline')

plt.savefig('figures/roc.pdf')


# In[8]:


fig, ax = plt.subplots()

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, normalize='true', cmap='binary', ax=ax)
ax.set_title('Normalized confusion matrix')

plt.savefig('figures/confusion_matrix.pdf')


# In[9]:


print('Model accuracy:', model.score(X_test, y_test))


# In[10]:


scores_test = model.predict_proba(X_test)[:, 1]


# In[11]:


scores_test_neg = scores_test[y_test == 0]
scores_test_pos = scores_test[y_test == 1]


# In[12]:


fig, ax = plt.subplots()

ax.hist(scores_test_neg, bins=50, label='Class 0', alpha=0.3);
ax.hist(scores_test_pos, bins=50, label='Class 1', alpha=0.3);
ax.set_ylabel('Count')
ax.set_xlabel('Predicted probability of sample being positive (class 1)')
ax.set_title(f'Test set size: {len(y_test)}')
ax.legend(loc='upper center')

plt.savefig('figures/class_scores.pdf')


# # Compare with experimental data

# In[13]:


df_receptor = pd.read_csv('data/Final_Receptor_dataset.csv', index_col='id')
all_ligand = pd.read_csv('data/df_smiles_final.csv')


# In[14]:


df_ligand = all_ligand.copy()
df_ligand.drop(['Value', 'Smiles 2'], axis=1, inplace=True)


# In[15]:


df_receptor.describe()


# ## Create a dataframe with receptor-ligand (using the same protein)

# In[16]:


# Create dummy protein rows.
df_protein = pd.concat([df_receptor.loc[['1lox']]]*len(df_ligand), ignore_index=True)


# In[17]:


df_pairs = pd.concat([df_protein, df_ligand], axis=1)
df_pairs


# ## Make predictions

# In[18]:


preds = model.predict_proba(df_pairs)[:, 1]  # Get score for "valid".


# In[19]:


df_results = all_ligand.copy()
df_results.insert(0, 'Score', preds)


# In[20]:


df_results.sort_values(by='Score', ascending=False)


# In[21]:


# Write predictions to csv for later analysis.
df_results.sort_values(by='Score', ascending=False).to_csv('results/1lox_ligand_scores.csv', index=False)


# In[22]:


k = 10
top_k =  (-preds).argsort()[:k]


# In[23]:


top_k


# # Visualize data

# In[24]:


pca_embeddings = X.copy()
pca_embeddings_pairs = df_pairs.copy()

for (name, step) in model.steps[:2]:
    pca_embeddings = step.transform(pca_embeddings)
    pca_embeddings_pairs = step.transform(pca_embeddings_pairs)


# In[25]:


fig, axes = plt.subplots(3, 3, layout='constrained')

for i, ax in enumerate(axes.flatten()):
    ax.scatter(pca_embeddings[:, 0], pca_embeddings[:, i], c='tab:green', s=1, label='Original data')
    ax.scatter(pca_embeddings_pairs[:, 0], pca_embeddings_pairs[:, i], c='tab:red', s=1, label='Data for "1lox"')
    ax.set_xlabel('PCA-0')
    ax.set_ylabel(f'PCA-{i}')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='outside upper center')

fig.savefig('figures/1lox_pca_scatter.png', dpi=300)


# In[26]:


fig, axes = plt.subplots(3, 3, layout='constrained')

for i, ax in enumerate(axes.flatten()):
    ax.hist(pca_embeddings[:, i], color='tab:green', alpha=0.7, bins=20, density=True, label='Original data')
    ax.hist(pca_embeddings_pairs[:, i], color='tab:red', alpha=0.7, bins=20, density=True, label='Data for "1lox"')
    ax.set_xlabel(f'PCA-{i}')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='outside upper center')

fig.savefig('figures/1lox_pca_hist.pdf')


# In[27]:


old_ligand = pd.read_csv('data/Final_Ligand_dataset.csv', index_col='id')


# In[29]:


fig, axes = plt.subplots(3, 3, layout='constrained')

for i, ax in enumerate(axes.flatten()):
    feat = old_ligand.columns[i+2]
    ax.hist(old_ligand[feat], color='tab:green', bins=15, alpha=0.7, density=True, label='Original data')
    ax.hist(df_ligand[feat], color='tab:red', bins=15, alpha=0.7, density=True, label='Data for "1lox"')
    ax.set_xlabel(feat)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='outside upper center')

fig.savefig('figures/1lox_feats_hist.pdf')


# In[30]:


fig, ax = plt.subplots()

pca_ratio = model.named_steps['reducer'].explained_variance_ratio_[:25]
ax.barh(range(len(pca_ratio)), pca_ratio)
ax.set_xlabel('Explained variance ratio')
ax.set_ylabel('PCA component')


# In[31]:


df_100 = df_results.sort_values(by='Score', ascending=False)[:100]

fig, ax = plt.subplots()

for i, k in enumerate(ticks := [10, 20, 50, 100]):
    ax.scatter(np.full(k, i), df_100[:k].Value, alpha=0.3)
    ax.set_yscale('log')
    ax.set_xlabel('Top-k ligands (as predicted by ML)')
    ax.set_ylabel('Affinity value (log scale)')
    ax.set_title('Affinity values for the top-k ligands of "1lox"')
ax.set_xticks(range(len(ticks)), ticks);

fig.savefig('figures/1lox_top_k.pdf')


# In[32]:


fig, ax = plt.subplots()

im = ax.scatter(df_results.Value, df_results.Score, c=df_results.Score)
ax.set_xscale('log')
ax.set_xlabel('Affinity')
ax.set_ylabel('Score (ML prediction)')
ax.set_title('Predicted score vs affinity for "1lox"')
fig.colorbar(im, ax=ax, label='Score')

fig.savefig('figures/1lox_score_vs_affinity.pdf')


# In[50]:


print(df_results[['Score', 'Value']].sort_values(by='Score', ascending=False)[:10].to_latex(float_format='%.3f'))


# In[ ]:




