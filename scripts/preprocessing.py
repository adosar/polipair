#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from umap import UMAP


# In[58]:


# For reproducible results
np.random.seed(1)


# # Import the datasets for receptors and ligands

# In[59]:


# Load the receptors dataset.
df_receptors = pd.read_csv('Final_Receptor_dataset.csv', index_col='id')

# Load the ligands dataset.
df_ligands = pd.read_csv('Final_Ligand_dataset.csv', index_col='id')


# In[60]:


df_receptors.head()


# In[61]:


df_ligands.head()


# In[62]:


# Check that both datasets have the same indinces.
np.all(df_receptors.index == df_ligands.index)


# In[63]:


# Get the indices.
indices = df_receptors.index


# In[64]:


# Check the size of the dataset.
print(f'The size of datasets is {len(df_receptors)} and {len(df_ligands)}')


# ## Split datasets into train and test sets

# In[65]:


# Create the train, test indices.
train_size = 0.75
val_size = 0.05
test_size = 0.15

train_indices, val_test_indices = train_test_split(indices.values, train_size=train_size)
val_indices, test_indices = train_test_split(val_test_indices, test_size=test_size / (test_size + val_size))


# In[66]:


# Print the size of train, val and test set.
for mode, idx in zip(('train', 'val', 'test'), (train_indices, val_indices, test_indices)):
    print(f'Size of {mode}:', len(idx))


# In[16]:


# Split dataframes to X_train, X_test, y_train, y_test.
X_train, X_val, X_test = df_receptors.loc[train_indices], df_receptors.loc[val_indices], df_receptors.loc[test_indices]
y_train, y_val, y_test = df_ligands.loc[train_indices], df_ligands.loc[val_indices], df_ligands.loc[test_indices]


# ## Pipelines

# In[17]:


# Pipeline that transforms X.
pipeline = Pipeline([
    ('reducer', PCA(n_components=30, random_state=1)),
    ('model', ExtraTreesRegressor(n_jobs=-1, random_state=1))
])

# Pipeline that transforms y.
transformer = Pipeline([
    ('reducer', PCA(n_components=5, random_state=1)),
])

# Combined pipeline.
reg = TransformedTargetRegressor(regressor=pipeline, transformer=transformer, check_inverse=False)


# In[18]:


# Fit the model.
reg.fit(X_train, y_train)


# ## Performance results

# ### Bar plot for top and worst predicted targets

# In[68]:


# Get the r2_score for each target.
# On test set only for the final results (publication).
#target_scores = r2_score(y_true=y_test, y_pred=reg.predict(X_test), multioutput='raw_values')
#target_names = y_test.columns

# On validation set.
target_scores = r2_score(y_true=y_val, y_pred=reg.predict(X_val), multioutput='raw_values')
target_names = y_val.columns

# Create a dataframe to store the results in sorted order.
df_scores = pd.DataFrame({'score': target_scores, 'ligand_descriptor': target_names})
df_scores.sort_values(by='score', inplace=True)


# In[56]:


#df_scores.to_csv('ligand_scores.csv', index=False)
df_scores


# In[29]:


# Dataframe is already sorted.
k = 5
worst = df_scores.iloc[:k]
best = df_scores.iloc[-k:]


# In[30]:


# Create the bar plot for top-5, worst-5 cases.
fig, ax = plt.subplots()

for i, mode in enumerate([worst, best]):
    label = 'worst' if i == 0 else 'best'
    ax.barh(mode.ligand_descriptor, mode.score, edgecolor='black', color=f'C{i}', label=f'{k}-{label}')

ax.set_title(f'Average $R^2$: {reg.score(X_val, y_val):.3f}')
ax.set_xlabel(r'$R^2$');
ax.legend();


# ### Learning curve

# In[19]:


# Training set sizes.
train_sizes = [100, 200, 300, 500, 1000, 2000, 5000, 7000, 10000, len(X_train)]

# Iterate over different training set and calculate train/test performance.
train_scores = []
test_scores = []
for size in train_sizes:
    # Fit the model.
    reg.fit(X_train[:size], y_train[:size])

    # Calculate test score.
    train_scores.append(reg.score(X_train[:size], y_train[:size]))
    test_scores.append(reg.score(X_test, y_test))


# In[20]:


# Learning curve.
fig, ax = plt.subplots()

ax.plot(train_sizes, train_scores, label='Train accuracy')
ax.plot(train_sizes, test_scores, label='Test accuracy')
ax.set_xlabel('Training set size')
ax.set_ylabel('$R^2$')
ax.set_title('Learning curve')
ax.legend()


# ## Similarity scores

# In[32]:


from sklearn.metrics.pairwise import linear_kernel, cosine_similarity, euclidean_distances


# In[99]:


# Define the function that returns the number of k-best matches.
def topk(Y, Y_pred, k=5, metric='cosine'):
    r"""
    Return the number of correct top-k matches.

    Parameters
    ----------
    Y : array_like of shape (n_samples, n_features)
        The array containing the ground truth values.
    Y_pred : array_like of shape (n_samples, n_features)
        The array containing the predictions.
    k : int, default=5
    metric : {'cosine', 'dot', 'euclidean'}, default='cosine'
        The similarity metric.

    Returns
    -------
    count : int
        The number of correct top-k matches.
    """
    distance_metrics = {
        'cosine': cosine_similarity,
        'dot' : linear_kernel,
        'euclidean': euclidean_distances,
    }

    distance_func = distance_metrics[metric]
    similarity = distance_func(X=Y, Y=Y_pred)

    # Negate the array to get sorting in descending order.
    order = 1 if metric == 'euclidean' else -1
    similarity_topk = np.argsort(order * similarity, axis=1)[:, :k]

    # Create dummy index to match compared to similarity_topk.
    index = np.tile(np.arange(len(similarity_topk)), reps=(k, 1)).T

    return (index == similarity_topk).sum()


# In[100]:


# Get the predictions.
y_pred = reg.predict(X_test)  # With val or test set.
topk(Y=y_test, Y_pred=y_pred, k=1)


# ### Measuring similarity with different scores

# In[95]:


# Calculate matches for different metrics and k values.
metrics = ['dot', 'cosine', 'euclidean']
k_values = [1, 5, 10, 20, 50]
scores = {m: [topk(Y=y_test, Y_pred=y_pred, metric=m, k=k) for k in k_values] for m in metrics}


# In[96]:


scores


# In[33]:


# Plot the results in a bar plot.
fig, ax = plt.subplots()

x = np.arange(len(k_values))
width = 0.25
multiplier = 0

for metric, matches in scores.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, matches, width, edgecolor='k', label=metric)
    ax.bar_label(rects, padding=2)
    multiplier += 1

ax.set_xlabel('k-best value')
ax.set_ylabel('Correct k-best matches')
ax.set_xticks(x + width, k_values)
ax.legend()


# ## Visualizing the dataset

# In[39]:


reducer = Pipeline([
    ('scaler', StandardScaler()),
    ('reducer', UMAP(n_jobs=-1)),
])

embeddings = reducer.fit_transform(X_train)


# In[58]:


# Show the embeddings.
fig, ax = plt.subplots()

# Choose property to color code the plot.
property = X_train.columns[150] # Change the number to select a different property.
properties = X_train.loc[:, property]

im = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=properties, alpha=1, s=0.8);
ax.set_xlabel('Embedding-1')
ax.set_ylabel('Embedding-2')

fig.colorbar(mappable=im, label=property, extend='max')


# In[ ]:




