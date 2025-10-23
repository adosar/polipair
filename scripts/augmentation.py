#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# For reproducibility.
np.random.seed(1)
random.seed(1)

# Avoid inplace modifications data in dataframes.
pd.options.mode.copy_on_write = True


# # Existing dataset

# In[3]:


df_receptors = pd.read_csv('data/Final_Receptor_dataset.csv', index_col='id')
df_ligands = pd.read_csv('data/Final_Ligand_dataset.csv', index_col='id')


# In[4]:


#ligands_drop_indices = list(set(df_ligands.index) - set(df_receptors.index))
#receptors_drop_indices = list(set(df_receptors.index) - set(df_ligands.index))
#
#df_ligands.drop(ligands_drop_indices, inplace=True)
#df_receptors.drop(receptors_drop_indices, inplace=True)
#df_receptors = df_receptors.loc[common_indices]
#df_ligands = df_ligands.loc[common_indices]

# Use only the common indices.
common_indices = list(set(df_ligands.index).intersection(set(df_receptors.index)))
df_receptors = df_receptors.loc[common_indices]
df_ligands = df_ligands.loc[common_indices]


# In[5]:


# Drop the smiles column.
df_ligands.drop('smiles', axis=1, inplace=True)


# In[6]:


# Check that they have the correct ordering.
print(f'Indices are the same: {np.all(df_receptors.index == df_ligands.index)}')


# In[7]:


feats_receptor = list(df_receptors.columns)
feats_ligand = list(df_ligands.columns)

print(f'Number of receptor features: {len(feats_receptor)}')
print(f'Number of ligand features: {len(feats_ligand)}')


# ### Concatenate all features into a single dataframe

# In[8]:


df = pd.concat((df_receptors, df_ligands), axis=1, join='inner')


# In[9]:


# The concatenated features should be the new columns.
feats = feats_receptor + feats_ligand

print(f'Number of complex (receptor-ligand) features: {len(feats)}')
print(f'Concatenation is correct: {list(df.columns) == feats}')


# ### Add a class column with 1's

# In[10]:


df['class'] = 1


# # Augmented dataset

# Since ``df_receptors`` and ``df_ligands`` have the same ordering we can create invalid pairs as following:
# 1. Shuffle both of them (remove any bias in the original dataset) and reverse one of them.
# 2. Create new indices of the form ``receptor_ligand``.
# 3. Concatenate the new dataframes.
# 4. Add a class column with all ``0``.
# 5. Concatenate with the original dataset.

# ### 1. Shuffle the dataframes and reverse one of them

# In[11]:


df_receptors_shuffled, df_ligands_shuffled = shuffle(df_receptors, df_ligands, random_state=1)
df_ligands_shuffled = df_ligands_shuffled[::-1]


# ### 2. Create and set new indices

# In[12]:


receptors_idx = list(df_receptors_shuffled.index)
ligands_idx = list(df_ligands_shuffled.index)

new_indices = [f'{receptor}_{ligand}' for receptor, ligand in zip(receptors_idx, ligands_idx)]

# It is necessary to use the same indices, otherwise pd.concat will produce incorrect results.
for frame in (df_receptors_shuffled, df_ligands_shuffled):
    frame.set_index(np.array(new_indices), inplace=True)

# DO NOT USE df_receptors_shuffled and df_ligands_shuffled for retrieving data.
# USE df_receptors and df_ligands instead.


# ### 3. Concatenate them

# In[13]:


df_invalid = pd.concat((df_receptors_shuffled, df_ligands_shuffled), axis=1, join='inner')


# ### 4. Add a class column with 0's

# In[14]:


df_invalid['class'] = 0


# ### 5. Concatenate with the original dataset

# In[15]:


df_augmentation = pd.concat((df, df_invalid), axis=0)


# # Store the new dataset

# In[16]:


df_augmentation.to_csv('data/augmented_dataset.csv', index=True, index_label='id')


# In[19]:


df_invalid.head(1)


# In[20]:


df_receptors.loc[['5a3h']]


# In[24]:


df_ligands.loc[['2pt9']]

