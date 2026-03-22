import numpy as np
from tqdm import tqdm
import pyarrow.parquet as pq
import pandas as pd
import joblib

X_pocket = pd.read_csv('../data/Final_Receptor_dataset.csv', index_col='id').loc[['1lox']]
model = joblib.load('../saved_models/best_model.joblib')

# Iterate over PubChem in batches and make predictions
pq_file = pq.ParquetFile('../data/ligands_pubchem.parquet')
pq_iter = pq_file.iter_batches(batch_size=10_000)

cids = []
preds = []
for b in tqdm(pq_iter, desc='Screening Pubchem'):
    X_ligands = b.to_pandas()

    X_input = pd.concat([X_pocket] * len(X_ligands))
    X_input.index = X_ligands.index

    X_input = pd.concat([X_input, X_ligands], axis=1)
    X_input.dropna(inplace=True)

    preds.append(model.predict_proba(X_input)[:, 1])
    cids.extend(list(X_input.index))

df = pd.DataFrame(dict(Score=np.concat(preds)), index=cids)
df.to_csv('../results/pubchem_1lox_ligand_scores.csv', index_label='CID')
