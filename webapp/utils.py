from stqdm import stqdm
import numpy as np
import joblib
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pyarrow.parquet as pq


descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
descriptors = rdMolDescriptors.Properties(descriptor_names)


def canonicalize_smiles(smiles: str):
    try:
        return Chem.CanonSmiles(smiles)
    except:
        return None


def smiles_to_desc(smiles: str):
    inp_smiles = canonicalize_smiles(smiles)

    if inp_smiles is None:
        return [None] * len(descriptor_names)
    else:
        mol = Chem.MolFromSmiles(inp_smiles)
        return list(descriptors.ComputeProperties(mol))


def ligands_to_desc(ligands: list[str]):
    return pd.DataFrame(
        [smiles_to_desc(l) for l in ligands],
        index=ligands,
        columns=descriptor_names
    )


@st.cache_data
def load_csv(**kwargs):
    return pd.read_csv(**kwargs)


def load_pubchem():
    pq_file = pq.ParquetFile('data/ligands_pubchem.parquet')
    return pq_file.iter_batches(batch_size=10_000)


@st.cache_resource
def load_model():
    return joblib.load('saved_models/best_model.joblib')


def prepare_inputs(X_poc, X_lig):
    X_input = pd.concat([X_poc] * len(X_lig))
    X_input.index = X_lig.index

    X_input = pd.concat([X_input, X_lig], axis=1)
    X_input.dropna(inplace=True)

    return X_input


def predict(X_poc, X_lig):
    model = load_model()

    # Ligands from user
    if isinstance(X_lig, pd.DataFrame):
        X_input = prepare_inputs(X_poc, X_lig)
        preds = model.predict_proba(X_input)[:, 1]

        return pd.DataFrame(dict(Score=preds), index=list(X_input.index))

    # Ligands from PubChem
    else:
        total_batches = 600  # 6_004_131 rows, 10_000 batch size
        cids = []
        preds = []

        for b in stqdm(X_lig, total=total_batches, desc='Screening PubChem'):
            b_lig = b.to_pandas()
            X_input = prepare_inputs(X_poc, b_lig)

            preds.append(model.predict_proba(X_input)[:, 1])
            cids.extend(list(X_input.index))

        return pd.DataFrame(dict(Score=np.concat(preds)), index=cids)
