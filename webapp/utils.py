import joblib
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
descriptors = rdMolDescriptors.Properties(descriptor_names)


def canonicalize_smiles(smiles: str):
    return Chem.CanonSmiles(smiles)


def smiles_to_desc(smiles: str):
    mol = Chem.MolFromSmiles(canonicalize_smiles(smiles))
    if mol is None:
        return [None] * len(descriptor_names)
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


@st.cache_resource
def load_model():
    return joblib.load('../saved_models/best_model.joblib')


@st.cache_data
def prepare_inputs(X_poc, X_lig):
    X_input = pd.concat([X_poc] * len(X_lig), ignore_index=True)
    X_input.index = X_lig.index

    X_input = pd.concat([X_input, X_lig], axis=1)
    X_input.dropna(inplace=True)

    return X_input
