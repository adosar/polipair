import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def canonicalize_smiles(smiles: str):
    return Chem.CanonSmiles(smiles)


def smiles_to_desc(smiles: str):
    mol = Chem.MolFromSmiles(canonicalize_smiles(smiles))
    if mol is None:
        return [None]*len(descriptor_names)
    return list(descriptors.ComputeProperties(mol))


def ligands_to_desc(ligands: list[str]):
    return pd.DataFrame(
        [smiles_to_desc(l) for l in ligands],
        index=ligands,
        columns=descriptor_names
    )
