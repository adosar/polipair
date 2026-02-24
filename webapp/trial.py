#!/usr/bin/env python
# coding: utf-8

# #Install micromamba

# In[ ]:


get_ipython().system('curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba')
get_ipython().system('./bin/micromamba --version')


# #Create an env with Python 3.11 + RDKit
# 

# In[ ]:


get_ipython().system('./bin/micromamba create -y -n rdkit -c conda-forge python=3.11 rdkit numpy pandas')


# #Install/update libstdc++ inside the micromamba env

# In[ ]:


# Make sure the env has the modern C++ runtime libraries
get_ipython().system('./bin/micromamba install -y -n rdkit -c conda-forge "libstdcxx-ng>=13" "libgcc-ng>=13"')


# #Check we have an RDKit version (e.g. 2025.09.5)

# In[ ]:


get_ipython().system('./bin/micromamba run -n rdkit python -c "from rdkit import Chem; print(\'RDKit OK\', Chem.rdBase.rdkitVersion)"')


# # Write a small RDKit “worker” script (runs inside the env)

# In[ ]:


get_ipython().run_cell_magic('writefile', '/content/rdkit_worker.py', 'import argparse, os\nimport numpy as np\nimport pandas as pd\nfrom rdkit import Chem\nfrom rdkit.Chem import rdMolDescriptors\n\ndescriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())\nget_descriptors = rdMolDescriptors.Properties(descriptor_names)\n\ndef canonicalize_smiles(s: str):\n    if s is None:\n        return None, False, False\n    s = str(s).strip()\n    if s == "":\n        return None, False, False\n    try:\n        can = Chem.CanonSmiles(s)\n        return can, (can != s), True\n    except Exception:\n        return None, False, False\n\ndef smiles_to_desc(smiles: str):\n    if smiles is None or str(smiles).strip() == "":\n        return [None]*len(descriptor_names)\n    mol = Chem.MolFromSmiles(str(smiles))\n    if mol is None:\n        return [None]*len(descriptor_names)\n    return list(get_descriptors.ComputeProperties(mol))\n\ndef main():\n    ap = argparse.ArgumentParser()\n    ap.add_argument("--mode", choices=["single", "csv"], required=True)\n    ap.add_argument("--smiles", default="")\n    ap.add_argument("--csv_in", default="")\n    ap.add_argument("--out", required=True)\n    ap.add_argument("--id_col", default="")\n    args = ap.parse_args()\n\n    if args.mode == "single":\n        smi_in = args.smiles\n        can, changed, ok = canonicalize_smiles(smi_in)\n        use = can if ok else None\n        row = {\n            "id": "input_1",\n            "smiles_input": smi_in,\n            "smiles_canonical": can,\n            "canonicalization_ok": ok,\n            "was_changed": changed if ok else None,\n        }\n        desc = smiles_to_desc(use)\n        df = pd.DataFrame([row])\n        df = pd.concat([df, pd.DataFrame([desc], columns=descriptor_names)], axis=1)\n        df.to_csv(args.out, index=False)\n        return\n\n    # CSV mode\n    df_in = pd.read_csv(args.csv_in)\n    if "smiles" not in df_in.columns:\n        raise SystemExit(f"CSV must contain a column named \'smiles\'. Found: {list(df_in.columns)}")\n\n    if args.id_col and args.id_col in df_in.columns:\n        ids = df_in[args.id_col].astype(str)\n    else:\n        ids = pd.Series([f"row_{i}" for i in range(len(df_in))])\n\n    can_list, changed_list, ok_list = [], [], []\n    for s in df_in["smiles"].tolist():\n        can, changed, ok = canonicalize_smiles(s)\n        can_list.append(can)\n        changed_list.append(changed if ok else None)\n        ok_list.append(ok)\n\n    df = pd.DataFrame({\n        "id": ids,\n        "smiles_input": df_in["smiles"],\n        "smiles_canonical": can_list,\n        "canonicalization_ok": ok_list,\n        "was_changed": changed_list,\n    })\n\n    desc_rows = []\n    for ok, s_can in zip(df["canonicalization_ok"], df["smiles_canonical"]):\n        desc_rows.append(smiles_to_desc(s_can if ok else None))\n\n    df = pd.concat([df, pd.DataFrame(desc_rows, columns=descriptor_names)], axis=1)\n    df.to_csv(args.out, index=False)\n\nif __name__ == "__main__":\n    main()\n')


# #Reset pandas/numpy to Colab-compatible versions

# In[ ]:


get_ipython().system('pip -q uninstall -y pandas numpy')
get_ipython().system('pip -q install "numpy==2.0.2" "pandas==2.2.2"')


# #Remember to restart session!! From the top Runtime -> Restart session

# In[ ]:


# Check I have the correct versions for numpy and pandas I installed above
import numpy as np, pandas as pd
print("numpy:", np.__version__)
print("pandas:", pd.__version__)


# In[ ]:


#Link to your gdrive

from google.colab import drive
drive.mount('/content/drive')


# #For single SMILES. Adjust the input SMILES code and the output path of the csv

# In[ ]:


# For a single SMILES

smiles = "[H]C1:C([H]):C([H]):C(C([H])([H])C([H])(C([H])([H])[H])C([H])([H])[H]):C([H]):C:1[H]" #adjust to the SMILES you want
out_path = "/content/drive/MyDrive/Ligand-Pocket Data/rdkit_features_single.csv" # adjust to have the output path on your gdrive

get_ipython().system('./bin/micromamba run -n rdkit python /content/rdkit_worker.py    --mode single --smiles "{smiles}" --out "{out_path}"')

import pandas as pd
df = pd.read_csv(out_path)
df.head()





# #For single SMILES. Adjust the input csv containing a column named 'smiles' with SMILES codes and the output path of the csv
# 

# In[ ]:


csv_in = "/content/drive/MyDrive/Ligand-Pocket Data/Ligand_dataset_SMILESonly.csv" #adjust to the csv file containing many SMILES in column named 'smiles'
out_path = "/content/drive/MyDrive/Ligand-Pocket Data/rdkit_features.csv" # adjust to have the output path on your gdrive

get_ipython().system('./bin/micromamba run -n rdkit python /content/rdkit_worker.py    --mode csv --csv_in "{csv_in}" --out "{out_path}" --id_col ""')

import pandas as pd
pd.read_csv(out_path).head()

