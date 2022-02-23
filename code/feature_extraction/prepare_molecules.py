"""
Regular script, which should be run from the current directory.
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from pathlib import Path
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import RDConfig
from rdkit.Chem import rdMolTransforms
from tqdm.auto import tqdm
import random
import json
import pyalignit

# Cell with constants
SEED = 2407
np.random.seed(SEED)
random.seed(SEED)

DATADIR = Path("../../data")
train_df = pd.read_csv(DATADIR / "train_splitted_val.csv", index_col=0)
test_df = pd.read_csv(DATADIR / "test_splitted.csv", index_col=0)

OUTPUTDIR = Path("../../tmp/pharmacophores")
OUTPUTDIR.mkdir(exist_ok=True)

MOLDIR = OUTPUTDIR/ "molecules_10"
MOLDIR.mkdir(exist_ok=True)

def make_molecule(smiles, seed=42, num_confs=3):
    ref = AllChem.MolFromSmiles(smiles)
    ref = Chem.AddHs(ref)
    params = AllChem.ETKDGv3()
    params.useSmallRingTorsions = True
    params.randomSeed = seed
    params.randomConformers = num_confs
    params.useBasicKnowledge = True
    params.enforceChirality = True
    #params.useExpTorsionAnglePrefs = True
    conf_ids = AllChem.EmbedMultipleConfs(
        ref, numConfs=num_confs, params=params
    )
    conf_ids = list(conf_ids)
    # conf_id = AllChem.EmbedMolecule(ref, params)
    # AllChem.EmbedMolecule(ref, randomSeed=seed,
    #     enforceChirality=True,
    #     useRandomCoords=True,
    #     useBasicKnowledge=True,
    #     params=params
    #     )
    return ref, conf_ids

molecules = {
    "train": [],
    "test": [],
}
processed = {
    "train": [],
    "test": [],
}
num_confs = 10
for name, df in [("train", train_df), ("test", test_df)]:
    print("-------------\n-------\n---Starting to process", name)
    # pos_mols = dict()
    for i, (smiles, active) in tqdm(enumerate(df[["part", "Active"]].values), total=df.shape[0]):
        # smiles = train_df.loc[i, "Smiles"]
        mol, conf_id = make_molecule(smiles, seed=SEED + i, num_confs=num_confs)
        activity = "active" if active else "inactive"
        mol.SetProp("_Name", f"molecule_{i}-{activity}")
        # pos_mols[i] = mol
        if len(conf_id) == num_confs:
            processed[name].append(i)
            molecules[name].append(mol)

print("Saving everything...")
with open((MOLDIR/"conformers_info.json").as_posix(), "w") as f:
    json.dump(processed, f)


for name in ["train", "test"]:
    with Chem.SDWriter((MOLDIR/ f'{name}.sdf').as_posix()) as w:
        for m in molecules[name]:
            w.write(m)



# print("-------------\n-------\n---Computing and saving pharmacophores")

# for name in ["train", "test"]:
#     writer = pyalignit.PharmacophoreWriter(MOLDIR/f'{name}.phar')
#     for mol in tqdm(molecules[name]):
#         pharm = pyalignit.CalcPharmacophore(mol)
#         writer.write(pharm)
#     writer.close()