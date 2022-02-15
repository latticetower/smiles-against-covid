"""
Regular script, which should be run from the current directory.
This precomputes pharmacophore features for all molecules
in the train and test datasets and saves them.

Uses the fork of align-it software from the Oliver B. Scott repo:
https://github.com/OliverBScott/align-it/blob/main/example/pyalignit_demo.ipynb
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
import pyalignit

# Cell with constants
SEED = 2407
np.random.seed(SEED)
random.seed(SEED)

DATADIR = Path("../../data")
train_df = pd.read_csv(DATADIR / "train.csv", index_col=0)
test_df = pd.read_csv(DATADIR / "test.csv", index_col=0)

OUTPUTDIR = Path("../../tmp/pharmacophores")
OUTPUTDIR.mkdir(exist_ok=True)

MOLDIR = OUTPUTDIR/ "molecules"
MOLDIR.mkdir(exist_ok=True)

def make_molecule(smiles, seed=42):
    ref = AllChem.MolFromSmiles(smiles)
    ref = Chem.AddHs(ref)
    AllChem.EmbedMolecule(ref, randomSeed=seed,
        enforceChirality=True,
        useRandomCoords=True,
        useBasicKnowledge=True,
        )
    return ref

molecules = {
    "train": [],
    "test": [],
}

for name, df in [("train", train_df), ("test", test_df)]:
    print("-------------\n-------\n---Starting to process", name)
    # pos_mols = dict()
    for smiles in tqdm(df.Smiles):
        # smiles = train_df.loc[i, "Smiles"]
        mol = make_molecule(smiles)
        # pos_mols[i] = mol
        molecules[name].append(mol)

print("Saving everything...")
for name in ["train", "test"]:
    with Chem.SDWriter((MOLDIR/ f'{name}.sdf').as_posix()) as w:
        for m in molecules[name]:
            w.write(m)



print("-------------\n-------\n---Computing and saving pharmacophores")

for name in ["train", "test"]:
    writer = pyalignit.PharmacophoreWriter(MOLDIR/f'{name}.phar')
    for mol in tqdm(molecules[name]):
        pharm = pyalignit.CalcPharmacophore(mol)
        writer.write(pharm)
    writer.close()