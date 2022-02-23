"""This contains methods to extract custom fingerprints from the data.

As described here: https://www.rdkit.org/docs/GettingStartedInPython.html#chemical-features-and-pharmacophores
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import numpy as np
from DeepPurpose.utils import smiles2pubchem, smiles2morgan

## TODO: add herehttps://www.rdkit.org/docs/GettingStartedInPython.html#molecular-fragments
class MolecularFragments:
    def __init__(self):
        pass
    def __call__(self):
        pass

def get_custom_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pubchem = list(smiles2pubchem(smiles))

    maccs_fp = list(MACCSkeys.GenMACCSKeys(mol))
    morgan_fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 256))
    rdk_fp = list(Chem.RDKFingerprint(mol))
    return [int(x) for x in maccs_fp + morgan_fp + rdk_fp + pubchem]