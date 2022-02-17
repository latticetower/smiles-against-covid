import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolHash


def get_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return rdMolHash.MolHash(mol, rdMolHash.HashFunction.MurckoScaffold)