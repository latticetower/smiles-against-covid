import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolHash


def get_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return rdMolHash.MolHash(mol, rdMolHash.HashFunction.MurckoScaffold)


def smiles2canonical(smiles):
    m = AllChem.MolFromSmiles(smiles, sanitize=True)
    # m = AllChem.AddHs(m)
    # isomeric = AllChem.MolToSmiles(m, isomericSmiles=True)
    canonical = AllChem.MolToSmiles(m, isomericSmiles=False, canonical=True)
    return canonical