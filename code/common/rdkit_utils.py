import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolHash
from standardiser import standardise

def get_murcko_scaffold(smiles, kekulize=True):
    mol = Chem.MolFromSmiles(smiles)    
    if kekulize:
        Chem.Kekulize(mol)
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
    
    return rdMolHash.MolHash(mol, rdMolHash.HashFunction.MurckoScaffold)


def smiles2canonical(smiles, kekulize=True):
    m = AllChem.MolFromSmiles(smiles, sanitize=True)
    # m = AllChem.AddHs(m)
    # isomeric = AllChem.MolToSmiles(m, isomericSmiles=True)
    canonical = AllChem.MolToSmiles(m, isomericSmiles=False, canonical=True)
    return canonical

def smiles2cleaned(smiles):
    mol = AllChem.MolFromSmiles(smiles)
    parent = None
    try:
        parent = standardise.run(mol)
    except standardise.StandardiseException as e:
        print(e.message)
        parent = mol
    try:
        parent = AllChem.MolToSmiles(parent, isomericSmiles=True, kekuleSmiles=False)
    except Exception as e:
        print("to smiles:", e.message)
        parent = smiles
    return parent