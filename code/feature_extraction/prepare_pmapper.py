"""Scripts gets name of the .sdf file with 
embedded molecules as an input and
and path to the directory where the pharmacophore files 
will be saved.

Produces pharmacophores defined at https://github.com/DrrDom/pmapper
"""
import click
import os
from pathlib import Path
from rdkit import Chem
from tqdm.auto import tqdm
from pmapper.pharmacophore import Pharmacophore as P

def load_molecules(filename):
    with Chem.SDMolSupplier(filename.as_posix()) as suppl:
        for mol in suppl:
            yield mol

@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
def main(path, output):
    output.mkdirs(exist_ok=True)
    name, _ = os.path.splitext(path.name)
    print(f"Processing {name}")
    for i, mol in tqdm(enumerate(load_molecules(path))):
        # print(mol)
        filename = output / f"{name}_{i}.pma"
        p = P()
        p.load_from_mol(mol)
        p.save_to_pma(filename.as_posix())

if __name__ == "__main__":
    main()