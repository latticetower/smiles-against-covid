"""Scripts gets name of the .sdf file with 
embedded molecules as an input and
and path to the directory where the pharmacophore files 
will be saved.

Uses the fork of align-it software from the Oliver B. Scott repo:
https://github.com/OliverBScott/align-it/blob/main/example/pyalignit_demo.ipynb

"""
import click
import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm
import pyalignit

def load_molecules(filename):
    with Chem.SDMolSupplier(filename.as_posix()) as suppl:
        for mol in suppl:
            yield mol

@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
def main(path, output):
    path = Path(path)
    output = Path(output)

    output.mkdir(exist_ok=True)
    name, _ = os.path.splitext(path.name)
    print(f"Processing {name}")
    filename = output / f"alignit_{name}.phar"
    writer = pyalignit.PharmacophoreWriter(filename.as_posix())
    for i, mol in tqdm(enumerate(load_molecules(path))):
        # print(mol)
        p = pyalignit.CalcPharmacophore(mol)
        writer.write(p)
    writer.close()

if __name__ == "__main__":
    main()
