import json
import os
from tqdm.auto import tqdm
import numpy as np
import pubchempy as pcp

def get_compounds_fingerprints(df, cache_dir="temp/train", smiles_column="Smiles"):
    """Downloads precomputed fingerprints for compounds from pubchem and saves them to cache.
    Returned cached versions.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        # we'll use this directory to cache downloaded fingerprints

    fingerprints = []
    for i in tqdm(df.index, total=df.shape[0]):
        fingerprint_path = os.path.join(cache_dir, f"fingerprint_{i}.json")
        if os.path.exists(fingerprint_path):
            with open(fingerprint_path) as f:
                try:
                    data = json.load(f)
                    fingerprints.append(data)
                except:
                    print(f"Error {i}")
            continue
        smiles = df.loc[i, smiles_column]
        try:
            compounds = pcp.get_compounds(smiles, 'smiles')
            compound = compounds[0]
        except Exception as e:
            print(f"Got error while loading {i} with smiles string {repr(smiles)}, error: {e}")
            continue

        if compound is None:
            print(f"No compound {i} found, skipping molecule {smiles}" )
            continue
        if compound.fingerprint is None:
            print(f"No fingerprint for molecule {i} is found, skipping" )
            with open(fingerprint_path, 'w') as f:
                data = {
                    smiles_column: smiles,
                    "fingerprint": None
                }
                json.dump(data, f)
        else:
            # cactvs_fingerprint contains the same bits as fingerprint property, 
            # the only difference is representation
            with open(fingerprint_path, 'w') as f:
                data = {
                    smiles_column: smiles,
                    "fingerprint": compound.fingerprint
                }
                json.dump(data, f)

    return fingerprints


def to_bits(x):
    try:
        unpacked = np.unpackbits(np.frombuffer(bytes.fromhex(x), dtype=np.uint8))
    except Exception as e:
        print(e)
        print(x)
        
    return unpacked