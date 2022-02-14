import json
import os
from tqdm.auto import tqdm
import numpy as np
import pubchempy as pcp
from DeepPurpose.utils import smiles2pubchem, smiles2morgan


def get_compounds_fingerprints(df, cache_dir="temp/train", smiles_column="Smiles", additional_cols=[]):
    """Downloads precomputed fingerprints for compounds from pubchem and saves them to
    """
    fingerprints = []
    for i in tqdm(df.index, total=df.shape[0]):
        smiles = df.loc[i, smiles_column]
        fingerprint = smiles2pubchem(smiles)
        fingerprint2 = smiles2morgan(smiles)
        data = {
            smiles_column: smiles,
            "fingerprint": [int(x) for x in fingerprint] + [int(x) for x in fingerprint2],
            "cactvs": [int(x) for x in fingerprint],
            "morgan": [int(x) for x in fingerprint2],
        }
        for col in additional_cols:
            if not col in data and col in df.columns:
                data[col] = df.loc[i, col]
        fingerprints.append(data)
    return fingerprints

def to_bits(x):
    return np.asarray(x)

def get_compounds_fingerprints_old(df, cache_dir="temp/train", smiles_column="Smiles"):
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
                    # if not 'fingerprint_bits' in data:
                    #     smiles = data[smiles_column]
                    #     pubchem_fingerprint = smiles2pubchem(smiles)
                    #     data["fingerprint_bits"] = pubchem_fingerprint
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
        #pubchem_fingerprint = smiles2pubchem(smiles)
        if compound.fingerprint is None:
            print(f"No fingerprint for molecule {i} is found, skipping" )
            with open(fingerprint_path, 'w') as f:
                data = {
                    smiles_column: smiles,
                    "fingerprint": None,

                }
                json.dump(data, f)
        else:
            # cactvs_fingerprint contains the same bits as fingerprint property, 
            # the only difference is representation
            with open(fingerprint_path, 'w') as f:
                data = {
                    smiles_column: smiles,
                    "fingerprint": compound.fingerprint,
                    # "fingerprint_bits": pubchem_fingerprint,
                }
                json.dump(data, f)

    return fingerprints
        #unpacked = np.unpackbits(np.frombuffer(bytes.fromhex(x), dtype=np.uint8))


def to_bits_old(x):
    try:
        unpacked = np.frombuffer(
            bytes.fromhex(x[8:]),
            dtype=np.uint8
        )
        unpacked = np.unpackbits(unpacked)
        unpacked = unpacked[:-7]
    except Exception as e:
        print(e)
        print(x)
    return unpacked