"""Methods for smiles cleaning and manipulating dataframes
"""
import re
import pandas as pd

square_brackets_regex = re.compile("\[([^\[\]]+)\]")
ATOM_NAMES = """
H He Li Be B C N O F Ne Na Mg Al Si P S
Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn
Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc
Ru Rh Pd Ag Cd In Sn Sb Te I Xe
Cs Ba La Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi
Po At Rn Fr Ra Ac Rf Ha Sg Ce Pr Nd
Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Th Pa U Np
Pu Am Cm Bk Cf Es Fm Md No Lr
"""
AROMATIC = "C,N,O,P,S,As,Se,\*".lower().split(",")
ATOM_NAMES = [name for line in ATOM_NAMES.split("\n") for name in line.split() if len(name) > 0]
ATOM_NAMES = sorted(ATOM_NAMES, key=len, reverse=True)
name_regex = "{}|H|C|N|O|S|P|F|Cl|Br|I|{}".format( "|".join(ATOM_NAMES), "|".join(AROMATIC))
name_regex = re.compile(name_regex)


def find_square_brackets(smiles):
    return square_brackets_regex.findall(smiles)


def break_to_parts(smiles):
    return smiles.split(".")


def clean_smiles(smiles):
    parts = smiles.split(".")
    parts = sorted(set(parts))
    if len(parts) == 1:
        return ".".join(parts)
    resulting_parts = []
    for part in parts:
        atom_names = name_regex.findall(part)
        has_carbon = len([name for name in atom_names if name.lower() == "c"])
        num_atoms = len(atom_names)
        if has_carbon or num_atoms > 1:
            resulting_parts.append(part)
    if len(resulting_parts) == 0:
        print("Cannot parse", parts, atom_names)
        for part in parts:
            atom_names = name_regex.findall(part)
            num_atoms = len(set(atom_names))
            if len(atom_names) > 3:
                resulting_parts.append(part)
        if len(resulting_parts) == 0:
            resulting_parts = parts
    smiles_new = ".".join(resulting_parts)
    return smiles_new


def split_df(df, target_col="Active", split_col="parts",
             index_col="original_index",
             smiles_col="Smiles",
             keep_columns=["Smiles"],
             renames={}):
    """splits dataset to submolecules"""
    def do_transform(row, parts_col="parts", target_col="Active", keep_columns=["Smiles"]):
        parts = row[parts_col]
        new_data = {}
        for i, part in enumerate(parts):
            prefix = f"part{i}"
            new_data[f"{prefix}_part"] = part
            new_data[f"{prefix}_num_{split_col}"] = len(parts)
            if target_col in row:
                new_data[f"{prefix}_{target_col}"] = row[target_col]
            for col in keep_columns:
                if col in row:
                    name = renames.get(col, col)
                    new_data[f"{prefix}_original_{name}"] = row[col]
        return new_data

    if index_col in df.columns:
        print("Column", index_col, "is defined, doing nothing")
        return df
    if split_col not in df.columns:
        df[split_col] = df[smiles_col].apply(break_to_parts)

    df = df.apply(
        lambda x: do_transform(
                x, parts_col=split_col,
                target_col=target_col, keep_columns=keep_columns
            ),
        axis=1,
        result_type='expand'
    )
    col_index = [(x[:x.find("_")+1], x[x.find("_")+1:]) for x in df.columns]
    df.columns = pd.MultiIndex.from_arrays(zip(*col_index))
    df = df.stack(0).reset_index().drop('level_1', axis=1).rename(columns={"level_0": index_col})
    df = df.loc[~df.part.isnull()].reset_index(drop=True)
    #for name, replacement in renames.items():
    #if "part" in renames:
    #df.rename(columns={"part": renames["part"]}, inplace=True)
    renames = {k:v for k, v in renames.items() if k in df.columns}
    df.rename(columns=renames, inplace=True)
    if target_col in df.columns:
        df[target_col] = df[target_col].astype(bool)
    return df


def collect_df(df, target_col="Active", split_col="parts", 
               index_col="original_index", split_sep=".",
               keep_columns=["original_Smiles", "Smiles"]):
    """collects submolecules to dataset"""
    aggregations = dict()
    if split_col in df.columns:
        print("Found split col:", split_col)
        aggregations[split_col] = lambda x: (x if isinstance(x, str) else split_sep.join(x))
    if target_col in df.columns:
        print("Found", target_col)
        aggregations[target_col] = lambda x: x.any()
    for col in keep_columns:
        if col  in df.columns:
            aggregations[col] = lambda x: x.iloc[0] if x.shape[0] > 0 else None
    return df.groupby(index_col).agg(aggregations)