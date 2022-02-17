import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from rdkit.Chem import AllChem


def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    pass


def train_cv(
        model_cls, dataset, n_splits=3, random_state=42,
        save_prefix="model_save", model_args=[], model_kwargs={}, model_random_state=None,
        strategy="stratified", group_column=None,
        fit_kwargs={}
    ):

    x_full_train, y_full_train = dataset
    split_data = dict()

    if strategy == "stratified":
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        kfsplits = kf.split(x_full_train, y_full_train)
    elif strategy == "stratified+grouped":
        kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        kfsplits = kf.split(x_full_train, y_full_train, groups=group_column)

    # cat_features=np.arange(x_full_train.shape[1])
    # NITERATIONS=1000
    for fold, (train_index, test_index) in enumerate(kfsplits):
        split_data[fold] = (train_index, test_index)
        x_train = x_full_train[train_index]
        y_train = y_full_train[train_index]
        x_val = x_full_train[test_index]
        y_val = y_full_train[test_index]
        if model_random_state is not None:
            model_kwargs[model_random_state] = random_state + fold + 1
        # next model-specific
        model = model_cls(*model_args, **model_kwargs)
        #train the model
        model.fit(x_train, y_train, eval_set=(x_val, y_val), **fit_kwargs)
        # make the prediction using the resulting model
        pred_train = model.predict_proba(x_train)
        pred_val = model.predict_proba(x_val)

        # path = f"{save_prefix}{fold}.cbm"
        model.save_fold(save_prefix, fold)
        yield (train_index, x_train, y_train, pred_train), (test_index, x_val, y_val, pred_val)

def eval_cv(model_cls, dataset, n_splits=3, random_state=42, save_prefix="model_save"):
    for fold in range(n_splits):
        model = model_cls.load_fold(save_prefix, fold)
        predictions = model.predict_proba(dataset)
        yield predictions


def smiles2canonical(smiles):
    m = AllChem.MolFromSmiles(smiles, sanitize=True)
    # m = AllChem.AddHs(m)
    # isomeric = AllChem.MolToSmiles(m, isomericSmiles=True)
    canonical = AllChem.MolToSmiles(m, isomericSmiles=False, canonical=True)
    return canonical
    
