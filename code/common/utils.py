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


def balance_data(y_train):
    df, counts = np.unique(y_train, return_counts=True)
    m = counts.max()
    index = np.arange(len(y_train))
    new_index = []
    for i, c in zip(df, counts):
        ids = y_train == i
        values = index[ids]
        if c == m:
            new_index.extend(values)
        else:
            new_index.extend(np.random.choice(values, m))
    np.random.shuffle(new_index)
    return new_index


def balance_data_df(train_df, target_col="Label"):
    return balance_data(train_df[target_col].values)

def train_cv(
        model_cls, dataset, n_splits=3, random_state=42,
        save_prefix="model_save", model_args=[], model_kwargs={}, model_random_state=None,
        strategy="stratified", group_column=None,
        fit_kwargs={},
        balance_train=False,
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
        if balance_train:
            balanced_index = balance_data(y_train)
            x_train = x_train[balanced_index]
            y_train = y_train[balanced_index]
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


    