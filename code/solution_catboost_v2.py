"""
"""

import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
import json
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import roc_curve, f1_score

from common.utils import seed_everything, train_cv, eval_cv, get_threshold
from common.rdkit_utils import smiles2canonical, smiles2cleaned
from common.pubchem import get_compounds_fingerprints, to_bits
from common.fingerprints import get_custom_fingerprint
from common.rdkit_utils import get_murcko_scaffold
from models import CatboostClassifierWrapper
from common.cleaner import split_df, collect_df, clean_smiles


def get_predictions(
            model_cls=CatboostClassifierWrapper, test_data=None, n_splits=3,
            save_prefix="model_save",
            mode=("mean", "ones")#("mean", "argmax",)
        ):
    all_predictions = []
    for predictions in eval_cv(
            model_cls,
            test_data, n_splits=n_splits,
            save_prefix=save_prefix):
        all_predictions.append(predictions)
    all_predictions = np.stack(all_predictions)
    if "mean" in mode:
        all_predictions = all_predictions.mean(0)
    if "argmax" in mode:
        if all_predictions.shape[-1] == 2:
            all_predictions = np.argmax(all_predictions, axis=-1)
    if "ones" in mode:
        if all_predictions.shape[-1] == 2:
            all_predictions = all_predictions[..., 1]

    return all_predictions


if __name__ == '__main__':
    SEED = 2407
    NFOLDS = 5
    NITERATIONS = 1000
    TMP_DIR = Path("../tmp/")
    SAVE_DIR = Path("../weights/")
    SAVE_DIR.mkdir(exist_ok=True)
    MODEL_PREFIX = (SAVE_DIR/ "cbt_fold_").as_posix()
    seed_everything(SEED)
    SMILES_COL = "part"  # before was: "canonical"
    # print(train_df)
    FINGERPRINT_COL = "fingerprint"
    RECOMPUTE = False

    train_df = pd.read_csv(Path("../data/train.csv"), index_col=0)
    test_df = pd.read_csv(Path("../data/test.csv"), index_col=0)
    train_df_val = pd.read_csv(Path("../data/train_splitted_val.csv"), index_col=0)
    # clean up everything - convert to canonical smiles
    train_df["cleaned"] = train_df.Smiles.apply(clean_smiles)
    test_df["cleaned"] = test_df.Smiles.apply(clean_smiles)

    train_df_splitted = split_df(
        train_df, smiles_col="cleaned",
        keep_columns=["Smiles", "cleaned"],
        # renames={"part": "cleaned"}
    )
    test_df_splitted = split_df(
        test_df, smiles_col="cleaned",
        keep_columns=["Smiles", "cleaned"],
        # renames={"part": SMILES_COL}
    )
    print(train_df_splitted.columns)
    print(
        train_df_splitted.part.apply(lambda x: x.find(".") >= 0).any(),
        test_df_splitted.part.apply(lambda x: x.find(".") >= 0).any(),
    )

    train_df_splitted['canonical'] = train_df_splitted["part"].apply(smiles2canonical)
    test_df_splitted['canonical'] = test_df_splitted["part"].apply(smiles2canonical)

    # train_df['cleaned'] = train_df.Smiles.apply(smiles2cleaned)
    # test_df['cleaned'] = test_df.Smiles.apply(smiles2cleaned)
    train_df_splitted["murcko_cleaned"] = train_df_splitted.part.apply(get_murcko_scaffold)
    train_df_splitted["murcko"] = train_df_splitted.part.apply(get_murcko_scaffold)
    # test_df["murcko"] = test_df.canonical.apply(get_murcko_scaffold)

    TRAIN_FPATH = TMP_DIR/"train_fingerprints_new.json"
    if TRAIN_FPATH.exists() and not RECOMPUTE:
        with open(TRAIN_FPATH.as_posix()) as f:
            train_fingerprints = json.load(f)
    else:
        train_fingerprints = []
        for smiles, i in tqdm(zip(train_df_splitted[SMILES_COL].values, train_df_splitted.index)):
            custom_fp = get_custom_fingerprint(smiles)
            data = {
                SMILES_COL: smiles,
                "fingerprint": list(custom_fp)
            }
            for col in ["original_Smiles", "part", "Smiles"]:
                if not col in data and col in train_df_splitted.columns:
                    data[col] = train_df_splitted.loc[i, col]
            if "original_index" in train_df_splitted.columns:
                data["original_index"] = int(train_df_splitted.loc[i, "original_index"])
            train_fingerprints.append(data)
        with open(TRAIN_FPATH.as_posix(), 'w') as f:
            json.dump(train_fingerprints, f)
    TEST_FPATH = TMP_DIR/"test_fingerprints_new.json"
    if TEST_FPATH.exists() and not RECOMPUTE:
        with open(TEST_FPATH.as_posix()) as f:
            test_fingerprints = json.load(f)
    else:
        test_fingerprints = []
        for smiles, i in tqdm(zip(test_df_splitted[SMILES_COL].values, test_df_splitted.index)):
            custom_fp = get_custom_fingerprint(smiles)
            data = {
                SMILES_COL: smiles,
                "fingerprint": list(custom_fp)

            }
            for col in ["original_index", "original_Smiles", "part", "Smiles"]:
                if not col in data and col in train_df_splitted.columns:
                    data[col] = test_df_splitted.loc[i, col]
            if "original_index" in test_df_splitted.columns:
                data["original_index"] = int(test_df_splitted.loc[i, "original_index"])
            
            test_fingerprints.append(data)
        
        with open(TEST_FPATH.as_posix(), 'w') as f:
            json.dump(test_fingerprints, f)

    train_fingerprints_df = pd.DataFrame(train_fingerprints)
    test_fingerprints_df = pd.DataFrame(test_fingerprints)
    train_df_ext = train_df_splitted.merge(
        train_fingerprints_df,
        on=["original_index", "original_Smiles", "part"], how="left")
    test_df_ext = test_df_splitted.merge(
        test_fingerprints_df,
        on=["original_index", "original_Smiles", "part"], how="left")
    print("Merge:", len(train_fingerprints_df), len(train_df_splitted), len(train_df_ext))

    print(train_df_ext.fingerprint.isnull().sum(), "train molecules have no associated fingerprint", train_df_ext.shape)
    print(test_df_ext.fingerprint.isnull().sum(), "test molecules have no associated fingerprint", test_df_ext.shape)

    train_df_ext = train_df_ext[~train_df_ext.fingerprint.isnull()]
    train_fingerprints = train_df_ext[FINGERPRINT_COL].apply(to_bits)  # lambda fingerprint_string: [x=='1' for x in fingerprint_string])
    train_fingerprints = np.stack(train_fingerprints.values)
    train_y = train_df_ext.Active.values
    validation_ids = train_df_val.loc[train_df_ext.index, "val_index"].values
    train_group = train_df_splitted["murcko_cleaned"].values

    # Next we filter train fingerprints
    # (to remove all 0 and duplicates from train dataset)
    index = dict()
    for i, row in enumerate(train_fingerprints):
        if np.sum(row) == 0:
            continue
        if tuple(row) in index:
            continue
        index[tuple(row)] = len(index)
    selected_ids = np.asarray(sorted(index.values()))

    train_fingerprints = train_fingerprints[selected_ids]
    train_y = train_y[selected_ids]
    train_group = train_group[selected_ids]
    validation_ids = validation_ids[selected_ids]
    print("Samples with no duplicates:", train_y.sum(), train_y.shape)

    test_df_ext = test_df_ext[~test_df_ext.fingerprint.isnull()]
    test_fingerprints = test_df_ext[FINGERPRINT_COL].apply(to_bits)
    test_fingerprints = np.stack(test_fingerprints.values)

    cv_y = []
    cv_predictions = []
    fold_thesholds = []
    for train_data, test_data in train_cv(
            CatboostClassifierWrapper, (train_fingerprints, train_y), n_splits=NFOLDS,
            random_state=SEED,
            save_prefix=MODEL_PREFIX,
            model_args=[],
            model_kwargs=dict(
                iterations=NITERATIONS,
                # learning_rate= 0.02,
                eval_metric="F1",
                metric_period=NITERATIONS//10,
                early_stopping_rounds=NITERATIONS//10*5,
                # auto_class_weights="Balanced",
                depth=2,
                use_best_model=True,
                # cat_features=np.arange(train_fingerprints.shape[1]),
                random_strength=1,
                random_seed=SEED,
                # rsm=1
            ),
            # fit_kwargs=dict(
            #     logging_level='Silent',
            #     plot=True
            # ),
            # model_random_state="random_seed",
            strategy="stratified+grouped",
            group_column=train_group,
            balance_train=True,
        ):
        # (train_index, xtrain, ytrain, ptrain) = train_data
        (test_index, xtest, ytest, ptest) = test_data
        if len(ptest.shape) == 2:
            ptest = ptest[..., 1]
        optimal_threshold, f1_ = get_threshold(ytest, ptest)
        print("Fold threshold:", optimal_threshold, "f1=", f1_)
        cv_y.append(ytest)
        cv_predictions.append(ptest > optimal_threshold)
        fold_thesholds.append(optimal_threshold)
    cv_y = np.concatenate(cv_y, axis=0)
    cv_predictions = np.concatenate(cv_predictions, axis=0)
    if len(cv_predictions.shape) == 2:
        cv_predictions = cv_predictions[:, 1]#np.argmax(cv_predictions, axis=1)
    if len(cv_predictions.shape) == 1:
        if set(cv_predictions) | {0, 1} == {0, 1}:
            print("Guessed correctly in train:", (cv_predictions == cv_y).mean())
            print("Guessed 1 correctly in train:", (cv_predictions[cv_y] == cv_y[cv_y]).sum())
            print("Total 1 in train:", cv_predictions.sum())
            print("F1 score in train:", f1_score(cv_y, cv_predictions))
        else:
            print("Train: float values")
            print("Guessed correctly in train:", ((cv_predictions > 0.5)*1 == cv_y).mean())
            print("Guessed 1 correctly in train:", ((cv_predictions[cv_y] > 0.5)*1 == cv_y[cv_y]).sum())
            print("Total 1 in train:", (cv_predictions > 0.5).sum())
            print("F1 score in train:", f1_score(cv_y, cv_predictions > 0.5))
            #print("F1", (cv_predictions == cv_y).mean())
        # these are 0 and 1
    print(cv_predictions.shape, train_y.shape)
    train_predictions = get_predictions(
        model_cls=CatboostClassifierWrapper, 
        test_data=train_fingerprints,
        n_splits=NFOLDS,
        save_prefix=MODEL_PREFIX,
        mode=["ones"]
    )
    for i, threshold in enumerate(fold_thesholds):
        train_predictions[i] = train_predictions[i] > threshold
    train_predictions = train_predictions.mean(0) > 0.5
    # print(train_predictions[:3], train_predictions.max(), train_predictions.mean())

    # optimal_threshold, f1_ = get_threshold(train_y, train_predictions)
    # train_predictions = train_predictions > optimal_threshold
    # f1_ = f1_score(train_y, train_predictions)

    print("----Predictions in test mode on train dataset----")
    # print("Optimal threshold is", optimal_threshold, "f1=", f1_)
    print("Total 1 in train (run in test mode)", train_predictions.sum())
    print("Guessed 1 correctly in train:", (train_predictions[train_y] == train_y[train_y]).sum())
    print("F1 score in train (run in test mode)", f1_score(train_y, train_predictions))
    score = f1_score(train_y[validation_ids], train_predictions[validation_ids])
    print("F1 score on selected val dataset", score)

    print("----Predictions in test mode on test dataset----")
    all_predictions = get_predictions(
        model_cls=CatboostClassifierWrapper,
        test_data=test_fingerprints, n_splits=NFOLDS,
        save_prefix=MODEL_PREFIX,
        mode=["ones"]
    )
    for i, threshold in enumerate(fold_thesholds):
        all_predictions[i] = all_predictions[i] > threshold
    all_predictions = all_predictions.mean(0) > 0.5
    # print(all_predictions[:10])
    print("Total 1 in test", all_predictions.sum(), all_predictions.shape)
    print(test_df_ext.columns)
    test_df_ext['Active'] = all_predictions
    test_df_ext = collect_df(test_df_ext, split_col="original_Smiles")
    # to ensure that the order in data is not modified
    print(test_df_ext.head())
    # preds_dict = {k:v for k, v in test_df_ext[["original_Smiles", "Active"]].values}
    #if "original_Smiles" in test_df.columns:
    #    test_df_ext.rename(columns={"original_Smiles": "Smiles"}, inplace=True)
    # test_df["Active"] = test_df.Smiles.apply(lambda x: preds_dict.get(x))
    test_df = test_df.merge(test_df_ext, left_index=True, right_index=True)
    assert test_df.Active.isnull().sum() == 0

    test_df.to_csv(TMP_DIR/"catboost_predictions_v2_cleaned_param.csv")


