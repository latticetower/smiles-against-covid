"""
"""
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
import json

from common.utils import seed_everything, train_cv, eval_cv
from common.rdkit_utils import smiles2canonical, smiles2cleaned
from common.pubchem import get_compounds_fingerprints, to_bits
from common.rdkit_utils import get_murcko_scaffold
from models import CatboostClassifierWrapper


def get_predictions(
            model_cls=CatboostClassifierWrapper, test_data=None, n_splits=3,
            save_prefix="model_save"
        ):
    all_predictions = []
    for predictions in eval_cv(
            model_cls,
            test_data, n_splits=n_splits,
            save_prefix=save_prefix):
        all_predictions.append(predictions)
    all_predictions = np.stack(all_predictions)
    all_predictions = all_predictions.mean(0)
    if all_predictions.shape[1] == 2:
        all_predictions = np.argmax(all_predictions, axis=1)
    return all_predictions


if __name__ == '__main__':
    SEED = 2407
    NFOLDS = 7
    NITERATIONS = 1000
    TMP_DIR = Path("../tmp/")
    SAVE_DIR = Path("../weights/")
    SAVE_DIR.mkdir(exist_ok=True)
    MODEL_PREFIX = (SAVE_DIR/ "cbt_fold_").as_posix()
    seed_everything(SEED)
    print("t")
    train_df = pd.read_csv(Path("../data/train.csv"), index_col=0)
    test_df = pd.read_csv(Path("../data/test.csv"), index_col=0)
    # clean up everything - convert to canonical smiles
    train_df['canonical'] = train_df.Smiles.apply(smiles2canonical)
    test_df['canonical'] = test_df.Smiles.apply(smiles2canonical)

    train_df['cleaned'] = train_df.Smiles.apply(smiles2cleaned)
    test_df['cleaned'] = test_df.Smiles.apply(smiles2cleaned)

    train_df["murcko"] = train_df.canonical.apply(get_murcko_scaffold)
    # test_df["murcko"] = test_df.canonical.apply(get_murcko_scaffold)

    SMILES_COL = "canonical"
    # print(train_df)
    FINGERPRINT_COL="cactvs"
    RECOMPUTE = True

    TRAIN_FPATH = TMP_DIR/"train_fingerprints.json"
    if TRAIN_FPATH.exists() and not RECOMPUTE:
        with open(TRAIN_FPATH.as_posix()) as f:
            train_fingerprints = json.load(f)
    else:
        train_fingerprints = get_compounds_fingerprints(
            train_df, cache_dir=str(TMP_DIR / "train"),
            smiles_column=SMILES_COL,
            additional_cols=["Smiles"]
        )
        with open(TRAIN_FPATH.as_posix(), 'w') as f:
            json.dump(train_fingerprints, f)
    TEST_FPATH = TMP_DIR/"test_fingerprints.json"
    if TEST_FPATH.exists() and not RECOMPUTE:
        with open(TEST_FPATH.as_posix()) as f:
            test_fingerprints = json.load(f)
    else:
        test_fingerprints = get_compounds_fingerprints(
            test_df, cache_dir=str(TMP_DIR/ "test"),
            smiles_column=SMILES_COL,
            additional_cols=["Smiles"]
        )
        with open(TEST_FPATH.as_posix(), 'w') as f:
            json.dump(test_fingerprints, f)

    train_fingerprints_df = pd.DataFrame(train_fingerprints)
    test_fingerprints_df = pd.DataFrame(test_fingerprints)
    train_df_ext = train_df.merge(train_fingerprints_df, on="Smiles", how="left")
    test_df_ext = test_df.merge(test_fingerprints_df, on="Smiles", how="left")

    print(train_df_ext.fingerprint.isnull().sum(), "train molecules have no associated fingerprint")
    print(test_df_ext.fingerprint.isnull().sum(), "test molecules have no associated fingerprint")

    train_df_ext = train_df_ext[~train_df_ext.fingerprint.isnull()]
    train_fingerprints = train_df_ext[FINGERPRINT_COL].apply(to_bits)  # lambda fingerprint_string: [x=='1' for x in fingerprint_string])
    train_fingerprints = np.stack(train_fingerprints.values)
    train_y = train_df_ext.Active.values

    test_df_ext = test_df_ext[~test_df_ext.fingerprint.isnull()]
    test_fingerprints = test_df_ext[FINGERPRINT_COL].apply(to_bits)
    test_fingerprints = np.stack(test_fingerprints.values)

    cv_y = []
    cv_predictions = []
    for train_data, test_data in train_cv(
            CatboostClassifierWrapper, (train_fingerprints, train_y), n_splits=NFOLDS,
            random_state=SEED, save_prefix=MODEL_PREFIX,
            model_args=[],
            model_kwargs=dict(
                iterations=NITERATIONS,
                # learning_rate= 0.02,
                eval_metric="F1",
                metric_period=NITERATIONS//10,
                # early_stopping_rounds=NITERATIONS//10*5,
                auto_class_weights="Balanced",
                depth=5,
                use_best_model=False,
                cat_features=np.arange(train_fingerprints.shape[1]),
            ),
            model_random_state="random_state",
            strategy="stratified+grouped",
            group_column=train_df["murcko"].values,
            # balance_train=True,
        ):
        # (train_index, xtrain, ytrain, ptrain) = train_data
        (test_index, xtest, ytest, ptest) = test_data
        cv_y.append(ytest)
        cv_predictions.append(ptest)
    cv_y = np.concatenate(cv_y, axis=0)
    cv_predictions = np.concatenate(cv_predictions, axis=0)
    if len(cv_predictions.shape)==2:
        cv_predictions = np.argmax(cv_predictions, axis=1)
    if len(cv_predictions.shape) == 1:
        if set(cv_predictions) | {0, 1} == {0, 1}:
            print("Guessed correctly in train:", (cv_predictions == cv_y).mean())
            print("Guessed 1 correctly in train:", (cv_predictions[cv_y] == cv_y[cv_y]).sum())
            print("Total 1 in train:", cv_predictions.sum())
            print("F1 score in train:", f1_score(cv_y, cv_predictions))
        else:
            print("Train: float values")
            #print("F1", (cv_predictions == cv_y).mean())
        # these are 0 and 1
    print(cv_predictions.shape, cv_y.shape)
    train_predictions = all_predictions = get_predictions(
        model_cls=CatboostClassifierWrapper, 
        test_data=train_fingerprints,
        n_splits=NFOLDS,
        save_prefix=MODEL_PREFIX
    )
    print("----Predictions in test mode on train dataset----")
    print("Total 1 in train (run in test mode)", train_predictions.sum())
    print("Guessed 1 correctly in train:", (train_predictions[train_y] == train_y[train_y]).sum())
    print("F1 score in train (run in test mode)", f1_score(train_y, train_predictions))

    print("----Predictions in test mode on test dataset----")
    all_predictions = get_predictions(
        model_cls=CatboostClassifierWrapper, test_data=test_fingerprints, n_splits=NFOLDS,
        save_prefix=MODEL_PREFIX
    )
    # print(all_predictions[:10])
    print("Total 1 in test", all_predictions.sum())
    test_df_ext['Active'] = all_predictions
    test_df_ext[["Smiles", "Active"]].to_csv(TMP_DIR/"catboost_predictions_v1.csv")


