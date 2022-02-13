"""
"""
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

from common.utils import seed_everything, train_cv, eval_cv
from common.pubchem import get_compounds_fingerprints, to_bits
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
    SEED = 3407
    NFOLDS = 5
    NITERATIONS = 1000
    TMP_DIR = Path("../tmp/")
    SAVE_DIR = Path("../weights/")
    SAVE_DIR.mkdir(exist_ok=True)
    MODEL_PREFIX = (SAVE_DIR/ "cbt_fold_").as_posix()
    seed_everything(SEED)
    print("t")
    train_df = pd.read_csv(Path("../data/train.csv"), index_col=0)
    test_df = pd.read_csv(Path("../data/test.csv"), index_col=0)
    # print(train_df)
    train_fingerprints = get_compounds_fingerprints(
        train_df, cache_dir=str(TMP_DIR / "train"))
    test_fingerprints = get_compounds_fingerprints(
        test_df, cache_dir=str(TMP_DIR/ "test"))
    train_fingerprints_df = pd.DataFrame(train_fingerprints)
    test_fingerprints_df = pd.DataFrame(test_fingerprints)
    train_df_ext = train_df.merge(train_fingerprints_df, on="Smiles", how="left")
    test_df_ext = test_df.merge(test_fingerprints_df, on="Smiles", how="left")

    print(train_df_ext.fingerprint.isnull().sum(), "train molecules have no associated fingerprint")
    print(test_df_ext.fingerprint.isnull().sum(), "test molecules have no associated fingerprint")

    train_df_ext = train_df_ext[~train_df_ext.fingerprint.isnull()]
    train_fingerprints = train_df_ext.fingerprint.apply(to_bits)#lambda fingerprint_string: [x=='1' for x in fingerprint_string])
    train_fingerprints = np.stack(train_fingerprints.values)
    train_y = train_df_ext.Active.values

    test_df_ext = test_df_ext[~test_df_ext.fingerprint.isnull()]
    test_fingerprints = test_df_ext.fingerprint.apply(to_bits)
    test_fingerprints = np.stack(test_fingerprints.values)

    cv_y = []
    cv_predictions = []
    for train_data, test_data in train_cv(
            CatboostClassifierWrapper, (train_fingerprints, train_y), n_splits=NFOLDS,
            random_state=SEED, save_prefix=MODEL_PREFIX,
            model_args=[],
            model_kwargs=dict(
                iterations=NITERATIONS,
                # "learning_rate": 0.1,
                eval_metric="F1",
                metric_period=NITERATIONS//10,
                # early_stopping_rounds=NITERATIONS//10*5,
                auto_class_weights="Balanced",
                depth=5,
                use_best_model=False,
            ),
            model_random_state="random_state"
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
    print("F1 score in train (run in test mode)", f1_score(train_y, train_predictions))

    print("----Predictions in test mode on test dataset----")
    all_predictions = get_predictions(
        model_cls=CatboostClassifierWrapper, test_data=test_fingerprints, n_splits=NFOLDS,
        save_prefix=MODEL_PREFIX
    )
    # print(all_predictions[:10])
    print("Total 1 in test", all_predictions.sum())
    test_df_ext['Active'] = all_predictions
    test_df_ext[["Smiles", "Active"]].to_csv(TMP_DIR/"catboost_predictions_v0.csv")


