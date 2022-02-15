from catboost import CatBoostClassifier
import numpy as np
from .base import BaseModelWrapper


class CatboostClassifierWrapper(BaseModelWrapper):
    def __init__(self, *args, **kwargs):
        self.model = CatBoostClassifier(*args, **kwargs)
    def predict(self, xtrain):
        return self.model.predict(xtrain)
    def predict_proba(self, xtrain):
        return self.model.predict_proba(xtrain)

    def fit(self, x_train, y_train, eval_set=(None, None)):
        self.model.fit(x_train, y_train,
            cat_features=np.arange(x_train.shape[1]),
            eval_set=eval_set)
    def save_fold(self, save_prefix, fold):
        self.model.save_model(f"{save_prefix}{fold}.cbm")
    @classmethod
    def load_fold(cls, save_prefix, fold):
        wrapper = cls()
        wrapper.model = CatBoostClassifier().load_model(f"{save_prefix}{fold}.cbm")
        return wrapper