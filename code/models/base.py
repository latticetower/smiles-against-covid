class BaseModelWrapper:
    def __init__(self):
        pass
    def fit(self, x_train, y_train, eval_set=(None, None), **kwargs):
        pass
    def predict(self, x_test):
        pass
    def save_fold(self, save_prefix, fold):
        pass