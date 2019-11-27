import lightgbm as lgb
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = lgb.LGBMRegressor(
                    n_jobs=3,
                    num_threads=3,
                    num_leaves=250,
                    learning_rate=0.05,
                    n_estimators=160,
                    )

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
