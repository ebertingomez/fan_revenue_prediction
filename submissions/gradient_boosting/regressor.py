import lightgbm as lgb
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = lgb.LGBMRegressor(
                    num_threads=5,
                    num_leaves=500,
                    n_estimators=45,
                    learning_rate=0.02,
                    max_depth=8,
                    reg_alpha=0.8,
                    min_data_in_leaf=1,
                    num_iterations=3000,
    )

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
