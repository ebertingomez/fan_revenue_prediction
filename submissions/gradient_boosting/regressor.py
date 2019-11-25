import lightgbm as lgb
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = lgb.LGBMRegressor(
                    num_leaves=250,
                    learning_rate=0.05,
                    n_estimators=160,
                    num_iterations=220,
                    tree_learner = 'voting',
                    min_sum_hessian_in_leaf = 1e-10,
                    )

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
