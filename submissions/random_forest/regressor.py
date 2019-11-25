from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = RandomForestRegressor(
                        n_estimators=5,
                        max_depth=35,
                        max_features=7,
                        min_impurity_decrease=0.018,
                        min_impurity_split=0.0063,
                        min_weight_fraction_leaf=1e-08
                    )

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
