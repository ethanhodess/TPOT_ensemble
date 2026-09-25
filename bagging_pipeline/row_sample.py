from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import check_random_state
import numpy as np

class RowSampler(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator=None, random_state=None):
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, X, y):
        rng = check_random_state(self.random_state)
        n_samples = X.shape[0]
        row_idx = rng.choice(n_samples, size=n_samples, replace=True)

        X_sub = X.iloc[row_idx] if hasattr(X, "iloc") else X[row_idx]
        y_sub = y.iloc[row_idx] if hasattr(y, "iloc") else np.asarray(y)[row_idx]

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_sub, y_sub)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)