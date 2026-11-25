pip install flask scikit-learn pandas numpy matplotlib seaborn joblib

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PhysicsFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = X.copy()
        d['Temp_Diff [K]'] = d['Process temperature [K]'] - d['Air temperature [K]']
        d['Power [W]'] = d['Torque [Nm]'] * (d['Rotational speed [rpm]'] * 2 * np.pi / 60)
        d['Overstrain [min*Nm]'] = d['Tool wear [min]'] * d['Torque [Nm]']
        return d
