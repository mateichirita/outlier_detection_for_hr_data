from data_processing.window_splitters import WindowSplitterNoOverlap
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class WindowedImputer(BaseEstimator, TransformerMixin):
    def __init__(self, hrTransformations: list = None, stepTransformations: list = None, groupFunction = None):
        self.hrTransformations = hrTransformations
        self.stepTransformations = stepTransformations
        self.groupFunction = groupFunction
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        x = WindowSplitterNoOverlap(hrTransformations = self.hrTransformations, stepTransformations = self.stepTransformations).split(X, groupFunction=self.groupFunction)
        x = pd.concat(x.values())
        x.reset_index(drop=True, inplace=True)
        x_matrix = x.loc[:, x.columns != "label"].to_numpy()
        return x_matrix
    