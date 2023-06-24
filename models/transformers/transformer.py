from data_processing.window_splitters import WindowSplitterNoOverlap
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import librosa

class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformations: list = [np.mean], pca_components:int = 1, mfcc_components:int = 1):
        self.transformations = transformations
        self.pca_components = pca_components
        self.mfcc_components = mfcc_components
        if self.pca_components > 0:
            self.pca = PCA(n_components = self.pca_components)
        
        
    def fit(self, X, y = None):
        data = X.copy()
        if self.pca_components > 0:
            self.pca.fit(data)
        return self
    
    def transform(self, X, y = None):
        data = X.copy()

        dataPca = list()
        if self.pca_components > 0:
            dataPca = self.pca.transform(data)

        dataPca = np.array(dataPca)
        
        statistical = list()
        for transformation in self.transformations:
            statistical.append(transformation(data, axis = 1))

        statistical = np.array(statistical).T

        mfcc = np.array([])
        if self.mfcc_components > 0:
            mfcc = np.mean(librosa.feature.mfcc(y=data, sr = 0.2, n_mfcc=self.mfcc_components), axis = 2)


        result = None
        if len(dataPca) > 0:
            result = dataPca
        if len(statistical) > 0:
            if result is None:
                result = statistical
            else:
                result = np.concatenate((result, statistical), axis = 1)
        if len(mfcc) > 0:
            if result is None:
                result = mfcc
            else:
                result = np.concatenate((result, mfcc), axis = 1)
        return result