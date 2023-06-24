from .train_test_splitter import TrainTestSplitter
from data_processing.window_splitters import WindowSplitter
import pandas as pd
from pandas import DataFrame
import numpy as np 
from sklearn.model_selection import train_test_split

class TrainTestSplitterOutlierPersons(TrainTestSplitter):
    def __init__(self, numberAuthPersons: int, trainPer: float) -> None:
        super().__init__()
        self.numberAuthPersons = numberAuthPersons
        self.trainPer = trainPer
        self.authIds = None
        self.unauthIds = None

    def normalize(self, data: DataFrame) -> DataFrame:
        for x in data:
            if x != "label":
                if data[x].std() == 0:
                    data[x] = 0
                else:
                    data[x] = (data[x] - data[x].mean()) / data[x].std()
        return data

    def getAuthAndUnauthIds(self) -> tuple[list[str], list[str]]:
        return self.authIds, self.unauthIds
    
    def split(self, windowedData: dict[str, DataFrame]) -> tuple[DataFrame, DataFrame]:
        if len(windowedData) < self.numberAuthPersons:
            raise Exception("too many accepted persons")
        
        keys = list(windowedData.keys())
        authIds = np.random.choice(keys, self.numberAuthPersons, replace=False)
        unauthIds = list(set(keys) - set(authIds))

        self.authIds = authIds
        self.unauthIds = unauthIds

        authList = []
        unauthList = []

        for id in authIds:
            windowedData[id]["label"] = 1
            authList.append(windowedData[id])
        
        for id in unauthIds:
            windowedData[id]["label"] = 0
            unauthList.append(windowedData[id])

        authDataSet: DataFrame = pd.concat(authList)
        unauthDataSet: DataFrame = pd.concat(unauthList)

        train, test = train_test_split(authDataSet, train_size=self.trainPer, shuffle=True)
        test = pd.concat([test, unauthDataSet])
        test = test.sample(frac=1)

        train = self.normalize(train)
        test  = self.normalize(test)
        
        
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        return train, test