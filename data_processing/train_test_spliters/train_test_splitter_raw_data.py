from .train_test_splitter import TrainTestSplitter
import pandas as pd
import numpy as np 
from pandas import DataFrame

class TrainTestSplitterRawData(TrainTestSplitter):
    def __init__(self, numberAuthPersons: int, trainPer: float, auth: list[str] = None, nonAuth: list[str] = None) -> None:
        super().__init__()
        self.numberAuthPersons = numberAuthPersons
        self.auth = auth
        self.nonAuth = nonAuth
        self.trainPer = trainPer
        self.authIds = None
        self.unauthIds = None

    def split(self, windowedData: dict[str, DataFrame]) -> tuple[DataFrame, DataFrame]:
        pass

    def normalize(self, data: DataFrame) -> DataFrame:
        for x in data:
            if x != "label" and x != "time":
                std = np.nanstd(data[x])
                if std == 0:
                    data[x] = 0
                else:
                    data[x] = (data[x] - np.nanmean(data[x])) / std
        return data

    def getAuthAndUnauthIds(self) -> tuple[list[str], list[str]]:
        return self.authIds, self.unauthIds

    def splitBeforeWindowing(self, data: dict[str, DataFrame]) -> tuple[dict[str, DataFrame], dict[str, DataFrame]]:
        if len(data) < self.numberAuthPersons:
            raise Exception("too many accepted persons")
        
        keys = list(data.keys())
        authIds: list[str] = None
        unauthIds: list[str] = None

        if self.auth is None:
            authIds = np.random.choice(keys, self.numberAuthPersons, replace=False)
            unauthIds = list(set(keys) - set(authIds))
        elif set(self.auth).issubset(keys) and set(self.nonAuth).issubset(keys):
            authIds = self.auth
            unauthIds = self.nonAuth
        else:
            raise Exception("auth not in data / unauth not in data")

        self.authIds = authIds
        self.unauthIds = unauthIds

        train: dict[str, DataFrame] = dict()
        test: dict[str, DataFrame] = dict()

        testTrueLen = 0
        testFalseLen = 0

        for id in authIds:
            train[id] = data[id][:int(len(data[id]) * self.trainPer)].copy()
            # train[id] = self.normalize(train[id])
            train[id]["label"] = 1
            test[id] = data[id][int(len(data[id]) * self.trainPer):].copy()
            # test[id] = self.normalize(test[id])
            test[id]["label"] = 1

            testTrueLen += len(test[id])

        for id in unauthIds:
            test[id] = data[id].copy()
            # test[id] = self.normalize(test[id])
            test[id]["label"] = -1
            
            testFalseLen += len(test[id])
        

        minim = min(testTrueLen, testFalseLen)
        toReduceTrue = int((testTrueLen - minim) / len(authIds))
        toReduceFalse = int((testFalseLen - minim) / len(unauthIds))

        # print(toReduceTrue)
        # print(toReduceFalse)
        
        if toReduceTrue > 0:
            for id in authIds:
                frac = len(test[id]) / testTrueLen      
                test[id] = test[id][: int(- toReduceTrue * frac)]
        
        if toReduceFalse > 0:
            for id in unauthIds:   
                frac = len(test[id]) / testFalseLen     
                test[id] = test[id][: int(- toReduceFalse * frac)]
        

        return train, test
    

    def splitBeforeWindowing2(self, data: dict[str, DataFrame]) -> tuple[dict[str, DataFrame], dict[str, DataFrame]]:
        if len(data) < self.numberAuthPersons:
            raise Exception("too many accepted persons")
        
        keys = list(data.keys())
        authIds: list[str] = None
        unauthIds: list[str] = None

        if self.auth is None:
            authIds = np.random.choice(keys, self.numberAuthPersons, replace=False)
            unauthIds = list(set(keys) - set(authIds))
        elif set(self.auth).issubset(keys) and set(self.nonAuth).issubset(keys):
            authIds = self.auth
            unauthIds = self.nonAuth
        else:
            
            raise Exception("auth not in data / unauth not in data")

        self.authIds = authIds
        self.unauthIds = unauthIds

        train: dict[str, DataFrame] = dict()
        test: dict[str, DataFrame] = dict()

        testTrueLen = 0
        testFalseLen = 0

        for id in authIds:
            train[id] = data[id][:int(len(data[id]) * self.trainPer)].copy()
            # train[id] = self.normalize(train[id])
            train[id]["label"] = 1
            test[id] = data[id][int(len(data[id]) * self.trainPer):].copy()
            # test[id] = self.normalize(test[id])
            test[id]["label"] = 1

            testTrueLen += len(test[id])

        for id in unauthIds:
            train[id] = data[id][:int(len(data[id]) * self.trainPer)].copy()
            # train[id] = self.normalize(train[id])
            train[id]["label"] = -1
            test[id] = data[id][int(len(data[id]) * self.trainPer):].copy()
            # test[id] = self.normalize(test[id])
            test[id]["label"] = -1
            
            testFalseLen += len(test[id])
        

        minim = min(testTrueLen, testFalseLen)
        toReduceTrue = int((testTrueLen - minim) / len(authIds))
        toReduceFalse = int((testFalseLen - minim) / len(unauthIds))

        # print(toReduceTrue)
        # print(toReduceFalse)
        
        if toReduceTrue > 0:
            for id in authIds:
                frac = len(test[id]) / testTrueLen      
                test[id] = test[id][: int(- toReduceTrue * frac)]
        
        if toReduceFalse > 0:
            for id in unauthIds:   
                frac = len(test[id]) / testFalseLen     
                test[id] = test[id][: int(- toReduceFalse * frac)]
        

        return train, test
        
