import pandas as pd
import numpy as np 
from pandas import DataFrame


class TrainTestSplitterWindows:
    def __init__(self, data: dict[str, DataFrame]) -> None:
        self.data = data
    
    def equalizeTest(self, test: dict[str, DataFrame], testTrueLen: int, testFalseLen: int, auth, unauth) -> dict[str, DataFrame]:
        minim = min(testTrueLen, testFalseLen)
        toReduceTrue = int(testTrueLen - minim)
        toReduceFalse = int(testFalseLen - minim)

        # print()
        # print(toReduceTrue)
        # print(toReduceFalse)

        if toReduceTrue > 0:
            for id in auth:
                frac = len(test[id]) / testTrueLen      
                test[id] = test[id][: int(- toReduceTrue * frac)]
        
        if toReduceFalse > 0:
            for id in unauth:   
                frac = len(test[id]) / testFalseLen     
                test[id] = test[id][: int(- toReduceFalse * frac)]

        return test

    def split(self, auth: list, unauth: list, trainPer: int) -> tuple[DataFrame, DataFrame]:
        keys = list(self.data.keys())
        if not (set(auth).issubset(keys) and set(unauth).issubset(keys)):
            raise Exception("auth not in data / unauth not in data")
        
        train: dict[str, DataFrame] = dict()
        test: dict[str, DataFrame] = dict()

        testTrueLen = 0
        testFalseLen = 0

        for id in auth:
            train[id] = self.data[id][ : int(len(self.data[id]) * trainPer)].copy()
            test[id] = self.data[id][int(len(self.data[id]) * trainPer) : ].copy()

            train[id]["label"] = 1
            test[id]["label"] = 1

            testTrueLen += len(test[id])


        for id in unauth:
            test[id] = self.data[id].copy()
            test[id]["label"] = -1

            testFalseLen += len(test[id])

        # print(testTrueLen)
        # print(testFalseLen)

        test = self.equalizeTest(test, testTrueLen, testFalseLen, auth, unauth)

        return train, test