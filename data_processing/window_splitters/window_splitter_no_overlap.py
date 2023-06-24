from .window_splitter import WindowSplitter
from pandas import DataFrame
import pandas as pd
import numpy as np

class WindowSplitterNoOverlap(WindowSplitter):
    def __init__(self, hrTransformations: list = None, stepTransformations: list = None):
        super().__init__()
        self.hrTransformations = hrTransformations
        self.stepTransformations = stepTransformations

    def split(self, subjectsData: dict[str, DataFrame], groupFunction = None) -> dict[str, DataFrame]:
        # print("windowing subjects data")
        windowedSubjectsData: dict[str, DataFrame] = dict()
        for subject, data in subjectsData.items():
            windowedData: DataFrame = DataFrame()
            data.reset_index(drop=True, inplace=True)
            group = None

            if groupFunction == None:
                groupFunction = lambda data: np.repeat(np.arange(len(data) // 720 + 1), 720)[:len(data)]

            # print(data)
            # first = pd.to_datetime(np.array(data["time"])[0])
            groupArray = groupFunction(data)
            if "hr" in data and "steps" in data:
                group = data[["time", "hr", "steps"]].groupby(groupArray)
            elif "hr" in data and (self.stepTransformations == None or len(self.stepTransformations) == 0):
                # print(data)
                group = data[["time", "hr"]].groupby(groupArray)
            elif "steps" in data and (self.hrTransformations == None or len(self.hrTransformations)) == 0:
                group = data[["time", "steps"]].groupby(groupArray)
            else:
                raise Exception("no hr data with hr transformations or no steps data with steps transformations")
            
            # print(group.mean())
            # def aaa(x):
            #     # print(len(x))
            #     return len(x) == 720

            # group = group.filter(aaa)
            # print(group)
            # group = group.groupby(groupFunction(group))
            # print(group.mean())

            # print("grouping done")
            if self.hrTransformations:
                for f in self.hrTransformations:
                    # print(group.filter(lambda x: len(x) == 720).groupby(groupFunction(data)))
                    windowedData = pd.concat([windowedData, group["hr"].apply(f)], axis=1)
                    windowedData.rename(columns={"hr": "hr_" + f.__name__}, inplace=True)
            
            if self.stepTransformations:
                for f in self.stepTransformations:
                    windowedData = pd.concat([windowedData, group["steps"].apply(f)], axis=1)
                    windowedData.rename(columns={"steps": "steps_" + f.__name__}, inplace=True)
            
            # windowedData = windowedData[~windowedData.isnull().any(axis=1)]
            for x in windowedData.columns:
                # print(x)
                # print(windowedData[x].isna().sum())
                windowedData = windowedData[windowedData[x].notna()]
            
            if "label" in data:
                windowedData["label"] = data["label"].iloc[0]
            
            windowedSubjectsData[subject] = windowedData
        # print("done")
        
        return windowedSubjectsData