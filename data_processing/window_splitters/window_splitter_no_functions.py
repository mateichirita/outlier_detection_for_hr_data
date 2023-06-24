from .window_splitter import WindowSplitter
from pandas import DataFrame
import pandas as pd
import numpy as np


class WindowSplitterNoFunctions(WindowSplitter):
    def __init__(self):
        super().__init__()

    def linear_interpolate(self, data: DataFrame) -> DataFrame:
        first = pd.to_datetime(np.array(data["time"])[0]).to_datetime64()
        last = pd.to_datetime(np.array(data["time"])[-1]).to_datetime64()
        timedf = DataFrame()
        timedf["time"] = np.arange(first, last, np.timedelta64(5, 's'))

        data.set_index("time", inplace=True)
        timedf.set_index("time", inplace=True)

        data = timedf.join(data, how="outer")
        data["time"] = data.index
        data.reset_index(drop = True, inplace=True)
        data["hr"].interpolate(method="linear", inplace=True)
        data["label"] = data["label"][0]
        return data

    def split(self, subjectsData: dict[str, DataFrame], groupFunction = None) -> dict[str, DataFrame]:
        windowedSubjectsData: dict[str, DataFrame] = dict()

        for subject, data in subjectsData.items():
            windowedData: DataFrame = DataFrame()
            data.reset_index(drop=True, inplace=True)
            group = None

            # data = self.linear_interpolate(data)
            # print(data.isna().sum())


            if groupFunction == None:
                groupFunction = lambda data: (np.repeat(np.arange(len(data) // 720 + 1), 720)[:len(data)], 720)
            
            groupArray = []
            samplesPerGroup = 0
            try:
                groupArray, samplesPerGroup = groupFunction(data)
            except Exception as e:
                print(e)
                print("1")
                print(data)
                continue

            if "hr" in data and "steps" in data:
                group = data[["time", "hr", "steps"]].groupby(groupArray)
            elif "hr" in data:
                group = data[["time", "hr"]].groupby(groupArray)
            elif "steps" in data:
                group = data[["time", "steps"]].groupby(groupArray)
            else:
                raise Exception("no hr data and no steps data")
            

            def aaa(x):
                # if len(x) == samplesPerGroup:
                #     print (len(x) == samplesPerGroup)
                return len(x) == samplesPerGroup
            
            # print(subject)
            # print(samplesPerGroup)
            group = group.filter(aaa)
            # print(group)

            try:
                group = group.groupby(groupFunction(group)[0])
            except Exception as e:
                print(e)
                print("2")
                print(data)
                print(subject)
                continue

            if "hr" in data:
                windowedData["hr"] = group["hr"].apply(list)
                # windowedData[[f"hr_{i}" for i in range(len(windowedData["hr"].iloc[0]))]]
                splitted = pd.DataFrame(windowedData["hr"].tolist(), index = windowedData.index)
                windowedData.drop(columns=["hr"], inplace=True)
                windowedData = pd.concat([windowedData, splitted], axis=1)
            
            if "steps" in data:
                windowedData["steps"] = group["steps"][~np.isnan(group["steps"])].apply(list)
            

            if "label" in data:
                windowedData["label"] = data["label"].iloc[0]
            
            windowedData.dropna(inplace=True)
            windowedSubjectsData[subject] = windowedData
        # print("done")
        
        return windowedSubjectsData