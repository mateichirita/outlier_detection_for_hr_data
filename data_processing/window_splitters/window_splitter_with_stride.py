from pandas import DataFrame
import pandas as pd
import numpy as np


class WindowSplitterWithStride:
    def __init__(self) -> None:
        pass

    def groupSplitter(self, data: DataFrame, window: int, stride: int) -> tuple[list[np.array], int]:
        if window % stride != 0:
            raise Exception("window not divisible by stride")

        first = pd.to_datetime(np.array(data["time"])[0])
        first = first.replace(hour = 0, minute = 0, second = 0, microsecond = 0)

        currentOffset = 0
        groups: list[np.array] = list()

        windowInHours = window // 60
        datapointsPerGroup = window * 60 // 5

        while currentOffset < window:
            offsetTime = data["time"] - pd.Timedelta(minutes = currentOffset)
            groups.append(np.array(offsetTime.dt.hour // windowInHours + (offsetTime - first).dt.days * (24 / windowInHours)).astype(int))
            currentOffset += stride
        return groups, datapointsPerGroup

    def removeSmallWindows(self, groupArray: np.array, samplesPerGroup: int) -> np.array:
        unique, counts = np.unique(groupArray, return_counts=True)
        freq = np.asarray((unique, counts)).T
        elemLess = freq[~(freq == samplesPerGroup)[:, 1]][:, 0]
        filter = np.in1d(groupArray, elemLess)
        groupArray[filter] = -1
        return groupArray

    def split(self, subjectsData: tuple[str, DataFrame], window, stride) -> dict[str, DataFrame]:
        subject = subjectsData[0]
        data = subjectsData[1]
        data.reset_index(drop=True, inplace=True)

        groupArrays, samplesPerGroup = self.groupSplitter(data, window, stride)
        
        windowedDataFinal: DataFrame = DataFrame()

        for groupArray in groupArrays:
            windowedData: DataFrame = DataFrame()
            groupArray = self.removeSmallWindows(groupArray, samplesPerGroup)

            group = data.groupby(groupArray)

            windowedData["hr"] = group["hr"].apply(list)
            windowedData["keys"] = group.groups.keys()
            windowedData = windowedData[windowedData["keys"] != -1]
            windowedData = windowedData.drop(columns=["keys"])


            splitted = pd.DataFrame(windowedData["hr"].tolist(), index = windowedData.index)
            windowedData.drop(columns=["hr"], inplace=True)
            windowedData = pd.concat([windowedData, splitted], axis=1)

            windowedData["time"] = group["time"].min()

            windowedDataFinal = pd.concat([windowedDataFinal, windowedData], axis=0)
        
        windowedDataFinal.sort_values(by=["time"], inplace=True)
        # windowedDataFinal.drop(columns=["time"], inplace=True)
        windowedDataFinal.reset_index(drop=True, inplace=True)
        return windowedDataFinal

        



