import pandas as pd
from pandas import DataFrame
from os import listdir
from os.path import isfile, join
import numpy as np

def me_time_parser(path: str) -> dict[int, DataFrame]:
    subjectsPath: str = join(path,"subjects")

    def normalize(data: DataFrame) -> DataFrame:
        data["hr"] = (data["hr"] - data["hr"].mean()) / data["hr"].std()
        data["steps"] = (data["steps"] - data["steps"].mean()) / data["steps"].std()


    data: dict[int, DataFrame] = dict()
    files = [f for f in listdir(subjectsPath) if isfile(join(subjectsPath, f))]
    print(files)

    for f in files:
        subjectPath: str = join(subjectsPath, f)
        subjectData: DataFrame = pd.read_pickle(subjectPath)

        normalize(subjectData)

        data[int(f[:-4])] = subjectData
        
    
    return data

def me_time_fused_parser(path: str) -> dict[str, DataFrame]:
    subjectsPath: str = join(path,"subjects")

    def normalize(data: DataFrame) -> DataFrame:
        data["hr"] = (data["hr"] - np.nanmean(data["hr"])) / np.nanstd(data["hr"])
        data["steps"] = np.asarray(data["steps"])
        data["steps"] = (data["steps"] - np.nanmean(data["steps"])) / np.nanstd(data["steps"])

    data: dict[str, DataFrame] = dict()
    files = [f for f in listdir(subjectsPath) if isfile(join(subjectsPath, f))]

    for f in files:
        print(f)
        subjectPath: str = join(subjectsPath, f)
        subjectData: DataFrame = pd.read_pickle(subjectPath)

        if len(subjectData[(subjectData["steps"] > 0)]) == 0:
            print(f"No steps data for subject: {f}")
            continue
        
        if len(subjectData[(subjectData["hr"] > 0)]) == 0:
            print(f"No hr data for subject: {f}")
            continue

        normalize(subjectData)

        # print(subjectData[(subjectData["steps"] > 0)])

        data[f[:-4]] = subjectData
        # break
        
    return data

def me_time_fused_parser_only_hr(path: str) -> dict[str, DataFrame]:
    subjectsPath: str = join(path,"subjects")

    def normalize(data: DataFrame) -> DataFrame:
        data["hr"] = (data["hr"] - np.nanmean(data["hr"])) / np.nanstd(data["hr"])

    data: dict[str, DataFrame] = dict()
    files = [f for f in listdir(subjectsPath) if isfile(join(subjectsPath, f))]

    for f in files:
        print(f)
        subjectPath: str = join(subjectsPath, f)
        subjectData: DataFrame = pd.read_pickle(subjectPath)

        if len(subjectData[(subjectData["hr"] > 0)]) == 0:
            print(f"No data for subject: {f}")
            continue

        normalize(subjectData)

        data[f[:-4]] = subjectData[["time", "hr"]]
        
    return data


def me_time_fused_parser_only_steps(path: str) -> dict[str, DataFrame]:
    subjectsPath: str = join(path,"subjects")

    def normalize(data: DataFrame) -> DataFrame:
        data["steps"] = np.asarray(data["steps"])
        data["steps"] = (data["steps"] - np.nanmean(data["steps"])) / np.nanstd(data["steps"])

    data: dict[str, DataFrame] = dict()
    files = [f for f in listdir(subjectsPath) if isfile(join(subjectsPath, f))]

    for f in files:
        print(f)
        subjectPath: str = join(subjectsPath, f)
        subjectData: DataFrame = pd.read_pickle(subjectPath)

        if len(subjectData[subjectData["steps"] > 0]) == 0:
            print(f"No data for subject: {f}")
            continue

        normalize(subjectData)

        data[f[:-4]] = subjectData[["time", "steps"]]
        
    return data

