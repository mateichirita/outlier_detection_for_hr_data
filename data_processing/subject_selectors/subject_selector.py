from abc import ABC, abstractclassmethod
from os import listdir
from os.path import isfile, join
from pandas import DataFrame
import pandas as pd
import numpy as np

class SubjectSelector(ABC):
    def __init__(self, subjectsPath: str, hr: bool, steps: bool) -> None:
        if not hr and not steps:
            raise Exception("At least one of hr or steps must be true")
         
        self.subjectsPath = subjectsPath
        self.allSubjects = [f[:-4] for f in listdir(subjectsPath) if isfile(join(subjectsPath, f))]
        self.hr = hr
        self.steps = steps

    @abstractclassmethod
    def get_subjects(self, start: int = None, finish: int = None) -> list[str]:
        pass
     

    def parse(self, subject: str, start: int, finish: int) -> DataFrame:
        subject = subject + ".pkl"
        subjectPath: str = join(self.subjectsPath, subject)
        subjectData: DataFrame = pd.read_pickle(subjectPath)

        if self.hr and len(subjectData[(subjectData["hr"] > 0)]) == 0:
            print(f"No hr data for subject: {subject}")
            return None
        
        if self.steps and len(subjectData[(subjectData["steps"] > 0)]) == 0:
            print(f"No steps data for subject: {subject}")
            return None
        
        parsedData: DataFrame = DataFrame()
        parsedData["time"] = subjectData["time"]
        if self.hr:
            parsedData["hr"] = subjectData["hr"]
        if self.steps:
            parsedData["steps"] = np.asarray(subjectData["steps"])
        
        if start is not None and finish is not None:
            parsedData = parsedData[(parsedData["time"].dt.hour.between(start, finish - 1) | ((parsedData["time"].dt.hour == finish) & (parsedData["time"].dt.minute == 0) & (parsedData["time"].dt.second == 0)))]
            
        return parsedData