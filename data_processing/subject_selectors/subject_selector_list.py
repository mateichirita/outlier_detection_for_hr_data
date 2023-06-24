from .subject_selector import SubjectSelector
import numpy as np
import pandas as pd
from pandas import DataFrame

class SubjectSelectorList(SubjectSelector):
    def __init__(self, subjectsPath: str, subjectsList: list[str], hr: bool, steps: bool) -> None:
        super().__init__(subjectsPath, hr, steps)
        self.subjectsList = subjectsList
    
    def normalize(self, data: DataFrame) -> DataFrame:
        for x in data:
            print(x)
            if x != "label":
                if data[x].std() == 0:
                    data[x] = 0
                else:
                    data[x] = (data[x] - data[x].mean()) / data[x].std()
        return data
    
    def get_subjects(self, start: int = None, finish: int = None) -> dict[str, DataFrame]:
        subjects: dict[str, DataFrame] = dict()

        for subject in self.subjectsList:
            data = self.parse(subject, start, finish)
            if data is None:
                raise Exception(f"Subject {subject} misses some data")

            subjects[subject] = data
        
        return subjects


