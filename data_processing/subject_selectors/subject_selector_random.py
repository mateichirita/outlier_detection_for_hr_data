from .subject_selector import SubjectSelector
import numpy as np
import pandas as pd
from pandas import DataFrame

class SubjectSelectorRandom(SubjectSelector):
    def __init__(self, subjectsPath: str, nSubjects: int, hr: bool, steps: bool) -> None:
        super().__init__(subjectsPath, hr, steps)
        self.nSubjects = nSubjects

    def get_subjects(self, start: int = None, finish: int = None) -> dict[str, DataFrame]:
        subjects: dict[str, DataFrame] = dict()
        selected = set()

        while len(selected) < self.nSubjects:
            subject: str = np.random.choice(self.allSubjects)
            if subject in selected:
                continue
            
            data = self.parse(subject, start, finish)
            if data is None:
                continue

            selected.add(subject)
            subjects[subject] = data
        
        return subjects



