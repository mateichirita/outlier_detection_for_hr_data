import pandas as pd
from pandas import DataFrame
from abc import ABC, abstractclassmethod

class WindowSplitter(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractclassmethod
    def split(self, subjectsData: dict[str, DataFrame], groupFunction = None) -> dict[str, DataFrame]:
        pass

