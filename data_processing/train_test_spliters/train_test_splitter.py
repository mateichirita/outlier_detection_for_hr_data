import pandas as pd
from pandas import DataFrame
from abc import ABC, abstractclassmethod
from data_processing.window_splitters import WindowSplitter

class TrainTestSplitter(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractclassmethod
    def split(self, windowedData: dict[str, DataFrame]) -> tuple[DataFrame, DataFrame]:
        pass

    def splitBeforeWindowing(self, data: dict[str, DataFrame]) -> tuple[dict[str, DataFrame], dict[str, DataFrame]]:
        pass
