from __future__ import annotations
from .data_processor import DataProcessor
from pandas import DataFrame
import pandas as pd

class StepProcessor(DataProcessor):
    def __init__(self, data: DataFrame):
        data["time"] = pd.to_datetime(data["time"].dt.floor("min"))
        data = data.groupby("time").agg({"hr": "mean", "steps": "sum"})
        data.reset_index(inplace = True)
        super().__init__(data)
    
    def process(self) -> StepProcessor:
        super().process_common_features("steps")
        return self

    