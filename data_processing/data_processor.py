from __future__ import annotations
from abc import ABC, abstractclassmethod
import pandas as pd
from pandas import DataFrame

class DataProcessor(ABC):
    window = "H"
    @abstractclassmethod
    def __init__(self, data) -> None:
        self.data: DataFrame = data
        self.processed: DataFrame = DataFrame()

        self.data["window"] = pd.to_datetime(self.data["time"].dt.floor(DataProcessor.window))
        self.processed["window"] = self.data["window"].unique()
        self.processed = self.processed.set_index("window")

    def get_data(self) -> DataFrame:
        return self.data

    def process_common_features(self, column: str) -> DataProcessor:
        grouped = self.data.groupby("window")
        
        self.processed[column + "_mean"] = grouped[column].mean().values
        self.processed[column + "_median"] = grouped[column].median().values
        self.processed[column + "_std"] = grouped[column].std().values
        self.processed[column + "_variance"] = grouped[column].var().values
        self.processed[column + "_cv"] = self.processed[column + "_std"] / self.processed[column + "_mean"]
        self.processed[column + "_min"] = grouped[column].min().values
        self.processed[column + "_max"] = grouped[column].max().values
        self.processed[column + "_range"] = self.processed[column + "_max"] - self.processed[column + "_min"]
        self.processed[column + "_cr"] = self.processed[column + "_range"] / (self.processed[column + "_max"] + self.processed[column + "_min"])
        self.processed[column + "_0.25q"] = grouped[column].quantile([0.25]).values
        self.processed[column + "_0.75q"] = grouped[column].quantile([0.75]).values
        self.processed[column + "_0.95q"] = grouped[column].quantile([0.95]).values
        self.processed[column + "_iqr"] = self.processed[column + "_0.75q"] - self.processed[column + "_0.25q"]
        self.processed[column + "_cqd"] = self.processed[column + "_iqr"] / (self.processed[column + "_0.75q"] + self.processed[column + "_0.25q"])
        # print(grouped.apply(lambda x: (x[column] - x[column].mean()).abs().mean()))
        self.processed = self.processed.join(grouped.apply(lambda x: (x[column] - x[column].mean()).abs().mean()).to_frame(name = column + "_mean_ad"))
        self.processed = self.processed.join(grouped.apply(lambda x: (x[column] - x[column].median()).abs().mean()).to_frame(name = column + "_median_ad"))
        # processed[column + "_median_ad"] = grouped.apply(lambda x: (x[column] - x[column].median()).abs().mean())
        self.processed[column + "_rms"] = grouped.apply(lambda x: (x[column] ** 2).mean() ** 0.5)
        self.processed[column + "_rss"] = grouped.apply(lambda x: (x[column] ** 2).sum() ** 0.5)
        self.processed[column + "_skewness"] = grouped[column].skew().values  #grouped[column].transform(lambda x: ((x - x.mean()) ** 3).sum()/((x.count() - 1) * (x.std() ** 3)))
        self.processed[column + "_kurtosis"] = grouped[column].apply(lambda x: x.kurtosis())
        
        return self