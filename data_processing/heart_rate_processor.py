from __future__ import annotations
from .data_processor import DataProcessor
from pandas import DataFrame
import pandas as pd


class HeartRateDataProcessor(DataProcessor):
    # window = "H" # hour\

    def __init__(self, data) -> None:
        super().__init__(data)
    
    def process(self) -> HeartRateDataProcessor:
        # grouped = self.data.groupby("window")
        
        # self.processed["hr_mean"] = grouped["hr"].mean().values
        # self.processed["hr_median"] = grouped["hr"].median().values
        # self.processed["hr_std"] = grouped["hr"].std().values
        # self.processed["hr_variance"] = grouped["hr"].var().values
        # self.processed["hr_cv"] = self.processed["hr_std"] / self.processed["hr_mean"]
        # self.processed["hr_min"] = grouped["hr"].min().values
        # self.processed["hr_max"] = grouped["hr"].max().values
        # self.processed["hr_range"] = self.processed["hr_max"] - self.processed["hr_min"]
        # self.processed["hr_cr"] = self.processed["hr_range"] / (self.processed["hr_max"] + self.processed["hr_min"])
        # self.processed["hr_0.25q"] = grouped["hr"].quantile([0.25]).values
        # self.processed["hr_0.75q"] = grouped["hr"].quantile([0.75]).values
        # self.processed["hr_0.95q"] = grouped["hr"].quantile([0.95]).values
        # self.processed["hr_iqr"] = self.processed["hr_0.75q"] - self.processed["hr_0.25q"]
        # self.processed["hr_cqd"] = self.processed["hr_iqr"] / (self.processed["hr_0.75q"] + self.processed["hr_0.25q"])
        # # print(grouped.apply(lambda x: (x["hr"] - x["hr"].mean()).abs().mean()))
        # self.processed = self.processed.join(grouped.apply(lambda x: (x["hr"] - x["hr"].mean()).abs().mean()).to_frame(name = "hr_mean_ad"))
        # self.processed = self.processed.join(grouped.apply(lambda x: (x["hr"] - x["hr"].median()).abs().mean()).to_frame(name = "hr_median_ad"))
        # # processed["hr_median_ad"] = grouped.apply(lambda x: (x["hr"] - x["hr"].median()).abs().mean())
        # self.processed["hr_rms"] = grouped.apply(lambda x: (x["hr"] ** 2).mean() ** 0.5)
        # self.processed["hr_rss"] = grouped.apply(lambda x: (x["hr"] ** 2).sum() ** 0.5)
        # self.processed["hr_skewness"] = grouped["hr"].skew().values  #grouped["hr"].transform(lambda x: ((x - x.mean()) ** 3).sum()/((x.count() - 1) * (x.std() ** 3)))
        # self.processed["hr_kurtosis"] = grouped["hr"].apply(lambda x: x.kurtosis())

        super().process_common_features("hr")
        
        return self
    
    def add_resting_hr(self) -> HeartRateDataProcessor:
        self.data["byMinute"] = pd.to_datetime(self.data["time"].dt.floor("min"))
        grouped = self.data.groupby(["byMinute", "window"]).agg({"hr": "mean", "steps": "sum"})
        grouped["active"] = (grouped["steps"] > 0)
        hrMeanSeries = grouped.groupby("window").apply(lambda x: x[x["steps"] == 0]["hr"].mean())
        self.processed = self.processed.join(hrMeanSeries.to_frame(name = "hr_rhr"))
        return self
    
    # def save(self, path):
    #     self.processed.to_csv(path, encoding='utf-8', index=False)
        # processed: DataFrame = self.processed

        
        
    
    # def split_by_date(self) -> list[DataFrame]:
    #     dateList = self.data["time"].dt.date.unique()
    #     split: list[DataFrame] = list()

    #     for date in dateList:
    #         print(date)
    #         split.append(self.data[(self.data["time"].dt.date == date)])

        # return split