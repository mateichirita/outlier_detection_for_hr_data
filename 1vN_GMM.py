import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from data_processing.subject_selectors import *
from data_processing.window_splitters import *
from data_processing.train_test_spliters import *
import pathlib
from os.path import join, isfile
from os import listdir
import itertools
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from models.transformers import *
import warnings 
import random


subjectsPath = join(pathlib.Path().resolve(), "data\\me_time_fused\\windowed_subjects\\3_hour_20_stride")
allSubjects = [f[:-4] for f in listdir(subjectsPath) if isfile(join(subjectsPath, f))]

first = allSubjects.copy()

def random_combination(iterable, r):
    a = np.random.choice(first)
    iterable = iterable.copy()
    iterable.remove(a)
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r - 1))
    l = [a]
    l.extend([pool[i] for i in indices])
    return tuple(l)

def get_combinations():
    combinations: list = list()

    random.seed(42)
    np.random.seed(42)
    counter = 0
    while counter < 3000:
        a = random_combination(allSubjects, 12)

        if a not in combinations:
            combinations.append(a)
            counter += 1
    return combinations

def normalizer_fit(data: DataFrame, scaler: StandardScaler) -> None:
    matrix = data.iloc[:, data.columns != "label"].to_numpy()
    matrix = matrix.flatten()
    scaler.fit(matrix.reshape(len(matrix), -1))

def normalizer_transform(data: DataFrame, scaler: StandardScaler) -> dict[str, DataFrame]:
    matrix = data.iloc[:, data.columns != "label"].to_numpy()
    l = len(matrix[0])
    matrix = matrix.flatten()
    matrix_norm = scaler.transform(matrix.reshape(len(matrix), -1))
    matrix_norm = matrix_norm.reshape(-1, l)
    data_norm = pd.DataFrame(matrix_norm, columns=data.columns[data.columns != "label"])
    data_norm["label"] = data["label"]
    return data_norm

def custom_scorer(estimator, X):
    prob = estimator.score_samples(test_matrix_global)
    return roc_auc_score(test_global["label"], prob)

def subject_selector(combination: list[str]) -> dict[str, DataFrame]:
    subjectsDict: dict[str, DataFrame] = dict()
    for subject in combination:
        subjectsDict[subject] = pd.read_pickle(join(subjectsPath, subject + ".pkl"))
    return subjectsDict


def quantile25(data: np.array, axis) -> float:
    return np.quantile(data, 0.25, axis = axis)

def quantile75(data: np.array, axis) -> float:
    return np.quantile(data, 0.75, axis = axis)

def quantile95(data: np.array, axis) -> float:
    return np.quantile(data, 0.95, axis = axis)

def interquartileRange(data: np.array, axis) -> float:
    return quantile75(data, axis) - quantile25(data, axis)

def rangeMaxMin(data: np.array, axis) -> float:
    return np.max(data, axis = axis) - np.min(data, axis = axis)

def meanAbsoluteDeviation(data: np.array, axis) -> float:
    return np.mean(np.abs((data.T - np.mean(data, axis = axis)).T), axis = axis)

def medianAbsoluteDeviation(data: np.array, axis) -> float:
    return np.median(np.abs((data.T - np.median(data, axis = axis)).T), axis = axis)

def rootMeanSquareError(data: np.array, axis) -> float:
    return np.sqrt(np.mean(np.square((data.T - np.median(data, axis = axis)).T), axis = axis))

transformations = [[],
                [np.mean, np.std, quantile25, np.median, quantile75, quantile95, interquartileRange, rangeMaxMin, meanAbsoluteDeviation, medianAbsoluteDeviation, rootMeanSquareError],
                [np.mean, np.median],
                [np.mean, np.median, np.std],
                [np.mean, np.median, quantile25, quantile75]]

def main():
    combinations = get_combinations()

    result: DataFrame = None
    taken = 0
    for x in combinations:
        subjects = subject_selector(list(x))
        
        train, test = TrainTestSplitterWindows(subjects).split([x[0]], list(x[1:]), 0.8)


        train = pd.concat(train.values())
        test = pd.concat(test.values())
        train.drop(columns=["time"], inplace=True)
        test.drop(columns=["time"], inplace=True)
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        scaler = StandardScaler()
        normalizer_fit(train, scaler)
        train = normalizer_transform(train, scaler)
        test = normalizer_transform(test, scaler)

        train_matrix = train.loc[:, train.columns != "label"].to_numpy()
        test_matrix = test.loc[:, test.columns != "label"].to_numpy()

        train_matrix = np.array(train_matrix)
        test_matrix = np.array(test_matrix)

        global test_matrix_global
        test_matrix_global = test_matrix
        global test_global
        test_global = test

        
        if len(train_matrix) < 1200 or len(test_matrix) < 500:
            continue

        

        pipe = Pipeline([("trans", Transformer()), ("gmm", GaussianMixture())])
        param_grid = {
            "trans__transformations": transformations,
            "trans__pca_components": [0, 4, 10, 20, 50, 100, 250, 400, 450, 500],
            "trans__mfcc_components": [0, 5, 10, 15, 20],
            "gmm__n_components": [4, 10, 20, 40]
        }

        grid = GridSearchCV(pipe, param_grid, cv=[(slice(None), slice(None))], scoring=custom_scorer, n_jobs=-1, refit = True)
        grid.fit(train_matrix)
        
        cv_results = pd.DataFrame(grid.cv_results_)
        df = pd.DataFrame(cv_results[["params", "mean_test_score"]])
        if result is None :
            df.rename(columns={"mean_test_score": str(x)}, inplace=True)
            result = df
        else:
            df.drop("params", axis=1, inplace=True)
            df.rename(columns={"mean_test_score": str(x)}, inplace=True)
            result = result.join(df, how = "outer")

        taken += 1
        result.to_csv("results/GMM_new/result_1vsN_3h.csv", index=False)

        with open("progress.txt", "a") as f:
            f.write(f"{taken} ======> done\n")

        print(f"{taken} ======> done")
        if taken >=  100:
            break



warnings.filterwarnings("ignore")
main()