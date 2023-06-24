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
from models.svdd import SVDD
from sklearn.metrics import confusion_matrix as conf_mat
from sklearn.metrics import accuracy_score
from sklearn.svm import OneClassSVM
import random
import warnings 
import timeit




def normalizer_fit(data: dict[str, DataFrame], scaler: StandardScaler, columns: list[str]) -> None:
    dataTrainer: DataFrame = pd.concat(data.values())
    scaler.fit(dataTrainer[columns].to_numpy().reshape(-1, len(columns)))

def normalizer_transform(data: dict[str, DataFrame], scaler: StandardScaler, columns: list[str]) -> dict[str, DataFrame]:
    for key in data.keys():
        data[key][columns] = scaler.transform(data[key][columns].to_numpy().reshape(-1, len(columns)))

    return data

def groupSplitterPerHour(data: DataFrame) -> tuple[np.array, int]:
    first = pd.to_datetime(np.array(data["time"])[0])
    first = first.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    return np.array(data["time"].dt.hour + (data["time"] - first).dt.days * 24), 720

def groupSplitterPerTimeframe(data: DataFrame) -> tuple[np.array, int]:
    first = pd.to_datetime(np.array(data["time"])[0])
    first = first.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    return np.array(data["time"].dt.hour // 6 + (data["time"] - first).dt.days * 4), 4320

def custom_scorer(estimator, X, y):
    prob = estimator.score_samples(test_matrix_global)
    return roc_auc_score(test_global["label"], prob)

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    a = estimator.steps[0][1].transform(X)
    return -estimator.steps[-1][1].bic(a)

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

def main():
    subjectsPath = join(pathlib.Path().resolve(), "data\\me_time_fused\\subjects")
    allSubjects = [f[:-4] for f in listdir(subjectsPath) if isfile(join(subjectsPath, f))]

    combinations: list = list()
    random.seed(42)
    counter = 0
    while counter < 3000:
        a = random_combination(allSubjects, 12)

        if a not in combinations:
            combinations.append(a)
            counter += 1

    result: DataFrame = None
    taken = 0

    for x in combinations:
        try:
            subjects: dict[str, DataFrame] = SubjectSelectorList(subjectsPath,list(x), True, False).get_subjects()
        except Exception as e:
            print(e)
            continue

        train, test = TrainTestSplitterRawData(1, 0.8, auth = [x[0]], nonAuth = list(x[1 : ])).splitBeforeWindowing(subjects)
        

        scaler = StandardScaler()
        normalizer_fit(train, scaler, ["hr"])
        train = normalizer_transform(train, scaler, ["hr"])
        test = normalizer_transform(test, scaler, ["hr"])


        train = WindowSplitterNoFunctions().split(train, groupFunction=groupSplitterPerHour) 
        test = WindowSplitterNoFunctions().split(test, groupFunction=groupSplitterPerHour) 
        

        train = pd.concat(train.values())
        test = pd.concat(test.values())
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        train_matrix = train.loc[:, train.columns != "label"].to_numpy()
        test_matrix = test.loc[:, test.columns != "label"].to_numpy()
        global test_matrix_global
        test_matrix_global = test_matrix
        global test_global
        test_global = test

        if len(train_matrix) < 500 or len(test_matrix) < 500:
            continue

        print(len(train_matrix))


        pipe = Pipeline([("pca", PCA()), ("svm", OneClassSVM(gamma="auto"))])
        param_grid = {
            "pca__n_components": [4, 10, 20, 25, 42, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            "svm__nu": [0.5, 0.7, 0.8, 0.9],
        }

        grid = GridSearchCV(pipe, param_grid, cv=[(slice(None), slice(None))], scoring=custom_scorer, n_jobs=-1, refit = True)
        grid.fit(train_matrix, train["label"])


        
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
        print(result)
        result.to_csv("results/OneClassSVM/result_1vsN_more_auc.csv", index=False)
        

        print(f"{taken} ======> done")
        if taken >= 100:
            break
    
    print(result)
    result.to_csv("results/OneClassSVM/result_1vsN_more_auc.csv", index=False)


main()