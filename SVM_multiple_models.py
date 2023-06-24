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
from sklearn.svm import OneClassSVM
import warnings 
import random
import json
import time

subjectsPath = join(pathlib.Path().resolve(), "data\\me_time_fused\\windowed_subjects\\3_hour_20_stride")
allSubjects = [f[:-4] for f in listdir(subjectsPath) if isfile(join(subjectsPath, f))]
largeDataset = allSubjects # the specific list was removed for privacy reasons



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


def get_combinations():
    combinations: list = list()
    combinationsFirst: list = list()

    random.seed(42)
    np.random.seed(42)
    counter = 0
    while counter < 1000:
        a = random_combination(largeDataset, 12)

        if a in combinationsFirst:
            continue

        combinationsFirst.append(a)

        aux = list(set(allSubjects) - set(a))
        aux = sorted(aux)
        b = random_combination(aux, 6)

        a = list(a)
        b = list(b)
        a.extend(b)
        c = tuple(a)
        combinations.append(c)
        counter += 1

    return combinations

def subject_selector(combination: list[str]) -> dict[str, DataFrame]:
    subjectsDict: dict[str, DataFrame] = dict()
    for subject in combination:
        subjectsDict[subject] = pd.read_pickle(join(subjectsPath, subject + ".pkl"))
    return subjectsDict


def main():
    combinations = get_combinations()
    result = {i: list() for i in range(2, 13)}
    count = 0
    
    tic = time.perf_counter()
    for x in combinations:
        scalers = []
        pipes = []

        subjects = subject_selector(list(x))

        for j in range(12):
            train, test = TrainTestSplitterWindows(subjects).split([x[j]], [x[14]], 0.8)

            train = pd.concat(train.values())
            train.drop(columns=["time"], inplace=True)
            train.reset_index(drop=True, inplace=True)

            scaler = StandardScaler()
            normalizer_fit(train, scaler)
            train = normalizer_transform(train, scaler)

            scalers.append(scaler)

            train_matrix = train.loc[:, train.columns != "label"].to_numpy()
            train_matrix = np.array(train_matrix)

            
            pipe = Pipeline([("trans", Transformer(mfcc_components = 5, pca_components = 450, transformations = [])), 
                             ("svm", OneClassSVM(nu = 0.8))])

            pipe.fit(train_matrix, train["label"])
            
            pipes.append(pipe)
            print(f"{j} trained")

        toContinue = False
        scores = []
        for i in range(2, 13):
            train, test = TrainTestSplitterWindows(subjects).split(list(x[ : i]), list(x[12 : ]), 0.8)

            test = pd.concat(test.values())
            test.drop(columns=["time"], inplace=True)
            test.reset_index(drop=True, inplace=True)

            if len(test) < 500:
                toContinue = True
                break

            pred_prob = []
            lables = test["label"]

            test_copy = test.copy()
            for j in range(i):
                test = test_copy.copy()
                scaler = scalers[j]
                pipe = pipes[j]

                test = normalizer_transform(test, scaler)

                test_matrix = test.loc[:, test.columns != "label"].to_numpy()
                
                pred_prob.append(pipe.score_samples(test_matrix))
            
            score = 0
            try:
                score = roc_auc_score(lables, np.min(pred_prob, axis=0))
            except Exception as e:
                print(e)
                print(x)
                continue
            scores.append(score)
            print(f"{i} auth ====> {score}")
        
        if toContinue:
            continue

        for i in range(2, 13):
            result[i].append(scores[i - 2])

        with open("results/SVM_new/result_NvsN_mfcc_multiple_3h.json", "w") as outfile:
            json.dump(result, outfile)
        
        count += 1
        print(f"count: {count}")
        if count >= 100: 
            break
            
    toc = time.perf_counter()
    print(f"run in {toc - tic:0.4f} seconds")

warnings.filterwarnings("ignore")
main()