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
import warnings 


def main():
    subjectsPath = join(pathlib.Path().resolve(), "data\\me_time_fused\\subjects")
    allSubjects = [f[:-4] for f in listdir(subjectsPath) if isfile(join(subjectsPath, f))]

    pairs: list = list(itertools.combinations(allSubjects, 2))
    np.random.shuffle(pairs)

    table = "| Authorized   |      Unauthorized      |  AUC |\n"
    table += "|--------------|------------------------|------|\n"

    taken = 0
    aucSum = 0
    for x in pairs:
        try:
            subjects: dict[str, DataFrame] = SubjectSelectorList(subjectsPath, ['B5F5KK', '122'], True, True).get_subjects()
        except Exception as e:
            print(e)
            continue

        taken += 1
        windowedData: dict[str, DataFrame] = \
            WindowSplitterNoOverlap(windowSize = 720, \
                                    hrTransformations = [np.nanmean, np.nanstd, np.nanmedian, np.nanvar, np.nanmin, np.nanmax], \
                                    stepTransformations = [np.nanmean, np.nanstd, np.nanmedian, np.nanvar, np.nanmin, np.nanmax]).split(subjects)
        
        trainTestSplitter: TrainTestSplitter = TrainTestSplitterOutlierPersons(1, 0.8)
        train, test = trainTestSplitter.split(windowedData)
        auth, unauth = trainTestSplitter.getAuthAndUnauthIds()

        train_matrix = train.loc[:, test.columns != "label"].to_numpy()
        test_matrix = test.loc[:, test.columns != "label"].to_numpy()

        gmm = GaussianMixture(n_components = 4).fit(train_matrix)
        score = gmm.score_samples(test_matrix)
        auc = roc_auc_score(test["label"], score)

        aucSum += auc

        print(train)
        print(test)

        table += f"| {auth} | {unauth} | {auc} |\n"
        print (f"{taken}  ------->  {auth} | {unauth} | {auc}")
        if taken >= 1:
            break
    
    table += f"| Average | | {aucSum / taken} |"
    with open("result.txt", "w") as f:
        f.write(table)


warnings.filterwarnings("ignore")
main()