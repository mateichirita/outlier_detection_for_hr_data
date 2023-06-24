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
import warnings 

def quantile25(data: np.array) -> float:
    return np.quantile(data, 0.25)
def quantile75(data: np.array) -> float:
    return np.quantile(data, 0.75)
def quantile95(data: np.array) -> float:
    return np.quantile(data, 0.95)

def interquartileRange(data: np.array) -> float:
    return quantile75(data) - quantile25(data)

transformations = [[np.nanmean, np.nanmedian, quantile25, quantile75, interquartileRange],
                   [np.nanmean, np.nanmedian, quantile25, quantile75, interquartileRange, lambda x: np.nanmax(x) - np.nanmin(x)]]

def main():
    subjectsPath = join(pathlib.Path().resolve(), "data\\me_time_fused\\subjects")
    allSubjects = [f[:-4] for f in listdir(subjectsPath) if isfile(join(subjectsPath, f))]

    pairs: list = list(itertools.combinations(allSubjects, 2))
    np.random.seed(42)
    np.random.shuffle(pairs)


    # taken = 0
    # aucSum = 0
    # aucs = []
    # for x in pairs:
    for i, t in enumerate(transformations):
        table = "| Authorized   |      Unauthorized      |  AUC |\n"
        table += "|--------------|------------------------|------|\n"
        n_components = [3, 4, 6, 7, 9]
        print(i)
        for n in n_components:
            taken = 0
            aucSum = 0
            aucs = []
            for x in pairs:
                
                try:
                    subjects: dict[str, DataFrame] = SubjectSelectorList(subjectsPath,list(x), True, False).get_subjects()
                except Exception as e:
                    print(e)
                    continue


                taken += 1

                trainTestSplitter = TrainTestSplitterRawData(1, 0.8, auth = [x[0]], nonAuth = [x[1]])
                train, test = trainTestSplitter.splitBeforeWindowing(subjects)
                auth, unauth = trainTestSplitter.getAuthAndUnauthIds()

                # print(test)

                def groupSplitterPerTimeframe(data: DataFrame) -> np.array:
                    first = pd.to_datetime(np.array(data["time"])[0])
                    return np.array(data["time"].dt.hour // 6 + (data["time"] - first).dt.days * 4)
                
                def groupSplitterPerHour(data: DataFrame) -> np.array:
                    # print(np.array(data["time"]))
                    first = pd.to_datetime(np.array(data["time"])[0])
                    # print(np.array(data["time"].dt.hour + (data["time"] - first).dt.days * 24))
                    return np.array(data["time"].dt.hour + (data["time"] - first).dt.days * 24)

                # print()
                # print()
                # print(test)
                # try:
                train = \
                    WindowSplitterNoOverlap(#stepTransformations=[np.nanmean, np.nanmedian, np.nanvar, lambda x: np.nanmax(x) - np.nanmin(x)],\
                                            hrTransformations = t).split(train, groupFunction=groupSplitterPerHour)
                test = \
                    WindowSplitterNoOverlap(#stepTransformations=[np.nanmean, np.nanmedian, np.nanvar, lambda x: np.nanmax(x) - np.nanmin(x)],\
                                            hrTransformations = t).split(test, groupFunction=groupSplitterPerHour)
                # except Exception as e:
                #     print(e)
                #     taken -= 1
                #     continue
                
                
                train = pd.concat(train.values())
                test = pd.concat(test.values())
                train.reset_index(drop=True, inplace=True)
                test.reset_index(drop=True, inplace=True)
                

                train_matrix = train.loc[:, train.columns != "label"].to_numpy()
                test_matrix = test.loc[:, test.columns != "label"].to_numpy()




                if len(train) < 100 or len(test) < 100:
                    taken -= 1
                    continue

                try:
                    gmm = GaussianMixture(n_components = n).fit(train_matrix)
                    # bic.append(gmm.bic(train_matrix))

                    # if bic[-1] < lowestBic:
                    #     lowestBic = bic[-1]
                    #     bestGmm = gmm

                    # print(bestGmm)
                    # gmm = bestGmm
                    prob = gmm.score_samples(test_matrix)
                    # print(prob)
                        
                    auc = roc_auc_score(test["label"], prob)
                    
                except:
                    print("ValueError")
                    taken -= 1
                    continue

                aucSum += auc
                aucs.append(auc)

                # table += f"| {auth} | {unauth} | {auc} |\n"
                print (f"{taken}  ------->  {auth} | {unauth} | {auc}")
                if taken >= 100:
                    break
            
            table += f"| Average for {n} | | {aucSum / taken} +- {np.std(aucs)} |\n"
            aux: int = i + 1
            print(aux)
            with open(f"result{aux}.txt", "w") as f:
                f.write(table)


warnings.filterwarnings("ignore")
main()