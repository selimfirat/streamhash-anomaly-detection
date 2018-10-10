#!/usr/bin/env python

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import Normalizer
import time
import pickle
from config import algorithms, num_trials, num_hp_trials, datasets

np.random.seed(1)


def pick_random_params(params_bag):
    params = {}
    for key, val in params_bag.items():
        ri = np.random.random_integers(0, len(val) - 1)
        params[key] = val[ri]

    return params


st_time = time.time()

scores = {}

for algorithm in algorithms:
    if not algorithm["active"]:
        continue

    scores[algorithm["name"]] = {}
    scoring = scores[algorithm["name"]]

    for j in range(1, num_hp_trials+1):
        params = pick_random_params(algorithm["params"])
        sh_params = pick_random_params(algorithm["sh_params"])

        scoring["params" + str(j)] = {
            "params": params,
            "sh_params": sh_params,
            "scores": []
        }
        for dataset in datasets:
            if not dataset["active"]:
                continue

            X = np.loadtxt("data/" + dataset["file"], delimiter=",")
            y = np.array([0] * (len(X) - dataset["num_anomalies"]) + [1] * dataset["num_anomalies"])

            start_time = time.time()
            ap = 0
            auc = 0
            for i in range(1, num_trials+1):
                print(algorithm["name"], dataset["name"] + ":", i, "-", j)
                _ap, _auc = algorithm["funct"](X, y, percentage=(1 - (dataset["num_anomalies"] / len(X))), params=params, sh_params=sh_params)

                ap += _ap / num_trials
                auc += _auc / num_trials

            scoring["params" + str(j)]["scores"].append({
                "dataset": dataset["name"],
                "ap": ap,
                "auc": auc,
                "runned_in_secs": (time.time() - start_time)
            })
    pickle.dump(scoring, open("pkl/" + str(st_time) + "_" + algorithm["name"] + ".pkl", "wb+"))


print(scores)