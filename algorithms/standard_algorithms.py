from algorithms.Chains import Chains
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from algorithms.StreamhashProjection import StreamhashProjection
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import Normalizer
import time
import pickle

def xstream(X, y, percentage=None, params={}, sh_params={}):
    cf = Chains(**params, **sh_params)
    cf.fit(X)
    anomalyscores = -cf.score(X)
    ap = average_precision_score(y, anomalyscores)
    auc = roc_auc_score(y, anomalyscores)

    return ap, auc

def iforest(X, y, percentage=None, params={}, sh_params={}):
    cf = IsolationForest(**params)
    cf.fit(X)
    anomalyscores = -cf.decision_function(X)
    ap = average_precision_score(y, anomalyscores)
    auc = roc_auc_score(y, anomalyscores)

    return ap, auc

def ocsvm(X, y, percentage=None, params={}, sh_params={}):

    normalizer = Normalizer(norm="l1")
    X = normalizer.fit_transform(X)

    cf = OneClassSVM(**params)
    cf.fit(X)
    anomalyscores = -cf.decision_function(X)

    ap = average_precision_score(y, anomalyscores)
    auc = roc_auc_score(y, anomalyscores)

    return ap, auc

def shiforest(X, y, percentage=None, params={}, sh_params={}):
    projector = StreamhashProjection(**sh_params)
    projected_X = projector.fit_transform(X)
    cf = IsolationForest(**params)
    cf.fit(projected_X)
    anomalyscores = -cf.decision_function(projected_X)
    ap = average_precision_score(y, anomalyscores)
    auc = roc_auc_score(y, anomalyscores)

    return ap, auc

def shocsvm(X, y, percentage=None, params={}, sh_params={}):
    projector = StreamhashProjection(**sh_params)
    X = projector.fit_transform(X)

    normalizer = Normalizer(norm="l1")
    X = normalizer.fit_transform(X)

    cf = OneClassSVM(**params) # nu=percentage,
    cf.fit(X)

    anomalyscores = -cf.decision_function(X)
    ap = average_precision_score(y, anomalyscores)
    auc = roc_auc_score(y, anomalyscores)

    return ap, auc