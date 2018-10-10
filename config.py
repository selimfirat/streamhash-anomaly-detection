from algorithms.standard_algorithms import xstream, iforest, ocsvm, shiforest, shocsvm
import numpy as np

num_trials = 1
num_hp_trials = 1

algorithms = [{
    "name": "xstream",
    "funct": xstream,
    "active": False,
    "params": {
        "k": np.arange(1, 251),
        "nchains": np.arange(1, 251),
        "depth": np.arange(1, 51)
    },
    "sh_params": {}
},
    {
        "name": "iforest",
        "funct": iforest,
        "active": True,
        "params": {
            "n_estimators": np.arange(10, 251),
            "max_samples": np.arange(10, 350)
        },
        "sh_params": {}
    },
    {
        "name": "shiforest",
        "funct": shiforest,
        "active": True,
        "params": {
            "n_estimators": np.arange(10, 251),
            "max_samples": np.arange(10, 350)
        },
        "sh_params": {
            "n_components": np.arange(1, 251),
            "density": np.arange(0, 1.0, 0.01)
        }
    },
    {
        "name": "ocsvm",
        "funct": ocsvm,
        "active": True,
        "params": {
            "kernel": ["linear", "poly", "sigmoid", "rbf"],
            "gamma": np.arange(0, 1.0, 0.001),
            "nu": np.arange(0, 1.0, 0.01)
        },
        "sh_params": {
            "n_components": np.arange(1, 251),
            "density": np.arange(0, 1.0, 0.01)
        }
    },
    {
        "name": "shocsvm",
        "funct": shocsvm,
        "active": True,
        "params": {
            "kernel": ["linear", "poly", "sigmoid", "rbf"],
            "gamma": np.arange(0, 1.0, 0.001),
            "nu": np.arange(0, 1.0, 0.01)
        },
        "sh_params": {
            "n_components": np.arange(1, 251),
            "density": np.arange(0, 1.0, 0.01)
        }
    },
]

datasets = [
    {
        "name": "gisette",
        "file": "gisette_sampled.txt",
        "num_anomalies": 351,
        "active": True
    },
    {
        "name": "isolet",
        "file": "isolet_sampled.txt",
        "num_anomalies": 389,
        "active": True
    },
    {
        "name": "letter",
        "file": "letter-recognition_sampled.txt",
        "num_anomalies": 389,
        "active": True
    },
    {
        "name": "madelon",
        "file": "madelon_sampled.txt",
        "num_anomalies": 130,
        "active": True
    },
    {
        "name": "cancer",
        "file": "breast-cancer-wisconsin_sampled.txt",
        "num_anomalies": 28,
        "active": True
    },
    {
        "name": "ionosphere",
        "file": "ionosphere_sampled.txt",
        "num_anomalies": 17,
        "active": True
    },
    {
        "name": "telescope",
        "file": "magic-telescope_sampled.txt",
        "num_anomalies": 951,
        "active": True
    },
    {
        "name": "indians",
        "file": "pima-indians_sampled.txt",
        "num_anomalies": 38,
        "active": True
    }
]
