

import sys
import os
import argparse
import json

import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import pytorch_lightning as pl
import collections


from uranus.ai.trainers.time_series import Trainer
from uranus.ai.models.mlp_v1 import MLP_v1
from uranus.ai.evaluation import Summary
from uranus.ai.loaders import DataLoader_v1
from uranus.ai.preprocessing import Lag
from uranus.ai import get_argparser_formatter


from torch.utils.data import Dataset
from torch.utils.data import Subset, DataLoader
from typing import List, Dict, Any
from functools import reduce

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.series.summarize import WindowSummarizer
from sktime.transformations.series.impute import Imputer
from sktime.transformations.compose import TransformerPipeline
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.transformations.series.dropna import DropNA
from sktime.transformations.series.subset import IndexSubset

from uranus.ai.evaluation.summary import Summary
from uranus.ai.callbacks.model_checkpoint import ModelCheckpoint

# Argument Parsing
parser = argparse.ArgumentParser(description="Train Time Series Model", formatter_class=get_argparser_formatter())
parser.add_argument("--path", "-p", dest="csv_path", type=str, default=None, help="Path to the CSV file")
parser.add_argument("--fold", "-f", dest="fold", type=int, default=None, help="Specific fold to train on")
parser.add_argument("--epochs", "-e", dest="epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--splits", "-s", dest="splits", type=int, default=10, help="Number of splits for TimeSeriesSplit")
parser.add_argument("--job_json", "-j", dest="job_json", type=str, default=None, help="Path to a JSON job configuration file")

args = parser.parse_args()

data_path = args.csv_path
fold      = args.fold
epochs    = args.epochs
splits    = args.splits

if args.job_json:
    if os.path.exists(args.job_json):
        with open(args.job_json, 'r') as f:
            job_config = json.load(f)
        data_path = job_config.get("csv_path", data_path)
        fold      = job_config.get("fold", fold)
        epochs    = job_config.get("epochs", epochs)
        splits    = job_config.get("splits", splits)

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

if not os.path.exists(data_path):
    raise ValueError(f"Data path not found: {data_path}")


   
features = {
        "input"   : "PH (CBM) 1st Stage Poly Head Dev",
        "extra_1" : "PH (CBM) 1st Stage Press Rat Dev",
        "extra_2" : "PH (CBM) 1st Stage ActCompr Poly Eff",
        "extra_3" : "PH (CBM) 1st Stg ActCompr Poly Head",
        "target"  : "PH (CBM) 1st Stage Poly Head Dev",
    }

lags = {
    "input"   : Lag(2) ,
    "extra_1" : Lag(2) ,
    "extra_2" : Lag(2) ,
    "extra_3" : Lag(2) ,
    "target"  : Lag(-1),
}

preprocessors = {
    "input"   : TransformerPipeline([("scaler", StandardScaler()) ]),
    "extra_1" : TransformerPipeline([("scaler", StandardScaler()) ]),
    "extra_2" : TransformerPipeline([("scaler", StandardScaler()) ]),
    "extra_3" : TransformerPipeline([("scaler", StandardScaler()) ]),
    "target"  : TransformerPipeline([("scaler", StandardScaler()) ]),
}

input_names = ['input', 'extra_1', 'extra_2', 'extra_3']
target_name = 'target'

dataset = DataLoader_v1(data_path, features, input_names, 'target', lags, preprocessors)

cv = TimeSeriesSplit(splits)

model = MLP_v1(dataset, n_hidden=2)

callbacks = [
    pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
]

evaluators = [
    Summary("Summary"),
]

trainer = Trainer(model, cv, callbacks=callbacks, evaluators=evaluators)

trainer.fit(dataset, num_epochs=epochs, specific_fold=fold)
