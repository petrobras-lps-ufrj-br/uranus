

import sys
import os

import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import pytorch_lightning as pl
import collections


from ai.trainers.time_series import Trainer
from ai.models.model_v1 import Model_v1
from ai.evaluation import Summary
from ai.loaders import DataLoader_v1
from ai.preprocessing import Lag


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

data_path = os.path.join(os.getenv("AI_DATA_PATH"), "compressor.csv")
   
features = {
        "input_1" : "PH (CBM) 1st Stage Poly Head Dev",
        "input_2" : "PH (CBM) 1st Stage Press Rat Dev",
        "input_3" : "PH (CBM) 1st Stage ActCompr Poly Eff",
        "input_4" : "PH (CBM) 1st Stg ActCompr Poly Head",
        "target"  : "PH (CBM) 1st Stg ActCompr Poly Head",
    }

lags = {
    "input_1" : Lag(5) ,
    "input_2" : Lag(5) ,
    "input_3" : Lag(5) ,
    "input_4" : Lag(5) ,
    "target"  : Lag(-1),
}

preprocessors = {
    "input_1" : TransformerPipeline([("scaler", StandardScaler()) ]),
    "input_2" : TransformerPipeline([("scaler", StandardScaler()) ]),
    "input_3" : TransformerPipeline([("scaler", StandardScaler()) ]),
    "input_4" : TransformerPipeline([("scaler", StandardScaler()) ]),
    "target"  : TransformerPipeline([("scaler", StandardScaler()) ]),
}

input_names = ['input_1','input_2','input_3', 'input_4']
target_name = 'target'

dataset = DataLoader_v1(data_path, features, input_names, 'target', lags, preprocessors)
print(len(dataset))

cv = TimeSeriesSplit(4)

model = Model_v1(dataset)

trainer = Trainer(model, cv)

trainer.fit(dataset, num_epochs=1)
