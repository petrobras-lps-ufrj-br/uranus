__all__ = [
    "DataLoader_v1",
]

import pandas as pd
from typing import Optional, Any, Dict, List
from uranus.ai.preprocessing import PreProcessing
import torch
from torch.utils.data import Dataset
from copy import deepcopy
from loguru import logger
import collections
from functools import reduce



class DataLoader_v1(Dataset):

    def __init__(
        self, 
        path                   : str,
        features               : Dict[str,str],
        input_features         : List[str],
        target_feature         : List[str],
        lags                   : Dict[str,str],
        preprocessors          : Dict[str,str],
        dry_run_with           : Optional[int] = None,
        device                 : Optional[str] = 'cpu',
        ):
        """
        Initializes the DataLoader.

        Args:
            path (str): Path to the CSV file.
            preprocessor (Optional[PreProcessing]): A preprocessing instance to apply to the loaded data.
        """
        self.path = path
        self.features = features
        self.input_features = input_features if type(input_features) == list else [input_features]
        self.target_feature = target_feature if type(target_feature) == list else [target_feature]
        self.lags = lags
        self.data = self.load(dry_run_with)
        self._data_processed = {}
        self.preprocessors = preprocessors
        self.device = device

    def index(self):
        return self.data[[*self.features][0]].index

    def load(self, dry_run_with : Optional[int] = None) -> Any:

        if dry_run_with is not None:
            df = pd.read_csv(self.path, index_col=0, parse_dates=True).iloc[0:dry_run_with]
        else:
            df = pd.read_csv(self.path, index_col=0, parse_dates=True)

        data = collections.OrderedDict()

        # lag all features
        for feature_name, col_name in self.features.items():
            feature_df = df[col_name].to_frame()
            feature_df = feature_df.rename(columns={col_name:feature_name})
            feature_df = self.lags[feature_name](feature_df)
            data[feature_name]=feature_df
      
        # Extract the indices and find the common intersection
        index = reduce(lambda left, right: left.intersection(right), [feature_df.index for feature_df in data.values()])
        data = {feature_name:feature_df.loc[index] for feature_name, feature_df in data.items()}
        return data

    def fit(self, indices : List[int]):
        for feature_name, feature_df in self.data.items():
            if feature_name in self.preprocessors:
                pipeline = self.preprocessors[feature_name]
                # fit on training set
                pipeline.fit(feature_df.iloc[indices])
                # transform all data
                self._data_processed[feature_name] = pipeline.transform(feature_df)
            else:
                self._data_processed[feature_name] = feature_df

            self._data_processed[feature_name] = torch.tensor(self._data_processed[feature_name].values, dtype=torch.float32)
            self._data_processed[feature_name].to(self.device)

    def __getitem__(self, index : int):
        inputs = {}
        for feature_name in self.input_features:
            data_values = self._data_processed[feature_name][index].reshape(1, -1)
            inputs[feature_name] = data_values

        targets = {}
        for feature_name in self.target_feature:
            data_values = self._data_processed[feature_name][index].reshape(1, -1)
            targets[feature_name] = data_values

        return inputs, targets

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data[[*self.features][0]])


