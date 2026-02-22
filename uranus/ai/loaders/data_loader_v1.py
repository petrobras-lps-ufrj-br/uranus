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
        preprocessors          : Dict[str,str]
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
        self.data = self.load()
        self.preprocessors = preprocessors

    def index(self):
        return self.data[[*self.features][0]].index

    def load(self) -> Any:

        df = pd.read_csv(self.path, index_col=0, parse_dates=True)#.iloc[0:100]

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

    def fit(self, indices : List[int] ):
        for feature_name, feature_df in self.data.items():
            if feature_name in self.preprocessors:
                pipeline = self.preprocessors[feature_name]
                pipeline.fit(feature_df.iloc[indices])

    def __getitem__(self, indices : List[int]):

        inputs = {}
        for feature_name in self.input_features:
            pipeline = self.preprocessors[feature_name] if feature_name in self.preprocessors else None 
            data_values = self.data[feature_name].iloc[indices].values.reshape(1, -1)
            data_values  = pipeline.transform(data_values) if pipeline is not None else data_values
            data_values  = torch.tensor(data_values, dtype=torch.float32)
            inputs[feature_name] = data_values

        targets = {}
        for feature_name in self.target_feature:
            pipeline = self.preprocessors[feature_name] if feature_name in self.preprocessors else None 
            data_values = self.data[feature_name].iloc[indices].values.reshape(1, -1)
            data_values  = pipeline.transform(data_values) if pipeline is not None else data_values
            data_values  = torch.tensor(data_values, dtype=torch.float32)
            targets[feature_name] = data_values

        return inputs, targets
        
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data[[*self.features][0]])
