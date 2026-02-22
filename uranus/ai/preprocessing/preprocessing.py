__all__ = [
    "PreProcessing",
    "Lag",
]

import numpy as np
import pandas as pd
import collections

from abc import ABC, abstractmethod
from typing import Any

from sklearn.preprocessing import StandardScaler
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.series.summarize import WindowSummarizer
from sktime.transformations.series.impute import Imputer
from sktime.transformations.compose import TransformerPipeline
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.transformations.series.dropna import DropNA
from sktime.transformations.series.subset import IndexSubset

class PreProcessing(ABC):
    """
    Base class for all preprocessing steps.
    """
    
    @abstractmethod
    def fit(self, X: Any, y: Any = None):
        """
        Fit the preprocessor to the data.
        """
        pass

    @abstractmethod
    def transform(self, X: Any) -> Any:
        """
        Transform the data.
        """
        pass

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        """
        Fit to data, then transform it.
        """
        self.fit(X, y)
        return self.transform(X)





class Lag:
    def __init__(self, lag , imputer_method = 'linear'):
        self.lag = lag
        self.imputer_method=imputer_method
        
    def __call__(self, df):

        lags = [lag for lag in range(self.lag)] if self.lag>=0 else [ -1*(lag + 1) for lag in range(-1*self.lag)]
        # This pipeline will process the entire DataFrame and return the result
        pipeline = TransformerPipeline([
            ("imputer", Imputer(method=self.imputer_method)),
            ("lags"   , WindowSummarizer(
                            lag_feature={"lag": lags},
                            target_cols=df.columns.tolist(),
                        )
            ),
            ("dropna", DropNA()),
            ])
        
        output_df  = pipeline.fit_transform(df)
        if self.lag < 0:
            col_names = { col_name : col_name.replace('lag_-', 'next_') for col_name in output_df.columns.tolist() }
            output_df = output_df.rename(columns=col_names)
        return output_df
