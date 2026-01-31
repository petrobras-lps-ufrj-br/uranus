from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd
import collections

from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

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


class StandardScale(PreProcessing):
    """
    StandardScaler wrapper that follows the PreProcessing interface.
    Uses sklearn.preprocessing.StandardScaler internally.
    """
    def __init__(self):
        self.scaler = SklearnStandardScaler()

    def fit(self, X: Any, y: Any = None):
        # Handle potential 3D input (batch, seq_len, features) for time series
        # Standard scaler usually expects 2D. We might need to reshape or apply per channel.
        # For simplicity in many time series tasks, we flatten to (N*T, F) or treat (N, T*F).
        # However, usually we want to scale features independent of time step.
        # If X is (N, T, F), we reshape to (N*T, F) fit, then reshape back?
        # Or if X is (N, F), just fit.
        
        # Let's assume input might be numpy array or similar
        X_np = X if isinstance(X, np.ndarray) else np.array(X)
        
        if X_np.ndim == 3:
            N, T, F = X_np.shape
            X_reshaped = X_np.reshape(-1, F)
            self.scaler.fit(X_reshaped)
        else:
            self.scaler.fit(X_np)
        return self

    def transform(self, X: Any) -> Any:
        X_np = X if isinstance(X, np.ndarray) else np.array(X)
        
        if X_np.ndim == 3:
            N, T, F = X_np.shape
            X_reshaped = X_np.reshape(-1, F)
            X_scaled = self.scaler.transform(X_reshaped)
            return X_scaled.reshape(N, T, F)
        
        return self.scaler.transform(X_np)


def interpolate(df : pd.DataFrame, method='linear') -> pd.DataFrame :
    return df.interpolate(method=method, limit_direction='both')

def create_window_dataframe(df : pd.DataFrame, window_size : int, feature_cols : list[str], target_cols : list[str], concatenate : bool) -> pd.DataFrame:
    data_inputs = collections.OrderedDict()
    data_outputs = collections.OrderedDict()
    for feature_name in feature_cols:
        df_feature = df[feature_name].to_frame()
        for lag in range(window_size - 1):
            df_feature[f"{feature_name}_lag_{lag+1}"] = df_feature[feature_name].shift(lag+1)
    
        df_feature = df_feature.dropna()
        data_inputs[feature_name] = df_feature

    for target_col in target_cols:
        data_outputs[target_col] = df[target_col].iloc[window_size-1::].to_frame()
    if concatenate:
        data_inputs = pd.concat(data_inputs.values(),axis=1)
        data_outputs = pd.concat(data_outputs.values(),axis=1)
    return data_inputs, data_outputs