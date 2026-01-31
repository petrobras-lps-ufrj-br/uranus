import pandas as pd
from typing import Optional, Any, Dict, List
from ai.preprocessing.preprocessing import PreProcessing
from ai.preprocessing.preprocessing import interpolate
from ai.preprocessing.preprocessing import create_window_dataframe    
import torch
from torch.utils.data import Dataset

class DataLoader_v1(Dataset):
    """
    DataLoader class responsible for loading data from a CSV file 
    and optionally applying preprocessing steps.
    Inherits from torch.utils.data.Dataset.
    """
    def __init__(
        self, path: str, 
        window_size: int,
        col_names : Dict[str,str],
        feature_cols: list[str],
        target_cols: list[str],
        preprocessor: Optional[List[PreProcessing]] = None):
        """
        Initializes the DataLoader.

        Args:
            path (str): Path to the CSV file.
            preprocessor (Optional[PreProcessing]): A preprocessing instance to apply to the loaded data.
        """
        self.path = path
        self.window_size = window_size
        self.col_names = col_names
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.preprocessor = preprocessor
        self.data = self.load()
        if isinstance(self.data, tuple) and len(self.data) == 2:
            self.inputs, self.targets = self.data
        else:
            raise ValueError("load() must return a tuple of (inputs, targets)")

    def load(self) -> Any:
        """
        Loads the CSV file and applies the preprocessor if one exists.

        Returns:
            Any: The loaded data (as a DataFrame or transformed object).
        """
        df = pd.read_csv(self.path, index_col=0, parse_dates=True)
        df = df[self.col_names.keys()]
        df = df.rename(columns=self.col_names)
        df = interpolate(df)
        inputs, outputs = create_window_dataframe(df, 
                                     window_size=self.window_size, 
                                     feature_cols=self.feature_cols, 
                                     target_cols=self.target_cols, 
                                     concatenate=True)   
        return inputs, outputs
        
    def fit(self, indices: list[int]):
        """
        Fits the preprocessor to the data at the specified indices.
        
        Args:
            indices: List of indices to fit use for fitting.
        """
        if self.preprocessor:
            
            for idx, processor in enumerate(self.preprocessor):
                # Select rows corresponding to the indices
                # self.inputs is expected to be a DataFrame
                X = self.inputs.iloc[indices]
                self.preprocessor.fit(X)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Any:
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The (features, target) pair.
        """
        # Retrieve raw features and target
        features = self.inputs.iloc[idx].values
        target = self.targets.iloc[idx].values
        
        # Apply transformation if preprocessor is set
        if self.preprocessor:
            # transform expects 2D array (samples, features)
            # We reshape to (1, -1), transform, then flatten back to 1D
            features = self.preprocessor.transform(features.reshape(1, -1)).flatten()

        try:
            return (
                torch.tensor(features, dtype=torch.float32), 
                torch.tensor(target, dtype=torch.float32)
            )
        except (ValueError, TypeError):
            # Fallback if conversion fails
            return features, target
