__all__ = [
    "Summary",
]

from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
from tqdm import tqdm as progress_bar
from prettytable import PrettyTable
from torch.utils.data import DataLoader


class Summary:
    """
    Class to calculate and store time series evaluation metrics.
    """
    def __init__(self, name: str):
        """
        Args:
            name: Name of the evaluator.
        """
        self.name = name

    def __call__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        """
        Computes various time series metrics: MSE, MAE, MAPE, R2.
        
        Returns:
            Dictionary containing the calculated metrics.
        """
        # Ensure model is in eval mode
        model.eval()
        history      = {}
        y_train_pred = []
        y_train_true = []
        y_val_pred   = []
        y_val_true   = []

        # Fill train true and pred values
        for batch in progress_bar(train_loader, desc="Training", leave=False):
            X, y_true = model.prepare_batch(batch)
            y_pred = model(X)
            y_train_pred.extend(y_pred.detach().cpu().numpy())
            y_train_true.extend(y_true.detach().cpu().numpy())

        # Fill val true and pred values
        for batch in progress_bar(val_loader, desc="Validation", leave=False):
            X, y_true = model.prepare_batch(batch)
            y_pred = model(X)
            y_val_pred.extend(y_pred.detach().cpu().numpy())
            y_val_true.extend(y_true.detach().cpu().numpy())

        # Calculate metrics
        history["mse"]      = mean_squared_error(y_train_true, y_train_pred)
        history["mae"]      = mean_absolute_error(y_train_true, y_train_pred)
        history["mape"]     = mean_absolute_percentage_error(y_train_true, y_train_pred)
        history["r2"]       = r2_score(y_train_true, y_train_pred)
        history["val_mse"]  = mean_squared_error(y_val_true, y_val_pred)
        history["val_mae"]  = mean_absolute_error(y_val_true, y_val_pred)
        history["val_mape"] = mean_absolute_percentage_error(y_val_true, y_val_pred)
        history["val_r2"]   = r2_score(y_val_true, y_val_pred)
        
        
        # Print table
        t = PrettyTable()
        t.field_names = ["📊 Metric", "Training", "Validation"]
        t.add_row(["MSE" , f"{history['mse']:.4f}" , f"{history['val_mse']:.4f}"]  )
        t.add_row(["MAE" , f"{history['mae']:.4f}" , f"{history['val_mae']:.4f}"]  )
        t.add_row(["MAPE", f"{history['mape']:.4f}", f"{history['val_mape']:.4f}"] )
        t.add_row(["R2"  , f"{history['r2']:.4f}"  , f"{history['val_r2']:.4f}"]   )
        print("\n" + str(t) + "\n")

        return history
