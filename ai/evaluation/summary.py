from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score

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

    def __call__(self, model: nn.Module, X: Any, y: Any) -> Dict[str, float]:
        """
        Computes various time series metrics: MSE, MAE, MAPE, R2.
        
        Returns:
            Dictionary containing the calculated metrics.
        """
        # Ensure model is in eval mode
        model.eval()
        history = {}
        # Prepare data
        X_tensor = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
        y_true = np.array(y) if not isinstance(y, np.ndarray) else y
        
        # Predict
        with torch.no_grad():
            y_pred_tensor = model(X_tensor)
            y_pred = y_pred_tensor.cpu().numpy()
        
        # Flatten if necessary to match shapes (e.g. (N, 1) to (N,))
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        if y_true.ndim > 1 and y_true.shape[1] == 1:
            y_true = y_true.flatten()

        # Calculate metrics
        history["MSE"] = mean_squared_error(y_true, y_pred)
        history["RMSE"] = np.sqrt(history["MSE"])
        history["MAE"] = mean_absolute_error(y_true, y_pred)
        history["MAPE"] = mean_absolute_percentage_error(y_true, y_pred)
        history["R2"] = r2_score(y_true, y_pred)

        return history

    def get_predictions(self, model: nn.Module, X: Any) -> np.ndarray:
        """
        Returns predictions for the input X.
        """
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
        with torch.no_grad():
            y_pred_tensor = self.model(X_tensor)
        return y_pred_tensor.cpu().numpy()
