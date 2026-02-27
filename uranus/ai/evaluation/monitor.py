__all__ = [
    "Monitor",
]

import os
import mlflow
import mpld3
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import atlas_mpl_style as ampl

from typing import Any, Dict, Optional, Union
from tqdm import tqdm as progress_bar
from torch.utils.data import DataLoader
from loguru import logger


class Monitor:
    """
    Class to calculate and store time series evaluation metrics and plots.
    """
    def __init__(self, name: str):
        """
        Args:
            name: Name of the evaluator.
        """
        self.name = name

    def __call__(self, ctx: Dict[str, Any], mlflow_active: bool = False) -> Dict[str, Any]:
        """
        Computes various time series metrics and plots curves.
        """
        # Ensure model is in eval mode
        model = ctx["pl_module"]
        model.eval()
        y_train_pred = []
        y_train_true = []
        y_val_pred   = []
        y_val_true   = []

        fold_dir = ctx["fold_dir"]
        basepath = os.path.join(fold_dir, "plots")
        os.makedirs(basepath, exist_ok=True)

        train_loader = ctx["train_loader"]
        val_loader   = ctx["val_loader"]
        model_history = ctx["history"]  

        # Fill train true and pred values
        with torch.no_grad():
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

        # Convert to numpy arrays for easier plotting
        y_train_true = np.array(y_train_true).flatten()
        y_train_pred = np.array(y_train_pred).flatten()
        y_val_true = np.array(y_val_true).flatten()
        y_val_pred = np.array(y_val_pred).flatten()


        ampl.use_atlas_style()

        print(model_history.keys())
        logger.info("Plotting loss history...")
        # Plotting
        fig, axes = plt.subplots(1, 1, figsize=(16, 6))
        # Loss plot
        if "loss" in model_history and "val_loss" in model_history:
            axes.plot(model_history["loss"], label='Train Loss', color='blue', linewidth=4)
            axes.plot(model_history["val_loss"], label='Val Loss', color='red', linewidth=4)

            if "best_epoch" in model_history:
                best_epoch = model_history["best_epoch"]    
                axes.plot(best_epoch, model_history["loss"][best_epoch], 'o', label='Best Epoch', color='black', markersize=20)
                
            axes.set_title(f"{self.name} - Loss History", fontsize=30)
            axes.set_xlabel("Epoch", fontsize=30, loc='right')
            axes.set_ylabel("Loss", fontsize=30, loc='top')
            plt.xlim(0, len(model_history["loss"]))
            plt.ylim(min(min(model_history["loss"]), min(model_history["val_loss"])), max(max(model_history["loss"]), max(model_history["val_loss"])))
            axes.legend()
            axes.grid(True, linestyle='--', alpha=0.5)
        else:
            axes.set_title("Loss history not available")

        plt.tight_layout()

       
        # Save plot to PDF
        save_path = os.path.join(basepath, "loss_history.pdf")
        logger.info(f"💾 Saving loss history plot to {save_path}")
        plt.savefig(save_path)

        if mlflow_active:

            #html_str = mpld3.fig_to_html(fig)
            # Save the HTML string to a file
            #with open(f"{basepath}/loss_history.html", "w") as f:
            #    f.write(html_str)
            mlflow.log_artifact(f"{basepath}/loss_history.pdf")

        # include distribution error plot.

        # include y vs y_pred plot with diagonal line.


        return ctx