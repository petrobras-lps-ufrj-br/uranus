import copy
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from ai.preprocessing import PreProcessing



class Trainer:
    """
    Trainer class responsible for training a time series model (torch model)
    using Cross Validation and PyTorch Lightning.
    """
    def __init__(
        self,
        model: pl.LightningModule,
        cv_strategy: Any,
        callbacks: List[pl.Callback],
        params: Dict[str, Any],
        evaluators: List[Any],
    ):
        """
        Args:
            model: The torch.nn.Module to be trained.
            cv_strategy: The cross validation strategy class or instance.
            preprocessing: List of pre-processing input steps (must inherit from PreProcessing).
            callbacks: List of PyTorch Lightning callbacks (e.g., ModelCheckpoint, EarlyStopping).
            params: Dictionary containing training parameters (e.g., 'optimizer', 'num_epochs').
            criterion: Loss function module. Defaults to nn.MSELoss() if None.
        """
        self.model = model
        self.cv_strategy = cv_strategy
        self.callbacks = callbacks
        self.params = params
        self.evaluators = evaluators

    def fit(
        self, 
        dataset: Dataset, 
        output_dir: str = "training_artifacts", 
        specific_fold: Optional[int] = None
    ) -> List[Dict[str, float]]:
        """
        Executes the training loop with Cross Validation.

        Args:
            dataset: The input dataset (must be a torch Dataset or compatible).
                     If it has a 'data' attribute (like the custom DataLoader), it will be used for splitting.
            output_dir: Directory to save training artifacts (checkpoints, weights).
            specific_fold: Optional 1-based index of the fold to train. If None, trains all folds.

        Returns:
            List of metrics results for each fold.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Instantiate CV strategy if it is a class, otherwise use as is
        splitter = self.cv_strategy
        if isinstance(self.cv_strategy, type):
            splitter = self.cv_strategy()
        
        if not hasattr(splitter, "split"):
            raise ValueError("cv_strategy must implement a 'split' method (sklearn-like).")

        # Determine what to pass to split
        # If dataset has 'data' attribute, use it (e.g. for StratifiedKFold or time series split logic)
        # Otherwise, pass indices
        if hasattr(dataset, "data"):
            # If the dataset exposes the raw data (like the custom DataLoader)
            # We try to use it. Note: If dataset.data is a DF, split usually works on it.
            # If it is a tuple (X, y), we might need to unpack.
            split_args = (dataset.data,)
            # If the dataset happens to be X, y separated in .data, we might need adjustments
            # For now, we assume dataset.data is sufficient for the splitter or the splitter ignores content (KFold)
        else:
            # Fallback: create dummy X array of length
            split_args = (np.arange(len(dataset)),)

        fold = 0
        results = []
        print(split_args)
        total_splits = splitter.get_n_splits(*split_args) if hasattr(splitter, 'get_n_splits') else '?'

        # We assume dataset behaves like a list/array for Subset or has __getitem__
        
        for train_index, val_index in splitter.split(*split_args):
            fold += 1
            
            if specific_fold is not None and fold != specific_fold:
                continue

            fold_dir = os.path.join(output_dir, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)
            logger.info(f"Starting Training for Fold {fold}/{total_splits}")

            # Fit preprocessor on training data if dataset supports it
            if hasattr(dataset, 'fit'):
                dataset.fit(train_index)

            # Create Subsets
            train_dataset = torch.utils.data.Subset(dataset, train_index)
            val_dataset = torch.utils.data.Subset(dataset, val_index)

            batch_size = self.params.get("batch_size", 32)
            num_workers = self.params.get("num_workers", 0)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

            # 3. Setup Model
            # Create a fresh copy of the model for this fold
            pl_module = copy.deepcopy(self.model)
            # Inject params into the model instance
            pl_module.params = self.params


            # 4. Setup Callbacks
            # Deepcopy callbacks to avoid state sharing and adjust paths
            fold_callbacks = []
            for cb in self.callbacks:
                cb_copy = copy.deepcopy(cb)
                # If it's a ModelCheckpoint, ensure it saves to the fold directory
                if isinstance(cb_copy, pl.callbacks.ModelCheckpoint):
                    if not cb_copy.dirpath: 
                        # It's safer to override to organize by fold
                        cb_copy.dirpath = os.path.join(fold_dir, "checkpoints")
                fold_callbacks.append(cb_copy)
            
           
            # 5. Initialize Trainer
            trainer = pl.Trainer(
                max_epochs=self.params.get("num_epochs", 10),
                callbacks=fold_callbacks,
                default_root_dir=fold_dir,
                enable_checkpointing=True,
                accelerator=self.params.get("accelerator", "auto"),
                devices=self.params.get("devices", "auto"),
            )

            # 6. Fit
            trainer.fit(pl_module, train_loader, val_loader)

            history = {}

            # Evaluation and Summary require X and y separately typically.
            # Since we only have a Dataset/DataLoader yielding batches, we cannot easily call evaluators
            # that expect full X_val, y_val arrays unless we collate the whole val_dataset.
            # For now, we will skip the explicit evaluator calls that rely on (X, y) or need adaptation.
            # Use 'trainer.validate' or similar if metrics are logged in the model.
            
            # Placeholder for where evaluation logic would go if adapted to Datasets
            # for evaluator in self.evaluators:
            #     history[evaluator.name] = ...

            # 7. Save Final Model Weights and Metrics
            final_weights_path = os.path.join(fold_dir, "final_model.pt")
            
            # Skipping Summary for now as it requires X, y
            # eval_summary = Summary(pl_module, current_X_val, current_y_val)
            # fold_eval_metrics = eval_summary.calculate_metrics()
            # logger.info(f"Fold {fold} Metrics: {fold_eval_metrics}")
            
            fold_eval_metrics = {} # Empty for now

            output_dict = {
                "state_dict": pl_module.state_dict(),
                "history": history,
            }
            torch.save(output_dict, final_weights_path)
            
            # 8. Collect metrics
            results.append(fold_eval_metrics)

        return results
