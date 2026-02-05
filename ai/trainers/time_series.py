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
from ai.callbacks import ModelCheckpoint
from ai import setup_logs


class Trainer:
    """
    Trainer class responsible for training a time series model (torch model)
    using Cross Validation and PyTorch Lightning.
    """
    def __init__(
        self,
        model: pl.LightningModule,
        cv_strategy: Any,
        callbacks: List[pl.Callback]=[],
        evaluators: List[Any]=[],
        accelerator : str='auto',
        devices : str='auto',
        output_dir: str = "training_artifacts",
        output_filename: str = "model.pth",
        checkpoint_metric: str = "val_loss",
        checkpoint_mode: str = "min",
        checkpoint_filename: str = "last_model.pth",
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
        setup_logs("trainer")
        self.model = model
        self.cv_strategy = cv_strategy
        self.callbacks   = callbacks
        self.evaluators = evaluators
        self.accelerator = accelerator
        self.devices = devices
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.checkpoint_filename = checkpoint_filename
        self.checkpoint = ModelCheckpoint(
            dirpath=self.output_dir,
            monitor=checkpoint_metric,
            mode=checkpoint_mode,
            filename=self.checkpoint_filename,
            save_weights_only=True
        )
        self.callbacks.append(self.checkpoint)

    def fit(
        self, 
        dataset: Dataset, 
        num_epochs: int=1,
        batch_size: int=32,
        num_workers: int=1,
        specific_fold: Optional[int] = None,
        save_model: bool=True,
    ) -> Dict[int, nn.Module]:
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
        os.makedirs(self.output_dir, exist_ok=True)

        models = {}

        # Instantiate CV strategy if it is a class, otherwise use as is
        splitter = self.cv_strategy
        if isinstance(self.cv_strategy, type):
            splitter = self.cv_strategy()
        
        if not hasattr(splitter, "split"):
            raise ValueError("cv_strategy must implement a 'split' method (sklearn-like).")

        results = []
        total_splits = splitter.get_n_splits(dataset.index()) if hasattr(splitter, 'get_n_splits') else '?'

        # We assume dataset behaves like a list/array for Subset or has __getitem__
        for fold, (train_index, val_index) in enumerate(splitter.split(dataset.index())):
            
            fold_dir = os.path.join(self.output_dir, f"fold_{fold}")

            if specific_fold is not None and fold != specific_fold:
                continue

            logger.info(f"📂 Fold {fold}/{total_splits}")

            os.makedirs(fold_dir, exist_ok=True)


            logger.info(f"🚀 Starting Training for Fold {fold}/{total_splits}")
            # Fit preprocessor on training data if dataset supports it
            if hasattr(dataset, 'fit'):
                logger.info("🛠️ Fitting preprocessor on training data")
                dataset.fit(train_index)

     
            # 4. Setup Callbacks
            # Deepcopy callbacks to avoid state sharing and adjust paths
            fold_callbacks = []
            for cb in self.callbacks:
                cb_copy = copy.deepcopy(cb)
                fold_callbacks.append(cb_copy)
            
            # the last callback is the checkpoint
            fold_callbacks[-1].dirpath = fold_dir


            # Create Subsets
            train_dataset = torch.utils.data.Subset(dataset, train_index)
            val_dataset   = torch.utils.data.Subset(dataset, val_index)

            def custom_collate(batch):
                return batch
            train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)
            val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)

    
            # 5. Initialize Trainer
            trainer = pl.Trainer(
                max_epochs=num_epochs,
                callbacks=fold_callbacks,
                default_root_dir=fold_dir,
                enable_checkpointing=True,
                accelerator=self.accelerator,
                devices=self.devices,
                enable_progress_bar=True,
            )


            # Setup Model
            # Create a fresh copy of the model for this fold
            pl_module = copy.deepcopy(self.model)
          
            print(pl_module.history)
            checkpoint_path = os.path.join(fold_dir, self.checkpoint_filename)
            if os.path.exists(checkpoint_path):
                logger.info(f"🔄 Reloading model weights from {checkpoint_path}")
                ModelCheckpoint.load_checkpoint(checkpoint_path, trainer, pl_module)
            else:
                logger.info(f"🆕 No checkpoint found at {checkpoint_path}. Training from scratch.")

            print(pl_module.history)
            # 6. Fit
            trainer.fit(pl_module, train_loader, val_loader)
            history = pl_module.history
            # Evaluation and Summary require X and y separately typically.
            # Since we only have a Dataset/DataLoader yielding batches, we cannot easily call evaluators
            # that expect full X_val, y_val arrays unless we collate the whole val_dataset.
            # For now, we will skip the explicit evaluator calls that rely on (X, y) or need adaptation.
            # Use 'trainer.validate' or similar if metrics are logged in the model.
            # Placeholder for where evaluation logic would go if adapted to Datasets
            for evaluator in self.evaluators:
                history[evaluator.name] = evaluator(pl_module, train_loader, val_loader)


            print(history)

            if save_model:
                # 7. Save Final Model Weights and Metrics
                output_dict = {
                    "state_dict": pl_module.state_dict(),
                    "history"   : history,
                    "fold"      : fold,
                }
                file_path = os.path.join(fold_dir, self.output_filename)
                logger.info(f"💾 Saving model to {file_path}")
                torch.save(output_dict, file_path)
            
            models[fold] = pl_module
        
        return models
       




if __name__ == "__main__":
    
    import sys
    import os
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import TimeSeriesSplit
    from ai.preprocessing import StandardScale, create_window_dataframe, interpolate
    from ai.trainers.time_series import Trainer
    from ai.models.model_v1 import Model_v1
    from ai.evaluation import Summary
    from ai.loaders import DataLoader_v1
    import torch.nn as nn
    import torch
    import pytorch_lightning as pl
    import collections

    # Setup Components
    cv = TimeSeriesSplit(n_splits=4)

    col_names = {
        "PH (CBM) 1st Stage Poly Head Dev"     : "input_1",
        "PH (CBM) 1st Stage Press Rat Dev"     : "input_2",
        "PH (CBM) 1st Stage ActCompr Poly Eff" : "input_3",
        "PH (CBM) 1st Stg ActCompr Poly Head"  : "target",
    }

    feature_cols = ['input_1','input_2','input_3']
    output_col = ["target"]

    data_path = os.path.join(os.getenv("AI_DATA_PATH"), "compressor.csv")

    dataset = DataLoader_v1(data_path, 
                        window_size=10, 
                        col_names=col_names,
                        feature_cols=feature_cols, 
                        target_cols=output_col)

    print(dataset.inputs.shape)

    model = Model_v1(input_dim=dataset.inputs.shape[1], n_hidden=32)
    evaluators = [Summary(name="metrics")]
    params = {
        "batch_size": 32,
        "num_epochs": 5,
        "lr": 1e-3,
        "optimizer": "Adam"
    }

    trainer = Trainer(
        model=model,
        cv_strategy=cv,
        callbacks=[],
        evaluators=evaluators,
        params=params
    )

    trainer.fit(dataset, output_dir="output")