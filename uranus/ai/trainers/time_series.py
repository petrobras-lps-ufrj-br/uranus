import copy
import os
import mlflow
import requests
import multiprocessing
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from uranus.ai.preprocessing import PreProcessing
from uranus.ai.callbacks import      ModelCheckpoint, MLFlowLogger
from uranus import setup_logs


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
        experiment_name: str = "mlp_v1",
        use_mlflow: bool = True,
        num_workers : int = multiprocessing.cpu_count(),
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
        self.num_workers = num_workers
     
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow

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
        
        # Check if MLflow is requested and available
        mlflow_active = self.use_mlflow
        if mlflow_active:
            try:
                tracking_uri = mlflow.get_tracking_uri()
                if tracking_uri.startswith("http"):
                    requests.get(tracking_uri, timeout=2)
                mlflow.set_experiment(self.experiment_name)
                logger.info(f"🟢 MLflow tracking active: {tracking_uri}")
                self.callbacks.append(MLFlowLogger())

            except Exception as e:
                logger.warning(f"⚠️ MLflow server at {mlflow.get_tracking_uri()} is unreachable. Proceeding without MLflow. Error: {e}")
                mlflow_active = False

        models = {}

        # Instantiate CV strategy if it is a class, otherwise use as is
        splitter = self.cv_strategy
        if isinstance(self.cv_strategy, type):
            splitter = self.cv_strategy()
        
        if not hasattr(splitter, "split"):
            raise ValueError("cv_strategy must implement a 'split' method (sklearn-like).")

        total_splits = splitter.get_n_splits(dataset.index()) if hasattr(splitter, 'get_n_splits') else '?'

        # Start MLflow Parent Run
        #parent_run = None
        #if mlflow_active:
        #    parent_run = mlflow.start_run(run_name="Cross-Validation")
        #    mlflow.log_param("num_splits", total_splits)
        #    mlflow.log_param("model_type", type(self.model).__name__)
        #    mlflow.log_param("max_epochs", num_epochs)
        #    mlflow.log_param("batch_size", batch_size)

     
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
            
            if mlflow_active:
                logger.info(f"🚀 Starting MLflow run for Fold {fold}/{total_splits}")
                mlflow.start_run(run_name=f"Fold_{fold}")#, nested=True):
                mlflow.log_param("fold_index", fold)
            
            # Create Subsets
            train_dataset = torch.utils.data.Subset(dataset, train_index)
            val_dataset   = torch.utils.data.Subset(dataset, val_index)
            def custom_collate(batch):
                return batch
            train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate)
            val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate)
    
            # 5. Initialize Trainer
            trainer_fold = pl.Trainer(
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
        
            checkpoint_path = os.path.join(fold_dir, self.checkpoint_filename)
            if os.path.exists(checkpoint_path):
                logger.info(f"🔄 Reloading model weights from {checkpoint_path}")
                ModelCheckpoint.load_checkpoint(checkpoint_path, trainer_fold, pl_module)
            else:
                logger.info(f"🆕 No checkpoint found at {checkpoint_path}. Training from scratch.")
            # 6. Fit
            trainer_fold.fit(pl_module, train_loader, val_loader)
            history = pl_module.history
            
            # Evaluation
            ctx = {
                "history"       : history,
                "pl_module"     : pl_module,
                "train_loader"  : train_loader,
                "val_loader"    : val_loader,
                "fold_dir"      : fold_dir
            }
            
            for evaluator in self.evaluators:
                ctx = evaluator(ctx , mlflow_active=mlflow_active )
            
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
                #if mlflow_active:
                #    mlflow.log_artifact(file_path)
            
            if mlflow_active:
                mlflow.end_run()
            
            models[fold] = pl_module
               

   
        return models
       
