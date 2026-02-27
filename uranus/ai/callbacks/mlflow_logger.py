__all__ = ["MLFlowLogger"]

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from loguru import logger

class MLFlowLogger(Callback):
    """
    A custom callback to log metrics to MLflow throughout the training process.
    """
    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: any, batch: any, batch_idx: int):
        """
        Log training loss at the end of each batch.
        """
        if mlflow.active_run():
            loss = trainer.callback_metrics.get("loss")
            if loss is not None:
                mlflow.log_metric("train_loss_step", loss.item(), step=trainer.global_step)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: any, batch: any, batch_idx: int, dataloader_idx: int = 0):
        """
        Log validation loss at the end of each batch.
        """
        if mlflow.active_run():
            val_loss = trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                mlflow.log_metric("val_loss_step", val_loss.item(), step=trainer.global_step)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Log training loss at the end of each epoch.
        """
        if mlflow.active_run():
            loss = trainer.callback_metrics.get("loss_epoch") or trainer.callback_metrics.get("loss")
            if loss is not None:
                mlflow.log_metric("train_loss_epoch", loss.item(), step=trainer.current_epoch)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Log validation loss at the end of each epoch.
        """
        if mlflow.active_run():
            val_loss = trainer.callback_metrics.get("val_loss_epoch") or trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                mlflow.log_metric("val_loss_epoch", val_loss.item(), step=trainer.current_epoch)
