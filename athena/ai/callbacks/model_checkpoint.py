__all__ = ["ModelCheckpoint"]

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from typing import Optional, Dict, Any
from loguru import logger

class ModelCheckpoint(Callback):
    """
    A custom ModelCheckpoint callback that saves:
    - Model state dict (weights)
    - Full model object (for history/attributes)
    - Optimizer state (implicitly via pl.LightningModule state if included, but here we focus on manual saving)
    
    It saves the best model based on a monitored metric.
    """
    def __init__(
        self, 
        dirpath: str, 
        monitor: str = "val_loss", 
        mode: str = "min", 
        filename: str = "best_model.pt",
        save_weights_only: bool = False
    ):
        """
        Args:
            dirpath: Directory to save the checkpoint.
            monitor: Metric to monitor.
            mode: 'min' or 'max'.
            filename: Name of the checkpoint file.
            save_weights_only: If True, only saves state_dict. If False, saves full checkpoint dict including history.
        """
        super().__init__()
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.save_weights_only = save_weights_only
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        
        # Ensure directory exists
        os.makedirs(self.dirpath, exist_ok=True)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Called at the end of validation epoch to check metric and save if best.
        """
        # Retrieve metric
        current_score = trainer.callback_metrics.get(self.monitor)
        
        if current_score is None:
            # Maybe available as tensor?
            return

        if isinstance(current_score, torch.Tensor):
            current_score = current_score.item()
            
        # Check if improvement
        improved = False
        if self.mode == 'min':
            if current_score < self.best_score:
                improved = True
        else:
            if current_score > self.best_score:
                improved = True
                
        if improved:
            self.best_score = current_score
            self._save_checkpoint(trainer, pl_module)

    def _save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Saves the checkpoint.
        """
        filepath = os.path.join(self.dirpath, self.filename)
        
        # Prepare checkpoint dictionary
        checkpoint = {
            'model':{
                'epoch'       : trainer.current_epoch,
                'global_step' : trainer.global_step,
                'state_dict'  : pl_module.state_dict(),
                'monitor'     : self.monitor,
                'best_score'  : self.best_score,
            },
        }
        
        # Save history if available
        if hasattr(pl_module, 'history'):
            checkpoint['history'] = pl_module.history
            
        logger.info(f"⭐ Saving new best model to {filepath} with {self.monitor}={self.best_score:.4f}")
        torch.save(checkpoint, filepath)


    @staticmethod
    def load_checkpoint(filepath: str, trainer: pl.Trainer, pl_module: pl.LightningModule) -> Dict[str, Any]:
        """
        Loads the checkpoint from the given filepath and restores the module state, history and trainer state.
        """
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        
        # Restore state dict
        pl_module.load_state_dict(checkpoint['model']['state_dict'])
        
        # Restore trainer state
        trainer.fit_loop.epoch_progress.current.completed = checkpoint['model']['epoch']
        trainer.fit_loop.global_step = checkpoint['model']['global_step']

        # Restore history if available
        if 'history' in checkpoint:
            pl_module.history = checkpoint['history']
            
        logger.info(f"📥 Loaded checkpoint from {filepath} (epoch: {checkpoint['model']['epoch']}, {checkpoint['model']['monitor']}: {checkpoint['model']['best_score']:.4f})")
        return checkpoint



