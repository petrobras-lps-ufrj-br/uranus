__all__ = [
    "MLP_v1",
]

import pytorch_lightning as pl
import torch
import torch.nn as nn

from typing import Any

class MLP_v1(pl.LightningModule):
    """
    A simple MLP model with structure: Dense(n_hidden) -> ReLU -> Dropout(p) -> Dense(1).
    """
    def __init__(self, 
                 dataset: Any,
                 n_hidden: int = 64, 
                 dropout_p: float = 0.2,
                 criterion: nn.Module = None, 
                 optimizer: str='Adam',
                 lr: float=1e-3,
                 weight_decay: float=0,
                ):

        self.history = {'loss': [], 'val_loss': []}


        super().__init__()
        #self.save_hyperparameters()
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.input_features = dataset.input_features
        self.target_feature = dataset.target_feature
        self.input_dim      = sum([dataset.data[feature_name].shape[1] for feature_name in self.input_features])
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(n_hidden, 1)
        )

    def prepare_batch(self, batch):
        """
        Prepares a batch of data for training.

        Args:
            batch: A batch of data from the DataLoader.

        Returns:
            A tuple of (x, y) where x is the input data and y is the target data.

        the batch is a list of tuples (x, y) where x is a dictionary of features and y is a dictionary of targets. Each index 
        of the list is a sample. Each feature is a tensor of shape (1, n_inputs).
        """
        x = { feature_name: torch.stack([ row[0][feature_name] for row in batch ]) for feature_name in self.input_features }
        x = torch.cat([x[feature_name] for feature_name in self.input_features], dim=2) # [batch_size, 1, input_dim]
        x = torch.squeeze(x, dim=1) # [batch_size, input_dim]
        y = { feature_name: torch.stack([ row[1][feature_name] for row in batch ]) for feature_name in self.target_feature }
        y = torch.cat([y[feature_name] for feature_name in self.target_feature], dim=2) # [batch_size, 1, 1]
        y = torch.squeeze(y, dim=1) # [batch_size, 1]

        x = x.to(self.device)
        y = y.to(self.device)
        return x, y

    def forward(self, x):
        return self.net(x)

    def predict_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self(x)
        return y_hat


    def training_step(self, batch, batch_idx):
  
        x, y = self.prepare_batch(batch)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.prepare_batch(batch)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics.get("loss")
        if loss is not None:
            self.history["loss"].append(loss.item())

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
             self.history["val_loss"].append(val_loss.item())

    def configure_optimizers(self):
        # If optimizer is passed as an object/class, use it
        if isinstance(self.optimizer, type) or callable(self.optimizer):
            return self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif isinstance(self.optimizer, str): 
            if hasattr(torch.optim, self.optimizer):
                optimizer_cls = getattr(torch.optim, self.optimizer)
            else:
                raise ValueError(f"Optimizer {self.optimizer} not supported or found in torch.optim")
            return optimizer_cls(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported or found in torch.optim")
