__all__ = [
    "Model_v1",
]

import pytorch_lightning as pl
import torch
import torch.nn as nn

from typing import Any

class Model_v1(pl.LightningModule):
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


        super().__init__()
        #self.save_hyperparameters()
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.input_features = dataset.input_features
        self.target_feature = dataset.target_feature[0]
        self.input_dim      = sum([dataset.data[feature_name].shape[1] for feature_name in self.input_features])
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x):
        return self.net(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

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
