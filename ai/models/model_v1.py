import pytorch_lightning as pl
import torch
import torch.nn as nn

from typing import Any

class Model_v1(pl.LightningModule):
    """
    A simple MLP model with structure: Dense(n_hidden) -> ReLU -> Dropout(p) -> Dense(1).
    """
    def __init__(self, input_dim: int, n_hidden: int = 64, dropout_p: float = 0.2, params: dict = None, criterion: nn.Module = None, optimizer: Any = None):
        super().__init__()
        self.save_hyperparameters()
        self.params = params if params is not None else {}
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.optimizer = optimizer
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
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
        # Extract optimizer settings from params
        lr = self.params.get("lr", 1e-3)
        weight_decay = self.params.get("weight_decay", 0.0)

        # If optimizer is passed as an object/class, use it
        if self.optimizer is not None:
            # Check if it's a class or callable factory
            if isinstance(self.optimizer, type) or callable(self.optimizer):
                return self.optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)
            # If it's already an instance? Assuming factory/class as per standard pattern
            # Use as fallback if it's already configured? 
            # But normally we need to pass params(). 
            # If the user meant "instantiated", they might have passed a partial or class.
            return self.optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

        optimizer_name = self.params.get("optimizer", "Adam")

        if hasattr(torch.optim, optimizer_name):
            optimizer_cls = getattr(torch.optim, optimizer_name)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported or found in torch.optim")
        
        return optimizer_cls(self.parameters(), lr=lr, weight_decay=weight_decay)
