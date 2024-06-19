from torch import Tensor
from jaxtyping import Float
import torch
import lightning as L
import torch.nn.functional as F
from typing import Iterable, Dict, Literal
from neuralop.models import SFNO, FNO
from omegaconf import DictConfig

class FNOModel(L.LightningModule):
    def __init__(
            self,
            model: Literal["sfno", "fno"],
            model_config: DictConfig,
            train_config: DictConfig
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = self.hparams["train_config"]["lr"]

        if model == "fno":
            self.model = FNO(**model_config)
        elif model == "sfno":
            self.model = SFNO(**model_config)
        else:
            raise NotImplementedError(f"No model {model} available.")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def forward(
            self,
            x: Float[Tensor, "b c h w"]
    ) -> Float[Tensor, "b c h w"]:
        return self.model(x)
    
    def training_step(
            self,
            batch: Dict[str, Tensor],
            batch_idx: int
    ) -> float:
        inputs = batch["x"]
        targets = batch["y"]
        preds = self(inputs)
        loss = F.mse_loss(preds, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(
            self,
            batch: Dict[str, Tensor],
            batch_idx: int
    ) -> None:
        inputs = batch["x"]
        targets = batch["y"]
        preds = self(inputs)
        loss = F.mse_loss(preds, targets)
        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)