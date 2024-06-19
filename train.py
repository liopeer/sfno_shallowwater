import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
from rich import print
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from neuralop.datasets.spherical_swe import SphericalSWEDataset
import torch
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from model import FNOModel
import custom_resolvers

def get_trainval_datasets(data_config: DictConfig) -> Tuple[Dataset]:

    train_val_split = data_config.train_val_split

    train_ds = SphericalSWEDataset(
        dims         = data_config.dims,
        num_examples = int(data_config.num_examples * train_val_split), 
        device       = torch.device("cpu")
    )

    val_ds = SphericalSWEDataset(
        dims         = data_config.dims,
        num_examples = int(data_config.num_examples * (1 - train_val_split)), 
        device       = torch.device("cpu")
    )

    return train_ds, val_ds

def get_trainval_dataloaders(
        train_ds    : Dataset,
        val_ds      : Dataset,
        train_config: DictConfig
    ) -> Tuple[DataLoader]:

    train_dl = DataLoader(
        dataset    = train_ds,
        batch_size = train_config.batch_size,
        shuffle    = True
    )

    val_dl = DataLoader(
        dataset    = val_ds,
        batch_size = train_config.batch_size,
        shuffle    = False
    )

    return train_dl, val_dl

def setup_wandb_logger(total_config: DictConfig) -> WandbLogger:

    train_config = total_config.training

    logger = WandbLogger(
        entity    = total_config.wandb_entity,
        project   = total_config.wandb_project,
        config    = OmegaConf.to_container(total_config, resolve = True),
        mode      = "dryrun" if total_config.debug else None,
        save_dir  = total_config.wandb_dir,
        log_model = False,
        name      = total_config.wandb_run_name
    )

    return logger

def setup_trainer(total_config: DictConfig, logger: WandbLogger) -> L.Trainer:

    trainer = L.Trainer(
        accelerator            = total_config.accelerator,
        strategy               = total_config.strategy,
        devices                = total_config.devices,
        logger                 = logger,
        num_sanity_val_steps   = total_config.num_sanity_val_steps,
        precision              = total_config.precision,
        log_every_n_steps      = total_config.log_every_n_steps,
        val_check_interval     = total_config.training.val_check_interval,
        max_epochs             = total_config.training.max_epochs,
        profiler               = "simple",
        fast_dev_run           = total_config.debug,
        callbacks              = [
            ModelCheckpoint(
                every_n_epochs = 1,
                monitor        = total_config.training.ckpt_monitor_val,
                dirpath        = total_config.ckpt_dir,
                mode           = "min"
            )
        ]
    )
    
    return trainer

@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    # set up dataloaders
    train_ds, val_ds = get_trainval_datasets(cfg.data)
    train_dl, val_dl = get_trainval_dataloaders(train_ds, val_ds, cfg.training)

    # set up model
    model_type = hydra_cfg["runtime"]["choices"]["model"]
    model = FNOModel(model_type, cfg.model, cfg.training)

    # set up logger and trainer
    logger = setup_wandb_logger(cfg)
    trainer = setup_trainer(cfg, logger)

    # run training
    trainer.fit(model, train_dl, val_dl)

if __name__ == "__main__":
    main()