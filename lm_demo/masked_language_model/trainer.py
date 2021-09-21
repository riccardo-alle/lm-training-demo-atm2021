import logging
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

logger = logging.getLogger(__name__)


class TrainerBuilder:
    def __init__(self,
                 job_dir: str,
                 max_steps: int,
                 validate_every_n_steps: int,
                 checkpoint_every_n_steps: int,
                 ):
        self.job_dir = job_dir
        self.max_steps = max_steps
        self.validate_every_n_steps = validate_every_n_steps
        self.checkpoint_every_n_steps = checkpoint_every_n_steps

    def build(self):
        tensorboard_logger = TensorBoardLogger(save_dir=self._get_tensorboard_dir(),
                                               name="",
                                               version="",
                                               default_hp_metric=False)
        gpus = torch.cuda.device_count()
        precision = 16 if gpus > 0 else 32
        logger.info(f"{gpus} gpus found")

        return pl.Trainer(
            max_steps=self.max_steps,
            val_check_interval=self.validate_every_n_steps,
            gpus=gpus,
            accelerator="ddp" if gpus > 1 else None,
            precision=precision,
            logger=tensorboard_logger,
            progress_bar_refresh_rate=0,
            callbacks=[
                LearningRateMonitor(),
                ModelCheckpoint(dirpath=self.job_dir,
                                save_last=True,
                                every_n_train_steps=self.checkpoint_every_n_steps)
            ]
        )

    def _get_tensorboard_dir(self):
        return os.path.join(self.job_dir, "tensorboard")
