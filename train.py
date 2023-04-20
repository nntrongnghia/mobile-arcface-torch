import argparse
import logging
import os
from datetime import datetime

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchsummary import summary
from lightning.pytorch.callbacks import LearningRateMonitor
from dataset import get_train_dataset
from dataset.lfw import LFWPair
from dataset.lfwbin import LFWBin
from dataset.mnist import MNISTVal
from lit_module import LitFaceRecognition
from utils import get_config, print_config
import torch
# seed_everything(42, workers=True)

log_root = logging.getLogger()
log_root.setLevel(logging.INFO)

def main(args):
    cfg = get_config(args.config)
    expe_name = f"{cfg.name}_{datetime.now().strftime('%y-%m-%d-%Hh%M')}"
    tb_logger = TensorBoardLogger(save_dir="./lightning_logs", name=expe_name)
    tb_logger.log_hyperparams(
        {key: value for key, value in cfg.items() if key != "model"}
    )
    print_config(cfg)
    train_dataset = get_train_dataset(**cfg)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8
    )
    if "debug" in cfg.name:
        val_dataset = MNISTVal("/home/nghia/dataset/mnist")
    else:
        val_dataset = LFWBin(cfg.bin_path)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=8)
    model = LitFaceRecognition(**cfg)
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(monitor="lfw_auroc", mode="max",
                        filename=f'{cfg.name}_best',
                        dirpath=tb_logger.log_dir)]
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        logger=tb_logger,
        gradient_clip_val=5,
        callbacks=callbacks,
        val_check_interval=cfg.val_check_interval,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint)
    print("hold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arcface Training in Pytorch")
    parser.add_argument(
        "--config", type=str, help="py config file", default="configs/ms1mv3_mbf.py"
    )
    parser.add_argument("--checkpoint", type=str, help="path to checkpoint to resume training", default=None)
    main(parser.parse_args())
