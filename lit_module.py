import os
from typing import Any, List
import torch
from torch import nn
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from loss.arcface import ArcFace
from torchmetrics import (
    AUROC,
    ROC,
    Accuracy,
    MaxMetric,
    MinMetric,
    SumMetric,
    MeanMetric,
)
import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image


class LitFaceRecognition(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        embedding_size,
        optimizer="sgd",
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        *arg,
        **kwargs,
    ) -> None:
        super().__init__()
        self.optimizer_name = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.backbone = model
        self.af = ArcFace(num_classes, embedding_size)
        self.auroc = AUROC("binary")
        self.accuracy = Accuracy("binary")
        self.max_cosine = MaxMetric()
        self.min_cosine = MinMetric()
        self.train_max_cosine = MaxMetric()
        self.train_min_cosine = MinMetric()
        self.sum_labels = SumMetric()
        self.pos_mean_score = MeanMetric("ignore")
        self.neg_mean_score = MeanMetric("ignore")
        self.train_loss_epoch = MeanMetric()

    def training_step(self, batch):
        imgs, labels = batch
        emb = self.backbone(imgs)
        loss, cosine = self.af(emb, labels, return_cosine=True)
        target_cosine = torch.gather(cosine, 1, labels[:, None])
        self.log("mean_target_cosine", target_cosine.mean())
        self.log("train_loss", loss)
        self.log("train_max_cosine", cosine.max())
        self.log("train_min_cosine", cosine.min())
        return loss

    def validation_step(self, batch, batch_idx):
        # imgs1,2 has shape (B, C, H, W)
        pair_imgs, is_same = batch
        self.sum_labels.update(is_same)
        # stack imgs to calculate embedding features
        pair_embs = self.backbone(torch.concat(pair_imgs))
        emb1, emb2 = torch.split(pair_embs, pair_imgs[0].shape[0])
        cosine = torch.cosine_similarity(emb1, emb2)
        score = (cosine + 1) / 2
        self.pos_mean_score.update(score[is_same.to(torch.bool)])
        self.neg_mean_score.update(score[~is_same.to(torch.bool)])
        self.max_cosine.update(cosine)
        self.min_cosine.update(cosine)
        self.auroc(score, is_same)
        self.accuracy(score, is_same)
        self.log("lfw_auroc", self.auroc, on_epoch=True)
        self.log("lfw_acc", self.accuracy, on_epoch=True)
        self.log("max_cosine", self.max_cosine, on_epoch=True)
        self.log("min_cosine", self.min_cosine, on_epoch=True)
        self.log("pos_mean_score", self.pos_mean_score, on_epoch=True)
        self.log("neg_mean_score", self.neg_mean_score, on_epoch=True)
        if batch_idx % (len(self.trainer.val_dataloaders[0]) // 20) == 0:
            self.log_val_images(pair_imgs, is_same, score, batch_idx)
        return score

    def configure_optimizers(self) -> Any:
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not implemented")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "lfw_auroc",
                "frequency": 1,
                "interval": "epoch"
            },
        }

    def log_val_images(self, pair_imgs, labels, scores, batch_idx) -> None:
        # Get tensorboard logger
        def torch2npimg(tensor):
            img = ((tensor * 0.5 + 0.5) * 255).type(torch.uint8)
            return np.asarray(F.to_pil_image(img))

        def make_img(imgs: List[torch.Tensor], label, score):
            img = cv2.hconcat([torch2npimg(im) for im in imgs])
            log_img = (
                np.zeros(
                    (img.shape[0] + 50, img.shape[1], img.shape[2]), dtype=np.uint8
                )
                + 255
            )
            log_img[: img.shape[0], : img.shape[1]] = img
            return cv2.putText(
                log_img,
                f"label {label} score {score:.3f}",
                (10, log_img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        if isinstance(self.trainer.logger, TensorBoardLogger):
            tb_logger = self.trainer.logger.experiment
        else:
            raise ValueError("TensorBoard Logger not found")
        img1s, img2s = pair_imgs
        img1s = torch.unbind(img1s)
        img2s = torch.unbind(img2s)
        log_pairs = [
            make_img([img1s[i], img2s[i]], labels[i], scores[i])
            for i in range(len(labels))
        ]
        tb_logger.add_image(
            f"val_image/{batch_idx}_0",
            log_pairs[0],
            self.global_step,
            dataformats="HWC",
        )
        print("hold")


# for debug
if __name__ == "__main__":
    from configs.ms1mv3_mbf import config

    lit = LitFaceRecognition(**config)
    print("hold")
