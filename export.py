import argparse
import os
import shutil
import sys
from typing import Any

import cv2
import numpy as np
import onnx
import torch
import tensorflow as tf
from PIL import Image
from torchvision import transforms
from torchvision.models import *
from torchsummary import summary
from lit_module import LitFaceRecognition
from onnx_tf.backend import prepare
from utils import get_config
from torch.nn import Sequential, Module

class ImageNorm(Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, img):
        img = img / 255
        img = (img - 0.5) / 0.5
        return img

def main(args):
    cfg = get_config(args.config)
    if args.checkpoint.endswith(".ckpt"):
        # load model
        backbone = LitFaceRecognition.load_from_checkpoint(args.checkpoint, **cfg).backbone
    elif args.checkpoint.endswith(".pt"):
        backbone = cfg.model
        backbone.load_state_dict(torch.load(args.checkpoint))
    backbone.cpu().eval()
    model = Sequential(ImageNorm(),backbone)
    sample_input = torch.rand((1, 3, 112, 112))
    checkpoint_name = os.path.basename(args.checkpoint).split(".")[0]
    dir_path = os.path.dirname(args.checkpoint)
    onnx_path = os.path.join(dir_path, checkpoint_name + ".onnx")
    torch.onnx.export(
        model,
        sample_input,
        onnx_path,
        opset_version=12,
        input_names=["image"],
        output_names=["emb_ft"],
    )
    # verify ONNX
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    # export to tflite
    tf_rep = prepare(onnx_model)  # Prepare TF representation
    tf_path = os.path.join(dir_path, checkpoint_name + "_pb")
    tf_rep.export_graph(tf_path)  # Export the model
    # make a converter object from the saved tensorflow file
    converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(tf_path)
    # tell converter which type of optimization techniques to use
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # to view the best option for optimization read documentation of tflite about optimization
    # go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional

    # convert the model
    tf_lite_model = converter.convert()
    # save the converted model
    open(os.path.join(dir_path, checkpoint_name + ".tflite"), "wb").write(tf_lite_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to TFlite")
    parser.add_argument(
        "--config",
        type=str,
        help="config file to get dataset path",
        default="configs/ms1mv3_mbf.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="checkpoint to test",
        default="lightning_logs/ms1mv3_arcface_mbf_23-04-16-21h45/version_0/ms1mv3_arcface_mbf_best.ckpt",
    )
    main(parser.parse_args())
