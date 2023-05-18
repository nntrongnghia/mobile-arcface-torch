import argparse
import os
import shutil
import sys
from typing import Any
from openvino.tools.mo import main as mo_main
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
from mltk.utils.shell_cmd import run_shell_cmd
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
    
    # Get the input tensor shape
    input_tensor = tf_rep.signatures[tf_rep.inputs[0]]
    input_shape = input_tensor.shape
    input_shape_str = '[' + ','.join([str(x) for x in input_shape]) + ']'


    openvino_out_dir = os.path.join(dir_path, "openvino")
    os.makedirs(openvino_out_dir, exist_ok=True)
    print(f'Generating openvino at: {openvino_out_dir}')
    cmd = [ 
        sys.executable, mo_main.__file__, 
        '--input_model', onnx_path,
        '--input_shape', input_shape_str,
        '--output_dir', openvino_out_dir,
        '--data_type', 'FP32'

    ]
    retcode, retmsg = run_shell_cmd(cmd,  outfile=sys.stdout)
    assert retcode == 0, 'Failed to do conversion' 
    
    openvino2tensorflow_out_dir = os.path.join(dir_path, 'openvino2tensorflow')
    openvino_xml_name = os.path.basename(onnx_path)[:-len('.onnx')] + '.xml'


    if os.name == 'nt':
        openvino2tensorflow_exe_cmd = [sys.executable, os.path.join(os.path.dirname(sys.executable), 'openvino2tensorflow')]
    else:
        openvino2tensorflow_exe_cmd = ['openvino2tensorflow']

    print(f'Generating openvino2tensorflow model at: {openvino2tensorflow_out_dir} ...')
    cmd = openvino2tensorflow_exe_cmd + [ 
        '--model_path', f'{openvino_out_dir}/{openvino_xml_name}',
        '--model_output_path', openvino2tensorflow_out_dir,
        '--output_saved_model',
        '--output_no_quant_float32_tflite'
    ]

    retcode, retmsg = run_shell_cmd(cmd)
    assert retcode == 0, retmsg
    print('done')
    
    # tf_path = os.path.join(dir_path, checkpoint_name + "_pb")
    # tf_rep.export_graph(tf_path)  # Export the model
    # # make a converter object from the saved tensorflow file
    # converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(tf_path)
    # # tell converter which type of optimization techniques to use
    # if args.optimize_tflite:
    #     converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #     checkpoint_name += "_opt"
    # # to view the best option for optimization read documentation of tflite about optimization
    # # go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional

    # # convert the model
    # tf_lite_model = converter.convert()
    # # save the converted model
    # open(os.path.join(dir_path, checkpoint_name + ".tflite"), "wb").write(tf_lite_model)


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
    parser.add_argument(
        "--hwc",
        action="store_true",
        help="convert input format to hwc",
    )
    parser.add_argument("--optimize-tflite", action="store_true")
    main(parser.parse_args())
