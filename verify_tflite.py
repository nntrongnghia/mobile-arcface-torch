import argparse
import cv2

import torch
from lit_module import LitFaceRecognition
from utils import get_config, get_inference_model
import numpy as np
import tflite_runtime.interpreter as tflite

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img_torch = torch.tensor(np.moveaxis(img, -1, 0))[None].cuda()
    return img, img_torch

def cosine_similarity(emb1, emb2):
    norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
    dot = np.dot(emb1, emb2)
    return dot / (norm1 * norm2)

def main(args):
    # -------------------------------- torch model ------------------------------- #
    cfg = get_config(args.config)
    backbone = (
        LitFaceRecognition.load_from_checkpoint(args.checkpoint, **cfg)
        .backbone.cuda()
        .eval()
    )
    model = get_inference_model(backbone).eval()
    img, img_torch = load_image(args.image)
    with torch.no_grad():
        torch_emb = model(img_torch.to(torch.float32)).cpu().numpy()
    # print(torch_emb)
    # ------------------------------- tflite model ------------------------------- #
    interpreter = tflite.Interpreter("pretrain_models/ms1mv3_arcface_mbf_best.tflite")
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Run tflite model
    interpreter.set_tensor(input_details[0]['index'], img_torch.cpu().numpy().astype(np.float32))
    interpreter.invoke()
    tflite_emb = interpreter.get_tensor(output_details[0]['index'])
    # print(tflite_emb)
    # ------------------- verify the difference between 2 embs ------------------- #
    diff = torch_emb - tflite_emb
    print("Diff norm:", np.linalg.norm(diff))
    print("Diff mean:", np.abs(diff).mean())
    print("Cosine:", cosine_similarity(torch_emb.flatten(), tflite_emb.flatten()))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--image", type=str, default="test_images/1.jpg")
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
        default="lightning_logs/ms1mv3_arcface_mbf_23-04-16-21h45/ms1mv3_arcface_mbf_best.ckpt",
    )
    main(parser.parse_args())
