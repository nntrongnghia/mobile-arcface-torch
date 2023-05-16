import argparse
import cv2

import torch
from lit_module import LitFaceRecognition
from utils import get_config, get_inference_model
import numpy as np



def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img_torch = torch.tensor(np.moveaxis(img, -1, 0))[None].cuda()
    return img, img_torch


def get_cosine_score(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        assert emb1.shape == emb2.shape
        cosine = torch.cosine_similarity(emb1, emb2)
        score = (cosine + 1) / 2
    return score


def main(args):
    cfg = get_config(args.config)
    img1, img1_torch = load_image(args.image1)
    img2, img2_torch = load_image(args.image2)
    img1, img2 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR), cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    # load model
    backbone = (
        LitFaceRecognition.load_from_checkpoint(args.checkpoint, **cfg)
        .backbone.cuda()
        .eval()
    )
    model = get_inference_model(backbone).eval()
    # logging.info(summary(model, (3, 112, 112)))
    with torch.no_grad():
        emb1, emb2 = model(img1_torch), model(img2_torch)
        score = get_cosine_score(emb1, emb2).detach().cpu().numpy()[0]
    himg = cv2.hconcat([img1, img2])
    log_img = np.zeros((himg.shape[0] + 30, himg.shape[1], 3), dtype=np.uint8) + 255
    log_img[: himg.shape[0], : himg.shape[1]] = himg
    log_img = cv2.putText(
        log_img,
        f"score {score:.3f}",
        (10, log_img.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
    )
    print('Embedding 1:')
    print(emb1)
    print('Embedding 2:')
    print(emb2)
    print(f"score: {score:.3f}")
    cv2.imshow("inference", log_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--image1", type=str, default="test_images/0.jpg")
    parser.add_argument("--image2", type=str, default="test_images/1.jpg")
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
