import argparse
from copy import copy
import logging
import os
from typing import Tuple
import cv2

import torch
from torchsummary import summary
from torch.nn import functional as F
from dataset.lfw import LFWPair
from dataset.lfwbin import LFWBin
from lit_module import LitFaceRecognition
from utils import get_config, get_inference_model
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from numpy.random import choice
from lightning.pytorch import Trainer, seed_everything

seed_everything(42)
log_root = logging.getLogger()
log_root.setLevel(logging.INFO)


def get_cosine_score(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        assert emb1.shape == emb2.shape
        cosine = torch.cosine_similarity(emb1, emb2)
        score = (cosine + 1) / 2
    return score


class VNFaces(Dataset):
    def __init__(self, root: str):
        super().__init__()
        self.root = root
        image_dict = {}
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        for person in os.listdir(root):
            person_dir = os.path.join(root, person)
            if person not in image_dict:
                image_dict[person] = []
            for image in os.listdir(person_dir):
                image_dict[person].append(os.path.join(person_dir, image))
        # create 2 positive pairs per person = 20 positives pairs
        self.data = []
        for person, image_paths in image_dict.items():
            pairs = choice(image_paths, (2, 2), replace=False)
            self.data += [(tuple(pair), 1) for pair in pairs]
        # create 20 negative pairs
        for person in image_dict.keys():
            other_people = copy(list(image_dict.keys()))
            other_people.remove(person)
            person2 = choice(other_people)
            images1 = choice(image_dict[person], 2, replace=False)
            images2 = choice(image_dict[person2], 2, replace=False)
            self.data += [((img1, img2), 0) for img1, img2 in zip(images1, images2)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tuple[Tuple[np.ndarray, np.ndarray], int]:
        images, label = self.data[index]
        images = [cv2.imread(im) for im in images]
        return images, label


def main(args):
    cfg = get_config(args.config)
    # load model
    backbone = LitFaceRecognition.load_from_checkpoint(
        args.checkpoint, **cfg
    ).backbone.cuda().eval()
    model = get_inference_model(backbone).eval()
    logging.info(summary(model, (3, 112, 112)))
    dataset = VNFaces("./vnfaces")
    labels = []
    scores = []
    log_images = []
    with torch.no_grad():
        for images, label in tqdm(dataset):
            images = cv2.resize(images[0], (112, 112)), cv2.resize(images[1], (112, 112))
            img1 = torch.tensor(np.moveaxis(images[0], -1, 0))[None].cuda()
            img2 = torch.tensor(np.moveaxis(images[1], -1, 0))[None].cuda()
            # img1, img2 = dataset.transform(images[0])[None].cuda(), dataset.transform(images[1])[None].cuda()
            emb1, emb2 = model(img1), model(img2)
            score = get_cosine_score(emb1, emb2).detach().cpu().numpy()[0]
            labels.append(label)
            scores.append(score)
            himage = cv2.hconcat(images)
            log_img = (
                np.zeros((himage.shape[0] + 30, himage.shape[1], 3), dtype=np.uint8)
                + 255
            )
            log_img[: himage.shape[0], : himage.shape[1]] = himage
            log_img = cv2.putText(
                log_img,
                f"label {label} score {score:.3f}",
                (10, log_img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0) if label else (0, 0, 255),
                2,
            )
            log_images.append(log_img)
            # cv2.imshow("pair", log_img)
            # key = cv2.waitKey(0)
            # if key == 27: # ESC to exit
            #     break
    # threshold_range = np.arange(0, 1, 0.005)
    # acc_range = np.array([accuracy_score(labels, np.array(scores) > t) for t in threshold_range])
    # best_threshold = threshold_range[acc_range.argmax()]
    # best_acc = acc_range.max()
    acc = accuracy_score(labels, np.array(scores) > args.threshold)
    cv2.destroyAllWindows()
    log_img = cv2.vconcat(log_images)
    cv2.imwrite('vnfaces_test.jpg', log_img)
    print(f"acc = {acc} @ {args.threshold}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arcface Test with LFW dataset")
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
    parser.add_argument("--threshold", type=float, default=0.63)
    main(parser.parse_args())
