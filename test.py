import argparse
import logging

import torch
from torchsummary import summary
from torch.nn import functional as F
from dataset.lfw import LFWPair
from dataset.lfwbin import LFWBin
from lit_module import LitFaceRecognition
from utils import get_config
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

log_root = logging.getLogger()
log_root.setLevel(logging.INFO)


def get_cosine_score(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        assert emb1.shape == emb2.shape
        cosine = torch.cosine_similarity(emb1, emb2)
        score = (cosine + 1) / 2
    return score

def get_euclidean_score(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        assert emb1.shape == emb2.shape
        emb1 = F.normalize(emb1)
        emb2 = F.normalize(emb2)
        distance = torch.norm(emb1 - emb2, dim=-1)
        score = 1 - distance / 2
    return score

def main(args):
    cfg = get_config(args.config)
    score_fn = get_euclidean_score if args.euclidean else get_cosine_score
    if args.checkpoint.endswith(".ckpt"):
        # load model
        model = LitFaceRecognition.load_from_checkpoint(
            args.checkpoint, **cfg
        ).backbone.cuda()
    elif args.checkpoint.endswith(".pt"):
        model = cfg.model.cuda()
        model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    logging.info(summary(model, (3, 112, 112)))
    # load data
    lfw = LFWBin(cfg.bin_path)
    with torch.no_grad():
        # Get embedding features
        emb1 = []
        emb2 = []
        emb1_flip = []
        emb2_flip = []
        logging.info("Calculate embedding features ...")
        for i in tqdm(range(len(lfw))):
            img1 = lfw.image1s[i][None].cuda()
            img2 = lfw.image2s[i][None].cuda()
            emb1.append(model(img1))
            emb2.append(model(img2))
            emb1_flip.append(model(img1.fliplr()))
            emb2_flip.append(model(img2.fliplr()))
        emb1 = torch.concat(emb1)
        emb2 = torch.concat(emb2)
        emb1_flip = torch.concat(emb1_flip)
        emb2_flip = torch.concat(emb2_flip)
        # Get similarity score
        # score_flip use Test-time augmentation with horizontal flip
        scores = score_fn(emb1, emb2).numpy(force=True)
        scores_flip = score_fn(emb1 + emb1_flip, emb2 + emb2_flip).numpy(force=True)
        print(scores.max(), scores.min())

    # Calculate accuracy
    labels = lfw.labels.numpy(force=True)
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    accuracies = np.zeros(10)
    accuracies_flip = np.zeros(10)
    chosen_thresholds = np.zeros(10)
    chosen_thresholds_flip = np.zeros(10)

    def evaluate(train_scores, train_labels, test_scores, test_labels):
        threshold_range = np.arange(0, 1, 0.002)  # threshold to be evaluated
        # Find best threshold in train set
        train_preds_list = [train_scores > t for t in threshold_range]
        train_acc_list = np.array(
            [accuracy_score(train_labels, preds) for preds in train_preds_list]
        )
        best_threshold = threshold_range[train_acc_list.argmax()]
        # print(train_acc_list.argmax(), train_acc_list.max())
        # Calculate test accuracy
        preds = test_scores > best_threshold
        acc = accuracy_score(test_labels, preds)
        # cheat = np.array([accuracy_score(test_labels, p) for p in [test_scores > t for t in threshold_range]])
        # print(cheat.max(), threshold_range[cheat.argmax()])
        return acc, best_threshold

    for i, (train_idx, test_idx) in enumerate(kf.split(list(range(len(lfw))))):
        # Normal inference
        train_labels = labels[train_idx]
        train_scores = scores[train_idx]
        test_labels = labels[test_idx]
        test_scores = scores[test_idx]
        accuracies[i], chosen_thresholds[i] = evaluate(
            train_scores, train_labels, test_scores, test_labels
        )
        # TTA inference
        train_scores_flip = scores_flip[train_idx]
        test_scores_flip = scores_flip[test_idx]
        accuracies_flip[i], chosen_thresholds_flip[i] = evaluate(
            train_scores_flip, train_labels, test_scores_flip, test_labels
        )
    logging.info(
        f"Accuracy: {accuracies.mean():.3f} +- {accuracies.std():.3f}, threshold: {chosen_thresholds.mean():.3f} +- {chosen_thresholds.std():.3f}"
    )
    logging.info(
        f"TTA Accuracy: {accuracies_flip.mean():.3f} +- {accuracies_flip.std():.3f}, threshold: {chosen_thresholds_flip.mean():.3f} +- {chosen_thresholds_flip.std():.3f}"
    )


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
        default="lightning_logs/ms1mv3_arcface_mbf_23-04-16-21h45/version_0/ms1mv3_arcface_mbf_best.ckpt",
    )
    parser.add_argument("--euclidean", action='store_true')
    main(parser.parse_args())
